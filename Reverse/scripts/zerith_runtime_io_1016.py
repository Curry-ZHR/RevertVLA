"""Startup progress and bounded, asynchronous diagnostic log output.

Inference threads enqueue LogRecords without formatting or disk I/O. A separate
supervisor drains process output and batches disk writes on its writer thread.
Camera snapshots and SigLIP result exchange remain live IPC, not diagnostic logs.
"""
from __future__ import annotations

import argparse
import atexit
import codecs
from contextlib import contextmanager
import logging
from pathlib import Path
import queue
import selectors
import signal
import subprocess
import sys
import threading
import time

STATUS_PREFIXES = ("[WARMUP_PLAN]", "[MODEL_LOAD]", "[TEXT_CACHE]", "[WARMUP]",
                   "[ALL_MODELS_WARMUP_DONE]", "[SERVICE_READY]", "[WARMUP_ONLY_DONE]", "[GRASP_GATE]", "[CLIENT_EXECUTION]", "[RTC_DIAGNOSTICS]", "[RTC_COMPAT]")


def startup_status(message):
    """Used only during startup, never in a production inference request."""
    print(message, flush=True)


class AsyncLogHandler(logging.Handler):
    def __init__(self, stream=None, capacity=4096):
        super().__init__()
        self.stream = sys.stderr if stream is None else stream
        self.records = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self._stop = threading.Event()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        self.thread = threading.Thread(target=self._run, name="zerith-log-format", daemon=True)
        self.thread.start()

    def emit(self, record):
        try:
            # Keep exception formatting and output on the worker thread.
            self.records.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        reported_drops = 0
        while not self._stop.is_set() or not self.records.empty():
            try:
                record = self.records.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.stream.write(self.format(record) + "\n")
                if self.dropped != reported_drops:
                    self.stream.write(f"[LOG_QUEUE_DROPPED] records={self.dropped}\n")
                    reported_drops = self.dropped
            except (OSError, ValueError):
                pass
            finally:
                self.records.task_done()

    def close(self):
        self._stop.set()
        if threading.current_thread() is not self.thread:
            self.thread.join(timeout=2)
        super().close()


def install_async_logging(level=logging.INFO):
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    handler = AsyncLogHandler()
    root.addHandler(handler)
    root.setLevel(level)
    atexit.register(handler.close)
    return handler


@contextmanager
def vla_warmup_progress():
    """Observe the shared warmup helper without changing its RNG/inference code."""
    class Observer(logging.Handler):
        def emit(self, record):
            if record.funcName != "warmup_policy":
                return
            if record.msg == "Startup warmup %d/2 begin: phase=%s seed=%d":
                index, phase, seed = record.args
                startup_status(f"[WARMUP] VLA {index}/2 START phase={phase} seed={seed}")
            elif record.msg == "Startup warmup %d/2 complete: phase=%s elapsed=%.3fs; output discarded":
                index, phase, elapsed = record.args
                startup_status(f"[WARMUP] VLA {index}/2 DONE phase={phase} elapsed={elapsed:.3f}s")
    observer = Observer()
    root = logging.getLogger()
    root.addHandler(observer)
    try:
        yield
    finally:
        root.removeHandler(observer)
        observer.close()


class BatchedFileSink:
    """Bounded output queue. Slow disks can drop logs, never stall the reader."""
    def __init__(self, path, *, flush_seconds=1.0, capacity=2048, batch_bytes=1024 * 1024):
        self.path = Path(path)
        self.flush_seconds, self.batch_bytes = flush_seconds, batch_bytes
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped_bytes = 0
        self.error = None
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, name="zerith-log-disk", daemon=True)
        self.thread.start()

    def offer(self, data):
        try:
            self.queue.put_nowait(data)
        except queue.Full:
            self.dropped_bytes += len(data)

    def _write_batch(self, stream, data):
        stream.write(data)
        stream.flush()  # Batch flush; no per-record flush or fsync.

    def _run(self):
        buffer = bytearray()
        reported_drops = 0
        try:
            with self.path.open("ab", buffering=1024 * 1024) as stream:
                deadline = time.monotonic() + self.flush_seconds
                while not self._stop.is_set() or not self.queue.empty():
                    try:
                        data = self.queue.get(timeout=max(0.001, min(0.05, deadline - time.monotonic())))
                        buffer.extend(data)
                        self.queue.task_done()
                    except queue.Empty:
                        pass
                    if self.dropped_bytes != reported_drops:
                        buffer.extend(f"\n[LOG_BUFFER_DROPPED] bytes={self.dropped_bytes}\n".encode())
                        reported_drops = self.dropped_bytes
                    if buffer and (len(buffer) >= self.batch_bytes or time.monotonic() >= deadline):
                        self._write_batch(stream, buffer)
                        buffer.clear()
                        deadline = time.monotonic() + self.flush_seconds
                    elif not buffer and time.monotonic() >= deadline:
                        deadline = time.monotonic() + self.flush_seconds
                if buffer:
                    self._write_batch(stream, buffer)
        except OSError as exc:
            self.error = str(exc)
            print(f"[LOG_WRITE_FAILED] {self.path}: {exc}", file=sys.stderr, flush=True)

    def close(self):
        self._stop.set()
        self.thread.join(timeout=3)


def supervise(command, log_file, *, console_status=False, flush_seconds=1.0):
    sink = BatchedFileSink(log_file, flush_seconds=flush_seconds)
    child = None
    stopping = None
    saw_ready = False
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pending = ""
    previous_handlers = {}

    def stop(signum, _frame):
        nonlocal stopping
        if stopping is None:
            stopping = time.monotonic()
            if child is not None and child.poll() is None:
                child.send_signal(signum)

    def show_lines(text, final=False):
        nonlocal pending, saw_ready
        pending += text
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            if line.startswith("[SERVICE_READY]"):
                saw_ready = True
            if console_status and line.startswith(STATUS_PREFIXES):
                print(line, flush=True)
        if len(pending) > 65536:
            pending = pending[-65536:]
        if final and pending and console_status and pending.startswith(STATUS_PREFIXES):
            print(pending, flush=True)

    try:
        for signum in (signal.SIGTERM, signal.SIGINT):
            previous_handlers[signum] = signal.signal(signum, stop)
        # Children only write pipes. This supervisor alone opens diagnostic files.
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, bufsize=0)
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout, selectors.EVENT_READ)
            while selector.get_map():
                if stopping is not None and time.monotonic() - stopping > 4 and child.poll() is None:
                    child.kill()
                for key, _ in selector.select(timeout=0.1):
                    data = key.fileobj.read(16384)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    sink.offer(data)
                    if console_status:
                        show_lines(decoder.decode(data))
        if console_status:
            show_lines(decoder.decode(b"", final=True), final=True)
        result = child.wait()
        if result and stopping is None:
            label = "SERVICE_EXIT" if saw_ready else "STARTUP_FAILED"
            print(f"[{label}] exit={result}; details={log_file}", file=sys.stderr, flush=True)
        return result if result >= 0 else 128 - result
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child.stdout is not None:
                child.stdout.close()
        sink.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--console-status", action="store_true")
    parser.add_argument("--flush-seconds", type=float, default=1.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not 0.05 <= args.flush_seconds <= 60:
        parser.error("A child command and a flush interval in [0.05, 60] are required")
    raise SystemExit(supervise(command, args.log_file, console_status=args.console_status,
                              flush_seconds=args.flush_seconds))


if __name__ == "__main__":
    main()
