"""Resident perception workers plus the client-owned rollback state machine.

The model workers have no ROS imports and no robot command interface. Frames are
read through Robot Station's existing camera reader only while rollback is on.
"""
import atexit
import base64
from collections import deque
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
import queue
import select
import subprocess
import sys
import threading
import time
from .core import ReverseEpisode, ReverseError
from .siglip_delivery import CATALOG_RELATIVE, CHECKPOINT_STEP, EXPECTED_OBJECTS, EXPECTED_PROMPTS, load_catalog

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger("robot_station.reverse")


class ModelProcess:
    def __init__(self, kind, python, log, *, timeout=120):
        self.kind = kind
        self.lock = threading.Lock()
        self.timeout = timeout
        self.pending = b""
        env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                   OPENBLAS_NUM_THREADS="2", TF_CPP_MIN_LOG_LEVEL="2",
                   TF_NUM_INTEROP_THREADS="1", TF_NUM_INTRAOP_THREADS="1",
                   XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION="0.18",
                   YOLO_AUTOINSTALL="false", YOLO_CONFIG_DIR=str(ROOT / "runtime/yolo_config"))
        env["JAX_PLATFORMS"] = "cuda,cpu"
        # Big Vision and custom YOLO code are isolated from the ROS interpreter.
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "vendor/big_vision"),
            str(ROOT / "vendor/retail_yolo_runtime"), str(ROOT / "scripts"), str(ROOT.parent)])
        if not Path(python).is_file():
            raise ReverseError(f"回滚 {kind} 环境缺失，请先安装 Reverse 独立依赖")
        self.process = subprocess.Popen([str(python), "-u", str(ROOT / "model_worker.py"), "--kind", kind],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, env=env, cwd=ROOT,
            start_new_session=True, bufsize=0)
        self.ready = None

    def wait_ready(self, timeout=300):
        try:
            self.ready = self._read(timeout)
            if not self.ready.get("ready"):
                raise ReverseError(f"{self.kind} warmup failed: {self.ready.get('error', 'no readiness acknowledgement')}")
        except Exception:
            self.close()
            raise

    def alive(self):
        return self.process.poll() is None

    def _read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ReverseError(f"{self.kind} 返回超时")
            ready, _, _ = select.select([self.process.stdout], [], [], remaining)
            if not ready:
                raise ReverseError(f"{self.kind} 返回超时")
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise ReverseError(f"{self.kind} worker 已退出")
            self.pending += block
            if len(self.pending) > 16 * 1024 * 1024:
                raise ReverseError("模型返回过大")
        line, self.pending = self.pending.split(b"\n", 1)
        value = json.loads(line)
        if value.get("error"):
            raise ReverseError(f"{self.kind}: {value['error']}")
        return value

    def request(self, payload, timeout=1.0):
        # Large image writes run off the RTC control thread for SigLIP; YOLO
        # is invoked only by an observation worker. Keep the pipe bounded.
        data = (json.dumps(payload, separators=(",", ":")) + "\n").encode()
        if len(data) > 12 * 1024 * 1024:
            raise ReverseError("模型输入图像过大")
        with self.lock:
            if not self.alive():
                raise ReverseError(f"{self.kind} worker 已退出")
            deadline = time.monotonic() + timeout
            offset = 0
            fd = self.process.stdin.fileno()
            try:
                os.set_blocking(fd, False)
                while offset < len(data):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([], [fd], [], max(0, remaining))[1]:
                        raise ReverseError(f"{self.kind} 输入超时")
                    try:
                        offset += os.write(fd, data[offset:offset+65536])
                    except BlockingIOError:
                        continue
                return self._read(max(.001, deadline-time.monotonic()))
            except Exception as exc:
                # A timed-out response cannot be mistaken for the next frame.
                self.close()
                if isinstance(exc, ReverseError):
                    raise
                raise ReverseError(f"{self.kind} worker 通信失败：{exc}") from exc

    def close(self):
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


class ReverseRuntime:
    def __init__(self, *, camera_reader=None, process_factory=ModelProcess, root=ROOT):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.enabled = False
        self.ready = False
        self.warming = False
        self.error = ""
        self.phase = "not_started"
        self.camera_reader = camera_reader
        self.process_factory = process_factory
        self.siglip = self.yolo = None
        self._closed = threading.Event()
        self._changed = threading.Event()
        self._warmup_thread = None
        self._score_thread = None
        self._logs = []
        self._events = queue.Queue(maxsize=1024)
        self._event_thread = None
        self._last_scored = {}
        catalog = self.root / CATALOG_RELATIVE
        self.names = load_catalog(catalog) if catalog.is_file() else []
        self.episode = ReverseEpisode(self.names, self.event)
        self.metrics = {"scoredFrames": 0, "scoreHz": 0.0, "maskedObservations": 0}

    def event(self, kind, **details):
        value = dict(kind=kind, monotonic_ns=time.monotonic_ns(), **details)
        try:
            self._events.put_nowait(value)
        except queue.Full:
            LOGGER.error("Reverse event queue full; dropped diagnostic event %s", kind)

    def _write_events(self, path):
        with path.open("a", buffering=1) as stream:
            while not self._closed.is_set() or not self._events.empty():
                try:
                    stream.write(json.dumps(self._events.get(timeout=.2), ensure_ascii=False) + "\n")
                except queue.Empty:
                    pass

    def warmup_background(self):
        with self.lock:
            if self._warmup_thread is not None:
                return
            self.warming = True
            self.phase = "warming"
            directory = self.root / "logs" / (time.strftime("%Y%m%d_%H%M%S") + "_" + str(os.getpid()))
            directory.mkdir(parents=True, exist_ok=True)
            self.log_directory = directory
            self._event_thread = threading.Thread(target=self._write_events, args=(directory/"events.jsonl",), daemon=True)
            self._event_thread.start()
            self._warmup_thread = threading.Thread(target=self._warmup, name="reverse-warmup", daemon=True)
            self._warmup_thread.start()

    def _warmup(self):
        try:
            siglip_python = os.environ.get("REVERSE_SIGLIP_PYTHON", str(self.root / ".venv/siglip/bin/python"))
            yolo_python = os.environ.get("REVERSE_YOLO_PYTHON", str(self.root / ".venv/yolo/bin/python"))
            for kind, python in (("siglip", siglip_python), ("yolo", yolo_python)):
                if self._closed.is_set():
                    return
                log = (self.log_directory / (kind + ".log")).open("ab", buffering=0)
                self._logs.append(log)
                worker = self.process_factory(kind, python, log, timeout=300)
                with self.lock:
                    setattr(self, kind, worker)
                worker.wait_ready(timeout=300)
                if self._closed.is_set():
                    worker.close()
                    return
            with self.lock:
                self.ready = True
                self.phase = "idle"
                self.error = ""
            self.event("REVERSE_READY", siglip=self.siglip.ready, yolo=self.yolo.ready)
            self._score_thread = threading.Thread(target=self._score_loop, name="reverse-siglip-15hz", daemon=True)
            self._score_thread.start()
        except Exception as exc:
            with self.lock:
                self.ready = False
                self.phase = "error"
                self.error = str(exc)
            self.event("WARMUP_FAILED", reason=str(exc))
            for worker in (self.siglip, self.yolo):
                if worker is not None:
                    worker.close()
            LOGGER.exception("Reverse warmup failed; normal inference remains available")
        finally:
            with self.lock:
                self.warming = False

    def status(self):
        with self.lock:
            alive = self.ready and all(worker is not None and worker.alive() for worker in (self.siglip, self.yolo))
            return dict(available=len(self.names) == EXPECTED_OBJECTS, enabled=self.enabled, ready=bool(alive), busy=self.warming,
                        phase=self.phase, detail=self.error, historyLimit=self.episode.history.maxlen,
                        historyChunks=len(self.episode.history), replayCount=self.episode.replays,
                        activeHand=self.episode.active_hand, rollbackHand=self.episode.rollback_hand,
                        historyPolicy="unified_last_8", rollbackPolicy="wrong_hand_only_hold_other",
                        jointMotionThresholdDegrees=self.episode.motion_threshold_deg,
                        lastJointSpanDegrees=dict(self.episode.last_motion),
                        maskActive=self.episode.mask_active, lastGate=deepcopy(self.episode.last_gate),
                        siglipHz=15, siglipStep=CHECKPOINT_STEP, siglipObjects=len(self.names),
                        siglipPrompts=EXPECTED_PROMPTS, metrics=dict(self.metrics),
                        supportedTrajectoryModes=["legacy_clip"],
                        logDirectory=str(getattr(self, "log_directory", "")), version="client-reverse-7of10-hold20-history8-siglip362-openlock-v2",
                        holdPolicy="open_full_20_control_steps", replayOpenGripperLock=True,
                        gateWindow=10, gateMinimumPass=7, gateNewestPass=2, holdLimit=20,
                        holdControlSteps=getattr(self, "hold_control_steps", 0))

    def set_enabled(self, enabled):
        # The manager owns the operation lock while calling this method.
        with self.lock:
            if enabled and not self.status()["ready"]:
                raise ReverseError(self.error or "回滚模块仍在预热，尚不能开启")
            self.enabled = bool(enabled)
            if enabled or self.status()["ready"]:
                self.error = ""
            self.episode.end()
            self.phase = "enabled" if enabled else "idle"
            self._last_scored.clear()
            self._changed.set()
        self.event("REVERSE_ENABLED" if enabled else "REVERSE_DISABLED")

    def begin(self, prompt, mask, *, reset=True, targets=None):
        with self.lock:
            if not self.enabled:
                return
            if not self.status()["ready"]:
                raise ReverseError("回滚模型不再就绪，不能开始受保护的推理")
            if targets is None and not reset and self.episode.active and prompt == self.episode.prompt:
                targets = dict(self.episode.targets)
            self.episode.begin(prompt, mask, reset=reset, targets=targets)
            self.phase = "forward"
            self._changed.set()

    def end(self):
        with self.lock:
            self.episode.end()
            if self.ready:
                self.phase = "enabled" if self.enabled else "idle"

    def observe(self, observation):
        with self.lock:
            if not self.enabled or not self.episode.active:
                return observation
            state = observation["observation"]["state"]
            if isinstance(state, dict):
                body = {"daogui_joint": state["lift"]["height"], "body_pitch_joint": state["waist"]["pitch"],
                        "body_yaw_joint": state["waist"]["yaw"], "neck_yaw_joint": state["head"]["yaw"],
                        "neck_pitch_joint": state["head"]["pitch"]}
                anchor = {side: [state[side][f"joint{i}"] for i in range(1,8)] + [state[side]["gripper"]]
                          for side in ("left", "right")}
                self.episode.anchor(dict(anchor, body=body, speed={"linear": 0., "angular": 0.}))
            elif isinstance(state, (list, tuple)) and len(state) == 23:
                body_names = ["daogui_joint", "body_pitch_joint", "body_yaw_joint", "neck_yaw_joint", "neck_pitch_joint"]
                self.episode.anchor(dict(left=list(state[:8]), right=list(state[8:16]),
                    body=dict(zip(body_names,state[16:21])), speed={"linear":0.,"angular":0.}))
            else:
                raise ReverseError("回滚不识别该模型的观测状态格式，需要适配器")
            masked = self.episode.mask_active
            generation = self.episode.generation
            targets = list(self.episode.targets.values())
        if not masked:
            return observation
        if any(w.get("camera") in ("cam_high", "camera_h") for w in getattr(observation, "camera_warnings", [])):
            raise ReverseError("回滚重试需要新鲜头部图像，当前图像不可用")
        head = getattr(observation, "metadata", {}).get("sources", {}).get("camera_h/color")
        if head is not None and (not head.get("ok") or not 0 <= (head.get("ageMs") or 0) <= 500):
            raise ReverseError("回滚遮罩的头部图像已过期")
        encoded = observation["observation"]["images"]["cam_high"]
        value = self.yolo.request(dict(op="mask", image=encoded, targets=targets), timeout=3.)
        with self.lock:
            if generation != self.episode.generation or not self.enabled:
                raise ReverseError("观测来自已退出的回滚阶段，禁止执行旧结果")
            result = deepcopy(observation)
            result["observation"]["images"]["cam_high"] = value["image"]
            self.metrics["maskedObservations"] += 1
            self.event("YOLO_MASK", generation=generation, **value["mask"])
            return result

    def gate(self, steps, stop_event):
        if not self.enabled:
            return {"state": "ORDINARY"}
        if stop_event.is_set():
            raise InterruptedError("Stopped before rollback gate")
        with self.lock:
            if not self.status()["ready"] or self.error:
                raise ReverseError(self.error or "回滚模型已退出")
            return self.episode.gate(steps)

    def _score_loop(self):
        scored = deque()
        while not self._closed.is_set():
            started = time.monotonic()
            try:
                with self.lock:
                    active = self.ready and self.enabled and self.episode.active
                    generation = self.episode.generation
                    targets = {s: t for s,t in self.episode.targets.items() if s not in self.episode.closed_hands}
                if active:
                    if self.camera_reader is None:
                        from modules.ros_io import get_camera_image_reader
                        self.camera_reader = get_camera_image_reader()
                    frames = []
                    for side, target in targets.items():
                        frame = self.camera_reader.read_frame("camera_l" if side == "left" else "camera_r", "color", timeout_sec=0.)
                        age = frame.get("ageMs", 10**9)
                        sequence = frame.get("sequence")
                        received = frame.get("receivedMonotonicNs")
                        if (not frame.get("ok") or not isinstance(sequence, int) or not isinstance(received, int)
                                or not 0 <= age <= 500 or not frame.get("body")):
                            continue
                        key = (generation, sequence)
                        if self._last_scored.get(side) == key:
                            continue
                        frames.append(dict(side=side, target=target, sequence=sequence, received_ns=received,
                                           image=base64.b64encode(frame["body"]).decode("ascii")))
                    if frames:
                        value = self.siglip.request(dict(op="score", generation=generation, frames=frames), timeout=1.)
                        with self.lock:
                            for decision in value.get("decisions", []):
                                item = dict(decision)
                                side = item.pop("side")
                                self.episode.ingest(side, generation=generation, **item)
                                self._last_scored[side] = (generation, item["sequence"])
                                self.metrics["scoredFrames"] += 1
                            scored.append(time.monotonic())
                            now = time.monotonic()
                            while scored and scored[0] < now-5:
                                scored.popleft()
                            self.metrics["scoreHz"] = len(scored) / 5.
            except Exception as exc:
                with self.lock:
                    self.error = str(exc)
                    self.ready = False
                    self.phase = "error"
                self.event("PERCEPTION_FAILED", reason=str(exc))
                LOGGER.exception("Reverse perception failed")
            self._closed.wait(max(.001, 1./15 - (time.monotonic()-started)))

    def close(self):
        self._closed.set()
        for worker in (self.siglip, self.yolo):
            if worker is not None:
                worker.close()
        if self._score_thread is not None:
            self._score_thread.join(timeout=2)
        if self._event_thread is not None:
            self._event_thread.join(timeout=2)
        for log in self._logs:
            log.close()


_instance = None
_instance_lock = threading.Lock()


def get_reverse_runtime():
    global _instance
    with _instance_lock:
        if _instance is None:
            _instance = ReverseRuntime()
            atexit.register(_instance.close)
        return _instance


def startup():
    service = get_reverse_runtime()
    if os.environ.get("ROBOT_STATION_REVERSE_AUTOWARMUP", "0") == "1":
        if os.environ.get("ROBOT_STATION_REVERSE_VARIANT") != "7of10-hold20-history8-siglip362":
            raise ReverseError("必须通过Reverse362入口配置，禁止混用旧版启动配置")
        service.warmup_background()
    return service
