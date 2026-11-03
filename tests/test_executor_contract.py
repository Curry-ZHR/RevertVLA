"""Actual HOLD/replay adapters exercised against a single mocked publisher."""
from copy import deepcopy
import threading
import time
import pytest
from Reverse.core import ReverseEpisode
from Reverse.rolling_gate import guard_step
from Reverse.integration import replay

MASK = {s: {"enabled": True, "gripper": True} for s in ("left", "right")}
MASK.update(lift=False, waist=False, head=False, speed=False)


class Service:
    def __init__(self, pass_after=None):
        self.enabled = True
        self.lock = threading.RLock()
        self.episode = ReverseEpisode(["Aojiru", "Cola"])
        self.episode.begin("Grasp Aojiru with the left hand.", MASK)
        self.episode.anchor({"left": [.02] * 7 + [0.], "right": [.1] * 7 + [1.5], "body": {}, "speed": {}})
        self.phase = "forward"
        self.events = []
        self.pass_after = pass_after
        self.hold_control_steps = 0

    def event(self, kind, **value):
        self.events.append((kind, value))

    def gate(self, steps, stop_event):
        if stop_event.is_set():
            raise InterruptedError("Stopped")
        state = "PASS" if self.phase == "hold" and self.pass_after is not None and self.hold_control_steps >= self.pass_after else "WRONG"
        self.episode.votes["left"].clear()
        now = time.monotonic_ns()
        for i in range(10):
            self.episode.ingest("left", "Aojiru", self.episode.generation, i, now - (10 - i) * 1000, state)
        return self.episode.gate(steps)


class Executor:
    def __init__(self, service):
        self.service = service
        self._stop_event = threading.Event()
        self.sent = []
        self.retired = False
        self.stop_on_send = None

    def _safe_hold(self, **kwargs):
        return {"chassis": True, "body": True}

    def _dispatch_body(self, *args, **kwargs):
        return {"ok": True}

    def _publish_step(self, step, *args, **kwargs):
        assert self.retired
        command = {s: {"positions": step[s][:7], "gripper": step[s][7]} for s in ("left", "right")}
        self.service.episode.command_allowed(command, holding=bool(step.get("reverse_hold")))
        self.sent.append(deepcopy(step))
        if self.stop_on_send == len(self.sent):
            self._stop_event.set()
        return {"ok": True, "action": command}


def run_hold(service, executor):
    candidate = deepcopy(service.episode.last_sent)
    candidate["left"] = [.8] * 7 + [1.5]
    def check_running():
        if executor._stop_event.is_set():
            raise InterruptedError("Stopped")
    def retire():
        executor.retired = True
    return guard_step(executor, service, candidate, chunk_id="partial", joint_speed=90.,
                      interval=.0001, mask=MASK, snapshot={}, fixed_targets={},
                      check_running=check_running, before_hold=retire)


@pytest.mark.parametrize("pass_after", [None, 1, 3, 20])
def test_twenty_complete_open_cycles_precede_release_or_replay(pass_after):
    service = Service(pass_after)
    executor = Executor(service)
    result = run_hold(service, executor)
    assert len(executor.sent) == (20 if pass_after is None else 21)
    assert all(step["left"] == [.02] * 7 + [0.] for step in executor.sent[:20])
    assert all(step["right"] == [.1] * 7 + [1.5] for step in executor.sent)
    assert result["state"] == ("WRONG" if pass_after is None else "REPLAN")
    if pass_after is not None:
        assert executor.sent[-1]["left"] == [.02] * 7 + [1.5]


def test_stop_during_hold_cannot_publish_close():
    service = Service(1)
    executor = Executor(service)
    executor.stop_on_send = 4
    with pytest.raises(InterruptedError):
        run_hold(service, executor)
    assert len(executor.sent) == 4
    assert all(step["left"][7] == 0. for step in executor.sent)


def test_timeout_replay_preserves_other_hand_and_open_lock():
    service = Service()
    executor = Executor(service)
    for index in range(3):
        step = deepcopy(service.episode.last_sent)
        step["left"] = [.02 + index * .01] * 7 + [0.]
        service.episode.record(str(index), step, {s: {"positions": step[s][:7], "gripper": step[s][7]} for s in ("left", "right")})
    assert run_hold(service, executor)["state"] == "WRONG"
    last, count = replay(executor, service, joint_speed=90., interval=.0001, mask=MASK,
                         snapshot={}, fixed_targets={}, check_running=lambda: None)
    assert count > 0 and last["left"][7] == 0.
    assert all(step["left"][7] == 0. for step in executor.sent)
    assert all(step["right"] == [.1] * 7 + [1.5] for step in executor.sent)
    assert service.episode.mask_active
