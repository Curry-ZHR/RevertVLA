"""Bounded preclose HOLD using the executor's existing guarded publisher."""
from copy import deepcopy
import time
from .core import ReverseError
from .gate_policy import HOLD_CYCLES


def guard_step(manager, service, step, *, chunk_id, joint_speed, interval, mask,
               snapshot, fixed_targets, check_running, before_hold=None):
    """Return ORDINARY/PASS, or WRONG/REPLAN after retiring the old VLA suffix.

    HOLD and release publications have their own count and history source. They
    never increment a VLA cursor or execute a cached arm target.
    """
    if not service.enabled:
        return {"state": "ORDINARY"}
    check_running()
    result = service.gate([step], manager._stop_event)
    if result["state"] != "WAIT":
        return result
    if before_hold is not None:
        before_hold()
    check_running()
    with service.lock:
        anchor = deepcopy(service.episode.last_sent)
        if anchor is None:
            raise ReverseError("HOLD缺少最后成功下发的目标，禁止猜测保持位置")
        hands = set(result["hands"])
        service.hold_control_steps = 0
        service.phase = "hold"
    # Cancel any queued body/chassis work too; arms continue through _publish_step.
    stopped = manager._safe_hold(hold_arms=False, signal_stop=False, preserve_reference=True)
    if any(stopped.get(k) is not True for k in ("chassis", "body")):
        raise ReverseError("HOLD无法停止机身或底盘计划")
    held = deepcopy(anchor)
    held.pop("trajectory", None)
    held["reverse_hold"] = True
    held["speed"] = {"linear": 0., "angular": 0.}
    for side in hands:
        held[side][7] = 0.
    service.event("HOLD_BEGIN", hands=sorted(hands), limit=HOLD_CYCLES,
                  basis="last_successfully_sent_targets", remote_compute_cancelled=False)

    def publish(action, source):
        check_running()
        value = manager._publish_step(action, joint_speed, mask, snapshot, fixed_targets,
                                      interval_sec=interval)
        if not value.get("ok"):
            raise ReverseError(value.get("detail", "HOLD command failed"))
        with service.lock:
            service.episode.record(chunk_id, action, value.get("action", {}), source=source)
        return value

    # A rejected prospective close starts one uninterrupted open observation
    # window. A transient PASS cannot pulse the gripper closed mid-window.
    for tick in range(HOLD_CYCLES):
        check_running()
        started = time.perf_counter()
        publish(held, "hold")
        with service.lock:
            service.hold_control_steps = tick + 1
        service.event("HOLD_SENT", control_cycle=tick+1, hands=sorted(hands),
                      gripper_targets={side: 0. for side in sorted(hands)})
        if manager._stop_event.wait(max(0., interval - (time.perf_counter()-started))):
            raise InterruptedError("Stopped during bounded HOLD")

    check_running()
    result = service.gate([step], manager._stop_event)
    if result["state"] == "PASS":
        closing = deepcopy(held)
        closing.pop("reverse_hold", None)
        closing["reverse_gate_release"] = True
        for side in hands:
            closing[side][7] = step[side][7]
        publish(closing, "gate_release")
        with service.lock:
            service.phase = "forward"
        service.event("HOLD_RELEASE", hands=sorted(hands), hold_cycles=HOLD_CYCLES,
                      gate=result, next="fresh_observation_without_old_suffix")
        return {**result, "state": "REPLAN", "hold_cycles": HOLD_CYCLES}
    with service.lock:
        service.phase = "forward"
        return service.episode.timeout(result)
