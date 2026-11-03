"""Adapters for Robot Station's existing executor. Never publishes through a second owner."""
from copy import deepcopy
import math
import time
from .core import ReverseError


def capture_body(manager, mask):
    if not any(mask.get(section) for section in ("lift", "waist", "head")):
        return None
    from modules.ros_io import get_ros_reader
    from modules.inference.observation import BODY_STATE_KEY
    value = get_ros_reader().read(BODY_STATE_KEY, timeout_sec=0.)
    if not value.get("ok") or value.get("stale") or not 0 <= value.get("ageMs", 10**9) <= 150:
        raise ReverseError("回滚无法记录新鲜机身姿态，本步未发送")
    data = value.get("data") or {}
    names, positions = data.get("name", []), data.get("position", [])
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ReverseError("机身姿态格式错误")
    actual = dict(zip(names, positions))
    required = ["daogui_joint", "body_pitch_joint", "body_yaw_joint", "neck_yaw_joint", "neck_pitch_joint"]
    if any(name not in actual or not isinstance(actual[name], (int,float)) or not math.isfinite(actual[name]) for name in required):
        raise ReverseError("机身回滚记录缺少有效关节")
    # Unlike arms, body goals run in a separate action gateway. Store measured
    # cache poses, never claim an unexecuted body plan was actually dispatched.
    return {name: float(actual[name]) for name in required}


def record_step(manager, service, chunk_id, step, result, *, body=None, control_step=None):
    if not service.enabled or not result.get("ok"):
        return
    with service.lock:
        previous = service.episode.history[-1]["chunk_id"] if service.episode.history else None
        service.episode.record(chunk_id, step, result["action"], body=body, control_step=control_step)
        if previous != chunk_id:
            service.event("HISTORY_CHUNK", chunk_id=chunk_id, history_chunks=len(service.episode.history),
                          basis="successful_sdk_arm_targets_and_sampled_body_feedback")


def replay(manager, service, *, joint_speed, interval, mask, snapshot, fixed_targets,
           check_running, on_publish=None, trajectory_config=None):
    with service.lock:
        steps = service.episode.replay_plan()
        service.phase = "rollback"
        anchor = deepcopy(service.episode.last_sent)
    # Joint rollback has no inverse-velocity/path contract. Keep the chassis
    # stopped instead of replaying forward velocities from historical frames.
    if mask.get("speed"):
        stopped = manager._zero_chassis()
        if not stopped.get("ok"):
            raise ReverseError(stopped.get("detail", "底盘停止失败"))
        steps = deepcopy(steps)
        for step in steps:
            step["speed"] = (0.0, 0.0)
    if steps and trajectory_config is not None:
        from modules.inference import trajectory
        # Keep the existing velocity/acceleration/jerk and feedback guards.
        plan = trajectory.retime(steps, anchor=anchor, mask=mask, fixed_targets=fixed_targets,
            speed_deg_s=joint_speed, limits=trajectory_config["trajectoryLimits"],
            source_chunk="client-reverse", source_origin=manager._rtc_control_step)
        steps = plan.with_elapsed_prefix(0)
    if steps:
        body = manager._dispatch_body(steps, mask, queue_if_busy=True, step_interval_sec=interval,
                                      stop_on_error=True)
        if not body.get("ok"):
            raise ReverseError(body.get("detail", "机身回滚轨迹接入失败"))
    last = anchor
    for index, step in enumerate(steps):
        check_running()
        started = time.perf_counter()
        result = manager._publish_step(step, joint_speed, mask, snapshot, fixed_targets,
                                      interval_sec=interval, trajectory_config=trajectory_config)
        if not result.get("ok"):
            raise ReverseError(result.get("detail", "回滚步骤发送失败"))
        last = deepcopy(step)
        for side in ("left", "right"):
            sent = result.get("action", {}).get(side, {})
            if "positions" in sent:
                last[side][:7] = list(sent["positions"])
            if sent.get("gripper") is not None:
                last[side][7] = sent["gripper"]
        release = step.get("reverse_release") or step.get("trajectory", {}).get("upstreamTarget", {}).get("reverse_release")
        if release:
            service.event("ROLLBACK_RELEASE", hands=release, replay_index=index)
        if on_publish is not None:
            on_publish(last, result)
        # No stop acknowledgement, settling wait, or status HTTP query.
        if manager._stop_event.wait(max(0., interval-(time.perf_counter()-started))):
            raise InterruptedError("Rollback cancelled by stop")
    with service.lock:
        service.episode.replay_finished(last)
        service.phase = "forward"
    return last, len(steps)
