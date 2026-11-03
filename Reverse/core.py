"""Pure control bookkeeping, independent of ROS, model runtimes and VLA weights."""
from collections import deque
from .gate_policy import evaluate_window, HISTORY_CHUNKS, WINDOW_SIZE
from copy import deepcopy
import math
import re
import time
import uuid


class ReverseError(RuntimeError):
    """A bounded, explicit rollback failure; never an instruction to reconnect."""


def canonical(name):
    key = re.sub(r"[^a-z0-9]", "", str(name).replace("Yili Ginger Ale", "Ginger Ale").casefold())
    # The YOLO v42 class table spells this same SKU "Miranda Peach".
    return "mirindapeach" if key == "mirandapeach" else key


def prompt_targets(prompt, names):
    catalog = {canonical(name): name for name in names}
    matches = re.findall(r"(?:grasp|pick up)\s+(.+?)\s+with\s+(?:the\s+)?(left|right)\s+hand", prompt, re.I)
    targets = {}
    for target, side in matches:
        key = canonical(target)
        if key not in catalog:
            raise ReverseError(f"回滚不识别目标商品：{target}")
        side = side.lower()
        if side in targets:
            raise ReverseError("一次推理每只手只能有一个回滚核验目标")
        targets[side] = catalog[key]
    if not targets:
        raise ReverseError("回滚需要明确商品和手别，例如 Grasp Aojiru with the left hand.")
    return targets


def checked_step(step):
    result = deepcopy(step)
    for side in ("left", "right"):
        values = result.get(side)
        if not isinstance(values, (list, tuple)) or len(values) != 8:
            raise ReverseError("回滚要求标准七关节加夹爪的客户端动作格式")
        if any((v is None and i != 7) or (v is not None and
               (isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v))) for i,v in enumerate(values)):
            raise ReverseError("回滚动作包含非法数值")
        result[side] = list(values)
    return result


class ReverseEpisode:
    """Called under the runtime lock; history only advances after successful SDK sends."""
    def __init__(self, names, event=None):
        self.names = list(names)
        self.event = event or (lambda *_args, **_kwargs: None)
        self.history = deque(maxlen=HISTORY_CHUNKS)
        self.votes = {side: deque(maxlen=WINDOW_SIZE) for side in ("left", "right")}
        self.closed_hands = set()
        self.approved_hands = set()
        self.rearm_ns = {side: 0 for side in ("left", "right")}
        self.retry_hand = None
        self.targets = {}
        self.prompt = ""
        self.generation = str(uuid.uuid4())
        self.active = False
        self.mask_active = False
        self.last_sent = None
        self.phase = "idle"
        self.last_gate = None
        self.replays = 0
        self.empty_retries = 0
        self.motion_threshold_deg = 3.0
        self.active_hand = None
        self.rollback_hand = None
        self.last_motion = {}

    def _reset_hand_tracking(self):
        self.active_hand = None
        self.rollback_hand = None
        self.last_motion = {}

    @staticmethod
    def joint_spans(points):
        # SDK arm positions are radians. Gripper is intentionally excluded.
        return {side: math.degrees(max(max(p[side][j] for p in points) -
                                      min(p[side][j] for p in points) for j in range(7)))
                for side in ("left", "right")} if points else {}

    def _gate_hands(self, upcoming):
        # Select the wrist for the currently proposed RTC action, without task
        # phases or history truncation. A held closed gripper is not a new task.
        pending = set(self.targets)
        if len(pending) == 1:
            self.active_hand = next(iter(pending))
            return pending
        points = ([self.last_sent] if self.last_sent else []) + list(upcoming)
        spans = self.joint_spans(points)
        moving = {side for side in pending
                  if spans and spans[side] > self.motion_threshold_deg + 1e-9}
        if moving:
            largest = max(spans[side] for side in moving)
            selected = {side for side in moving if abs(spans[side] - largest) <= 1e-9}
            if len(selected) > 1 and self.active_hand in selected:
                selected = {self.active_hand}
        elif self.active_hand in pending:
            selected = {self.active_hand}
        else:
            closing = {side for side in pending if sum(v[side][7] is not None and
                       v[side][7] >= .75 for v in upcoming) >= 5}
            selected = closing if len(closing) == 1 else pending
        if len(selected) == 1:
            self.active_hand = next(iter(selected))
        return selected

    def begin(self, prompt, mask, *, reset=True, targets=None):
        if targets is None:
            targets = prompt_targets(prompt, self.names)
        else:
            catalog = {canonical(name): name for name in self.names}
            if (not isinstance(targets, dict) or not targets or set(targets)-{"left","right"}
                    or any(not isinstance(value,str) or canonical(value) not in catalog for value in targets.values())):
                raise ReverseError("reverseTargets 必须明确指定已知商品和 left/right 手别")
            targets = {side:catalog[canonical(value)] for side,value in targets.items()}
        for side in targets:
            if not mask[side]["enabled"] or not mask[side]["gripper"]:
                raise ReverseError(f"回滚目标 {side} 的机械臂和夹爪必须参与执行")
        if reset or targets != self.targets or not self.active:
            self.history.clear()
            self._reset_hand_tracking()
            self.last_sent = None
            self.mask_active = False
            self.replays = 0
            self.empty_retries = 0
            self.generation = str(uuid.uuid4())
            self.clear_votes()
            self.closed_hands.clear()
            self.approved_hands.clear()
            self.rearm_ns = {side: 0 for side in ("left", "right")}
            self.retry_hand = None
        self.targets = targets
        self.prompt = prompt
        self.active = True
        self.phase = "forward"
        return self.generation

    def clear_votes(self):
        for values in self.votes.values():
            values.clear()

    def ingest(self, side, target, generation, sequence, received_ns, state, **details):
        if (not self.active or side not in self.targets or side in self.closed_hands
                or generation != self.generation
                or canonical(target) != canonical(self.targets[side])):
            return
        if state not in ("PASS", "WRONG", "UNCERTAIN"):
            return
        if (not isinstance(sequence, int) or isinstance(sequence, bool)
                or not isinstance(received_ns, int) or isinstance(received_ns, bool)):
            return
        if received_ns < self.rearm_ns[side] or received_ns > time.monotonic_ns():
            return
        values = self.votes[side]
        if values and (sequence <= values[-1]["sequence"] or received_ns <= values[-1]["received_ns"]):
            return
        values.append(dict(sequence=sequence, received_ns=received_ns, state=state, **details))

    def gate(self, steps, *, now_ns=None):
        # The executor calls this on the NEXT step, before submitting any close.
        # A chunk preview must never authorize or terminate a retry mask.
        upcoming = steps[:1]
        if not upcoming:
            return {"state": "ORDINARY"}
        self._gate_hands(upcoming)
        closing = {side for side in self.targets if side not in self.closed_hands
                   and upcoming[0][side][7] is not None and upcoming[0][side][7] >= .75}
        if not closing:
            return {"state": "ORDINARY"}
        now_ns = time.monotonic_ns() if now_ns is None else now_ns
        hands = {side: evaluate_window(list(self.votes[side]), now_ns) for side in sorted(closing)}
        passed = all(v["passed"] for v in hands.values())
        self.approved_hands.difference_update(closing)
        if passed:
            self.approved_hands.update(closing)
        result = dict(state="PASS" if passed else "WAIT", hands=hands, checked_steps=1)
        self.last_gate = result
        return result

    def timeout(self, result):
        self.approved_hands.difference_update(result.get("hands", {}))
        self.last_gate = {**result, "state": "WRONG", "reason": "20_hold_control_cycles_exhausted"}
        failed = [s for s,v in result.get("hands", {}).items() if not v.get("passed")]
        if len(failed) == 1:
            self.active_hand = failed[0]
        self.event("GATE_WRONG", **self.last_gate)
        return self.last_gate

    def command_allowed(self, sent, *, holding=False):
        # Final publisher boundary: observers must not introduce an unapproved close.
        if self.phase == "rollback":
            return
        for side in self.targets:
            cmd = sent.get(side) or {}
            grip = cmd.get("gripper")
            if grip is not None and grip >= .75 and side not in self.closed_hands:
                if holding or side not in self.approved_hands:
                    raise ReverseError(f"{side} 闭爪未通过7/10门控，禁止发送")
                window = evaluate_window(list(self.votes[side]), time.monotonic_ns())
                if not window["passed"]:
                    self.approved_hands.discard(side)
                    raise ReverseError(f"{side} 闭爪前评分已过期，禁止发送")

    def _record_grippers(self, actual, sent):
        for side in self.targets:
            grip = (sent.get(side) or {}).get("gripper")
            if grip is None:
                continue
            if grip >= .75:
                self.closed_hands.add(side)
                self.approved_hands.discard(side)
                self.votes[side].clear()
                if self.mask_active and side == self.retry_hand:
                    self.mask_active = False
                    self.event("MASK_END", hand=side, reason="approved_close_successfully_sent")
                    self.retry_hand = None
            elif grip < .75 and side in self.closed_hands:
                self.closed_hands.discard(side)
                self.approved_hands.discard(side)
                self.votes[side].clear()
                self.rearm_ns[side] = time.monotonic_ns()
                self.event("SIGLIP_REARM", hand=side)

    def anchor(self, step):
        if self.last_sent is None:
            if any(step[side][7] is None for side in ("left","right")):
                raise ReverseError("执行起点缺少夹爪状态")
            self.last_sent = checked_step(step)
            self.closed_hands.update(s for s in self.targets if step[s][7] >= .75)

    def record(self, chunk_id, requested, sent, *, body=None, control_step=None, source="vla"):
        if not self.active or self.phase != "forward":
            return
        if source != "hold":
            self.empty_retries = 0
        actual = checked_step(requested)
        actual.pop("reverse_hold", None)
        actual.pop("reverse_gate_release", None)
        actual.pop("reverse_open_hands", None)
        actual.pop("reverse_release", None)
        for side in ("left", "right"):
            output = sent.get(side) or {}
            if "positions" in output:
                actual[side][:7] = list(output["positions"])
            elif self.last_sent is not None:
                actual[side][:7] = list(self.last_sent[side][:7])
            if output.get("gripper") is not None:
                actual[side][7] = float(output["gripper"])
            elif self.last_sent is not None:
                actual[side][7] = self.last_sent[side][7]
        if body is not None:
            actual["body"] = dict(body)
        if self.last_sent is None:
            raise ReverseError("缺少执行起点，禁止构造猜测的回滚历史")
        if not self.history or self.history[-1]["chunk_id"] != chunk_id:
            self.history.append({"chunk_id": chunk_id, "before": deepcopy(self.last_sent), "steps": []})
        self.history[-1]["steps"].append({"action": deepcopy(actual), "control_step": control_step, "source": source})
        self._record_grippers(actual, sent)
        self.last_sent = actual
        group = self.history[-1]
        self.last_motion = self.joint_spans([group["before"]] + [v["action"] for v in group["steps"]])

    def _select_rollback_hand(self):
        pending = set(self.targets)
        failed = set()
        if self.last_gate and self.last_gate.get("state") == "WRONG":
            failed = {side for side, vote in self.last_gate.get("hands", {}).items()
                      if side in pending and not vote.get("passed", vote.get("pass_count", 0) >= 7)}
        if len(failed) == 1:
            return next(iter(failed))
        choices = failed or pending
        if self.active_hand in choices:
            return self.active_hand
        if len(choices) == 1:
            return next(iter(choices))
        if choices and self.history:
            group = self.history[-1]
            spans = self.joint_spans([group["before"]] + [v["action"] for v in group["steps"]])
            ranked = sorted(choices, key=lambda side: spans[side], reverse=True)
            if spans[ranked[0]] > 1e-9 and spans[ranked[0]] > spans[ranked[1]] + 1e-9:
                return ranked[0]
        raise ReverseError("无法确定本次回滚手别，禁止同时倒放双臂；请检查门控手别和活动臂记录")

    def replay_plan(self):
        self.phase = "rollback"
        self.mask_active = True
        self.approved_hands.clear()
        groups = list(self.history)
        self.rollback_hand = self._select_rollback_hand() if groups else None
        self.retry_hand = self.rollback_hand or self.active_hand
        held = {side: list(self.last_sent[side]) for side in ("left", "right")
                if side != self.rollback_hand} if self.rollback_hand else {}
        # A rejected close was already opened by HOLD. Historical close targets
        # must never re-close that hand while returning along its joint path.
        open_gripper_hands = {self.rollback_hand} if (self.rollback_hand is not None
            and self.last_sent[self.rollback_hand][7] < .75) else set()
        self.event("ROLLBACK_BEGIN", source_chunks=[g["chunk_id"] for g in groups],
                   executed_counts=[len(g["steps"]) for g in groups], history_limit=self.history.maxlen,
                   rollback_hand=self.rollback_hand, held_hands=sorted(held), hold_targets=deepcopy(held),
                   open_gripper_hands=sorted(open_gripper_hands))
        if not groups or not any(v.get("source", "vla") != "hold" for g in groups for v in g["steps"]):
            self.empty_retries += 1
            if self.empty_retries > 3:
                raise ReverseError("连续遮罩重试仍识别到错误目标，且没有已执行历史可回滚")
            if not groups:
                return []
        points = [deepcopy(groups[0]["before"])]
        points.extend(deepcopy(item["action"]) for group in groups for item in group["steps"])
        closing = {}
        for side in (self.rollback_hand,):
            transitions = [i for i in range(1, len(points)) if points[i-1][side][7] < .75 <= points[i][side][7]]
            if transitions:
                closing[side] = transitions[-1]
            else:
                earliest = next((i for i, point in enumerate(points) if point[side][7] >= .75), None)
                if earliest is not None:
                    closing[side] = earliest
        self.event("ROLLBACK_CLOSE_POINTS", close_points=closing)
        released = set(open_gripper_hands)
        plan = []
        for index in range(len(points)-1, -1, -1):
            point = checked_step(points[index])
            point.pop("trajectory", None)
            point.pop("reverse_release", None)
            point["reverse_open_hands"] = sorted(released)
            for side, values in held.items():
                point[side] = list(values)
            point["speed"] = {"linear": 0.0, "angular": 0.0}
            for side in released:
                point[side][7] = 0.0
            plan.append(point)
            arriving = [side for side, close_index in closing.items() if index == close_index and side not in released]
            if arriving:
                opened = deepcopy(point)
                for side in arriving:
                    opened[side][7] = 0.0
                    released.add(side)
                opened["reverse_release"] = list(arriving)
                opened["reverse_open_hands"] = sorted(released)
                plan.append(opened)
        return plan

    def replay_finished(self, last_sent=None):
        self.replays += 1
        self.history.clear()
        self.last_sent = checked_step(last_sent) if last_sent is not None else None
        if self.last_sent is not None:
            self.last_sent.pop("reverse_open_hands", None)
            self.last_sent.pop("reverse_release", None)
        self.generation = str(uuid.uuid4())
        self.clear_votes()
        self.approved_hands.clear()
        self.closed_hands = {s for s in self.targets if self.last_sent and self.last_sent[s][7] >= .75}
        self.rearm_ns = {s: time.monotonic_ns() for s in ("left", "right")}
        self.active_hand = self.rollback_hand
        self.phase = "forward"
        self.event("ROLLBACK_DONE", replay=self.replays, rollback_hand=self.rollback_hand,
                   next="fresh_masked_observation")
        self.rollback_hand = None

    def end(self):
        self.active = False
        self.phase = "idle"
        self._reset_hand_tracking()
        self.history.clear()
        self.clear_votes()
        self.last_sent = None
        self.mask_active = False
        self.closed_hands.clear()
        self.approved_hands.clear()
        self.retry_hand = None
