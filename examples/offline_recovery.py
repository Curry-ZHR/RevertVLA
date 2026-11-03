"""Deterministic core walkthrough: no models, network, ROS or robot commands."""
from copy import deepcopy
import json
import time
from Reverse.core import ReverseEpisode


def main():
    mask = {side: {"enabled": True, "gripper": True} for side in ("left", "right")}
    episode = ReverseEpisode(["Aojiru", "Cola"])
    episode.begin("Grasp Aojiru with the left hand.", mask)
    anchor = {"left": [0.] * 8, "right": [.1] * 7 + [1.5], "body": {}, "speed": {}}
    episode.anchor(anchor)
    # Emulate successful executor acknowledgements, rather than recording predictions.
    for chunk in range(3):
        action = deepcopy(anchor)
        action["left"][:7] = [.02 * (chunk + 1)] * 7
        sent = {s: {"positions": action[s][:7], "gripper": action[s][7]} for s in ("left", "right")}
        episode.record(str(chunk), action, sent)
    candidate = deepcopy(action)
    candidate["left"][7] = 1.5
    now = time.monotonic_ns()
    for i in range(10):
        episode.ingest("left", "Aojiru", episode.generation, i, now - (10 - i) * 1000, "WRONG")
    gate = episode.gate([candidate])
    rejected = episode.timeout(gate)
    replay = episode.replay_plan()
    assert all(p["right"] == anchor["right"] for p in replay)
    episode.replay_finished(replay[-1])
    print(json.dumps({"gate": gate["state"], "after_hold_timeout": rejected["state"],
                      "rollback_hand": "left", "replay_steps": len(replay),
                      "other_hand_preserved": True, "head_mask_on_retry": episode.mask_active}, indent=2))


if __name__ == "__main__":
    main()
