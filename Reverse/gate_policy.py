"""Pure preclose vote policy. No model, ROS, network or command side effects."""
HISTORY_CHUNKS = 8
WINDOW_SIZE = 10
MIN_PASS = 7
HOLD_CYCLES = 20
MAX_WINDOW_AGE_NS = 1_000_000_000  # 10 distinct samples at 15 Hz span about .6 s.
MAX_NEWEST_AGE_NS = 500_000_000


def evaluate_window(window, now_ns):
    complete = len(window) == WINDOW_SIZE
    ordered = all(b["sequence"] > a["sequence"] and b["received_ns"] > a["received_ns"]
                  for a,b in zip(window, window[1:]))
    fresh = bool(complete and ordered and
                 all(0 <= now_ns-v["received_ns"] <= MAX_WINDOW_AGE_NS for v in window) and
                 0 <= now_ns-window[-1]["received_ns"] <= MAX_NEWEST_AGE_NS)
    states = [v["state"] for v in window]
    count = states.count("PASS")
    newest_two = states[-2:] == ["PASS", "PASS"]
    return dict(passed=fresh and count >= MIN_PASS and newest_two, fresh=fresh,
                decisions=len(window), pass_count=count, newest_two_pass=newest_two,
                recent_states=states)
