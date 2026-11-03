# Executor integration

RevertVLA provides the deployed recovery state machine and its executor adapters.
Your host application owns camera acquisition, action normalization, control
authority, joint limits, the stop button, and the RTC connection. The original
Robot Station server and general robot administration code are external host
components.

## Data contracts

An action step has `left` and `right`, each containing seven joint targets in
radians followed by a gripper target. The released H1 configuration uses open
`0.0`, closed `1.5`, and a prospective-close threshold of `0.75`. Adapt these values
for different hardware. Optional `body` and `speed` fields follow the host schema.

The execution mask has `left/right.enabled`, `left/right.gripper`, optionally
seven joint booleans, and `lift`, `waist`, `head`, `speed` flags. An observation
contains `observation.state` (the documented 23-value physical vector or the
structured state accepted by `ReverseRuntime.observe`) and base64 JPEG images
named `cam_high`, `cam_left_wrist`, and `cam_right_wrist`.

Pass a `camera_reader` to `ReverseRuntime(camera_reader=...)`. Its
`read_frame(camera, "color", timeout_sec=0.)` returns:

```python
{
    "ok": True,
    "body": encoded_jpeg_bytes,
    "sequence": increasing_integer,
    "receivedMonotonicNs": capture_time_in_the_host_monotonic_clock,
    "ageMs": current_frame_age_milliseconds,
}
```

The camera names are `camera_l` and `camera_r`. Sequence and capture time must
increase for each camera. Repeated, stale, future, previous-generation, and
pre-reopen scores do not authorize a close. If no camera reader is injected, the
runtime attempts Robot Station's `modules.ros_io` reader.

## Lifecycle and execution order

1. Configure the host with `source Reverse/configure.sh --configure`. Call
   `Reverse.runtime.startup()` at backend startup, or create a runtime and call
   `warmup_background()` explicitly. This warms two perception workers only.
2. Expose `status()`. Enable with `set_enabled(True)` only after readiness and
   outside an active task. Model residency is retained when disabling recovery.
3. Call `begin(prompt, mask, targets=...)` at episode start. Standard prompts use
   `Grasp Aojiru with the left hand.` Explicit targets can map each hand to a SKU.
4. Pass each fresh VLA input through `observe(observation)` before sending it to
   the external VLA server. This applies retry masking to the head image.
5. Before each actual prospective close, call `rolling_gate.guard_step(...)`.
   Gate results are `ORDINARY`, `PASS`, `WRONG`, or `REPLAN` after a bounded HOLD.
6. After the guarded SDK publisher succeeds, call `integration.record_step(...)`
   with its actual acknowledged targets. Unsent predictions and raw RTC prefixes
   must never enter history. SDK acknowledgement is not physical arrival proof.
7. On `WRONG`, retire old RTC results, call `integration.replay(...)`, capture a
   fresh observation, and bootstrap the same RTC connection. On `REPLAN`, the
   approved close has already been sent at the held pose; discard the old suffix
   and capture a fresh observation without sending it again.
8. Call `end()` after the task and `close()` when the host exits.

The executor must also invoke `episode.command_allowed(actual_command)` at its
final publish boundary. The check and successful-send bookkeeping must be
serialized with the episode lock so an old approval cannot leak into a new task.

## Required executor hooks

`rolling_gate.guard_step` and `integration.replay` accept the host manager with:

| Hook | Contract |
|---|---|
| `_stop_event` | A `threading.Event` shared with the existing stop path |
| `_publish_step(step, speed, mask, snapshot, fixed_targets, interval_sec=..., trajectory_config=...)` | The single guarded publisher; returns `{"ok": True, "action": {"left": {"positions": [...], "gripper": ...}, ...}}` after SDK success |
| `_safe_hold(hold_arms=False, signal_stop=False, preserve_reference=True)` | Cancels queued body/chassis work and returns `{"chassis": True, "body": True}` |
| `_zero_chassis()` | Stops chassis velocity and returns `{"ok": True}` |
| `_dispatch_body(steps, mask, ...)` | Existing body gateway; returns `{"ok": True}` |
| `_rtc_control_step` | Host cursor, needed only for trajectory retiming |

Supply `check_running()` to check task ownership and stop state. `before_hold()`
must retire cached continuations, invalidate preparation already in flight, and
prevent new VLA observations/requests during HOLD. Already running remote compute
may finish, but its results must be ignored. HOLD's 20 successful publications do
not advance the VLA cursor or guidance. The interval is one complete control
period; HOLD lasts approximately `20 / control_frequency` seconds.

Body feedback capture and coordinated-path retiming retain imports from Robot
Station. The released baseline supports normalized H1 actions with standard C2
RTC / synchronous execution (`legacy_clip`). Do not claim coordinated-path
support without validating your host's reverse retiming contract.

## Reference backend endpoints

The original host exposes `GET /api/inference/status`,
`POST /api/inference/reverse/start`, and `POST /api/inference/reverse/stop` under
the same operation lock as task execution. Its frontend implementation is included
as `examples/robot_station/inference-reverse.js` for reference; these routes must
be implemented by the integrating application.

The core walkthrough and tests run without Robot Station or ROS. They exercise
state transitions; they do not validate hardware motion or perception accuracy.
