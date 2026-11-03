/* UI only: the backend must advertise reverse support before enabling this control. */
const inferenceReverseControl = {
  backend: null,
  statusSeen: false,
  statusAt: 0,
  busy: false,
  error: "",
};

function applyInferenceReverseStatus(result) {
  if (!result || typeof result !== "object") return;
  const fullStatus = typeof result.status === "string" && typeof result.rtcActive === "boolean";
  if (!fullStatus && !("reverse" in result)) return;
  const previous = inferenceReverseControl.backend;
  const reverse = result.reverse;
  inferenceReverseControl.backend = reverse && typeof reverse === "object" &&
    typeof reverse.enabled === "boolean" ? {
      available: reverse.available === true,
      enabled: reverse.enabled,
      ready: reverse.ready === true,
      busy: reverse.busy === true,
      phase: reverse.phase || "",
      gateWindow: reverse.gateWindow,
      siglipStep: reverse.siglipStep,
      siglipObjects: reverse.siglipObjects,
      siglipPrompts: reverse.siglipPrompts,
      holdControlSteps: reverse.holdControlSteps || 0,
      detail: typeof reverse.detail === "string" ? reverse.detail : "",
    } : null;
  if (fullStatus) {
    inferenceReverseControl.statusSeen = true;
    inferenceReverseControl.statusAt = Date.now();
  }
  const current = inferenceReverseControl.backend;
  if (previous?.enabled !== current?.enabled || previous?.ready !== current?.ready ||
      previous?.available !== current?.available) inferenceReverseControl.error = "";
}

function inferenceReverseLocked() {
  const info = state.inference;
  return Boolean(info.statusPollError || !inferenceReverseControl.statusSeen ||
    Date.now() - inferenceReverseControl.statusAt > 5000 ||
    inferenceTrajectoryModeLocked() || info.status === "chunk_ready" || state.task === "inference");
}

function renderInferenceReverse() {
  const button = document.getElementById("inferenceReverseButton");
  const hint = document.getElementById("inferenceReverseStatus");
  if (!button || !hint) return;
  const control = inferenceReverseControl;
  const reverse = control.backend;
  const enabled = reverse?.enabled === true;
  const changing = control.busy || reverse?.busy;
  const locked = inferenceReverseLocked();
  button.textContent = control.busy ? (enabled ? "正在停止…" : "正在开启…")
    : enabled ? "停止回滚" : "启动回滚";
  button.disabled = Boolean(changing || locked || !reverse?.available || (!enabled && !reverse.ready));
  button.setAttribute("aria-pressed", String(enabled));
  button.setAttribute("aria-busy", String(Boolean(changing)));
  hint.textContent = control.error || (!control.statusSeen ? "正在读取回滚状态…"
    : !reverse?.available ? "回滚模块尚未接入"
    : state.inference.statusPollError || Date.now() - control.statusAt > 5000 ? "回滚状态待确认"
    : reverse.busy ? "回滚模块预热中…"
    : enabled && !reverse.ready ? "回滚模块异常，请停止回滚"
    : !reverse.ready ? "回滚模块未就绪"
    : enabled && reverse.phase === "hold" ? `活动臂保持并张爪，等待7/10及最新两票PASS（${reverse.holdControlSteps}/20）`
    : enabled ? `已开启${reverse.gateWindow === 10 ? "7/10" : "当前"}回滚 · SigLIP${reverse.siglipStep} · ${reverse.siglipObjects}类/${reverse.siglipPrompts}提示词`
    : `回滚已停止 · SigLIP${reverse.siglipStep}就绪`);
  button.title = !reverse?.available ? "回滚模块接入后可使用"
    : locked ? "本轮推理结束且状态确认后才可切换回滚"
    : changing ? "正在处理，请稍候" : !reverse.ready ? reverse.detail || "回滚模块未就绪"
    : enabled ? "停止使用回滚" : "开启使用回滚";
}

async function toggleInferenceReverse() {
  if (inferenceReverseControl.busy || inferenceReverseLocked() ||
      !inferenceReverseControl.backend?.available) return;
  const control = inferenceReverseControl;
  control.busy = true;
  control.error = "";
  renderInferenceReverse();
  try {
    // Recheck the existing task status before any user-requested toggle.
    const fresh = await refreshInferenceStatus();
    if (!fresh.ok || inferenceReverseLocked()) throw new Error("请等待本轮推理结束并确认状态");
    const reverse = control.backend;
    if (!reverse?.available || reverse.busy || (!reverse.enabled && !reverse.ready)) {
      throw new Error("回滚模块尚未就绪");
    }
    const enabled = !reverse.enabled;
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 10000);
    try {
      const response = await fetch(`/api/inference/reverse/${enabled ? "start" : "stop"}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: "{}",
        signal: controller.signal,
      });
      const result = await response.json();
      if (!response.ok || result.ok !== true) throw new Error("切换回滚失败，请确认模块状态");
      if (result.reverse?.available !== true || result.reverse.enabled !== enabled ||
          (enabled && result.reverse.ready !== true)) throw new Error("回滚状态待确认");
      applyInferenceReverseStatus(result);
    } finally {
      window.clearTimeout(timeout);
    }
  } catch (error) {
    control.error = error.name === "AbortError" ? "回滚状态待确认" : error.message;
  } finally {
    control.busy = false;
    renderInferenceReverse();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("inferenceReverseButton")?.addEventListener("click", toggleInferenceReverse);
  renderInferenceReverse();
}, { once: true });
