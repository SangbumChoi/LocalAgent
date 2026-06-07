/* LocalAgent — in-browser tool calling on onnxruntime-web (WebGPU + WASM fallback).
 *
 * The transformer forward pass runs as an ONNX graph emitting `logits` and `hidden`.
 * The tool head, argument grounding, and the planner rollout are ported here from the Python
 * `tool_head` / grounding / `plan_rollout`. Bundle contract (see localagent.inference.export):
 *   model.fp16.onnx  inputs: input_ids[int64, 1xT]  outputs: logits[1,T,256], hidden[1,T,d]
 *   heads.json       { tool_head:{weight:[C][d], bias:[C], classes:[C], stop_index:int}, ... }
 *   meta.json        { vocab_size, d_model, pad_id, markers:{...}, tools:[{name,args,schema}], tool_classes }
 */

const MODEL_URL = "model.fp16.onnx";
let SESSION = null;
let HEADS = null;
let META = null;
let BACKEND = "wasm";

// ---- bundle loading -------------------------------------------------------
async function loadBundle() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  [HEADS, META] = await Promise.all([
    fetch("heads.json").then((r) => r.json()),
    fetch("meta.json").then((r) => r.json()),
  ]);
  try {
    SESSION = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["webgpu", "wasm"],
    });
    BACKEND = "webgpu";
  } catch (e) {
    console.warn("WebGPU unavailable, falling back to WASM:", e);
    SESSION = await ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] });
    BACKEND = "wasm";
  }
}

// ---- byte tokenizer (vocab 256) ------------------------------------------
// Markers are literal strings encoded as UTF-8 bytes — identical to the Python byte tokenizer.
const enc = new TextEncoder();
function bytesOf(s) { return Array.from(enc.encode(s)); }
function mark(name) { return META.markers[name]; } // literal string

// Render a user turn the way the model was trained / `plan_rollout` renders it.
function renderContext(query, steps) {
  let s = mark("user") + query + mark("assistant");
  for (const st of steps || []) {
    s += mark("tool_call_open") + st.tool + "(" + JSON.stringify(st.args) + ")" + mark("tool_call_close");
    s += mark("tool") + mark("tool_response_open") + (st.response || "ok") + mark("tool_response_close");
    s += mark("assistant");
  }
  return bytesOf(s);
}

// ---- model forward --------------------------------------------------------
async function forward(ids) {
  const arr = BigInt64Array.from(ids.map((x) => BigInt(x)));
  const input = new ort.Tensor("int64", arr, [1, ids.length]);
  const out = await SESSION.run({ input_ids: input });
  return out; // { logits, hidden }
}

// ---- tool head (linear on the last hidden vector) -------------------------
function softmaxArgmax(logits) {
  let m = -Infinity;
  for (const v of logits) m = Math.max(m, v);
  let z = 0;
  const p = logits.map((v) => { const e = Math.exp(v - m); z += e; return e; });
  let bi = 0;
  for (let i = 1; i < p.length; i++) if (p[i] > p[bi]) bi = i;
  return { index: bi, conf: p[bi] / z };
}

function selectTool(hiddenTensor, T) {
  const d = META.d_model;
  const H = hiddenTensor.data;            // Float32Array length T*d
  const off = (T - 1) * d;                // last position
  const last = H.subarray ? H.subarray(off, off + d) : Array.from(H).slice(off, off + d);
  const { weight, bias, classes, stop_index } = HEADS.tool_head;
  const logits = new Array(classes.length);
  for (let c = 0; c < classes.length; c++) {
    let acc = bias[c];
    const Wc = weight[c];
    for (let k = 0; k < d; k++) acc += Wc[k] * last[k];
    logits[c] = acc;
  }
  const { index, conf } = softmaxArgmax(logits);
  return { name: classes[index], index, conf, isStop: index === stop_index };
}

// ---- argument grounding (browser approximation of the Python grounder) ----
// Copies arg values from spans of the prompt by format/name. The full Python grounder
// (heuristic extractors + pointer head + schema-constrained decode) is the source of truth.
function groundArgs(tool, prompt) {
  const spec = (META.tools || []).find((t) => t.name === tool);
  const args = {};
  if (!spec) return args;
  const props = (spec.schema && spec.schema.properties) || {};
  for (const arg of spec.args || Object.keys(props)) {
    const fmt = (props[arg] && (props[arg].format || props[arg].type)) || "";
    args[arg] = extractByFormat(arg, fmt, prompt) || "";
  }
  return args;
}

function extractByFormat(arg, fmt, p) {
  const path = p.match(/\b[\w./-]*\/[\w./-]+|\b[\w-]+\.[a-z]{1,5}\b/i);
  const url = p.match(/\b(?:https?:\/\/)?[\w-]+\.(?:com|org|io|ai|net|dev|co)[\w./-]*/i);
  const quoted = p.match(/['"“](.+?)['"”]/);
  const afterTo = p.match(/\bto\s+([A-Z][\w]+)/);
  const num = p.match(/\b\d+\b/);
  if (fmt === "path" || /file|path|source|dest/.test(arg)) return path && path[0];
  if (fmt === "uri" || /url/.test(arg)) return url && url[0];
  if (/recipient|name|to|assignee/.test(arg)) return (afterTo && afterTo[1]) || (quoted && quoted[1]);
  if (/message|summary|title|note|text|content|body/.test(arg)) return quoted && quoted[1];
  if (fmt === "integer" || fmt === "number" || /count|minutes|amount/.test(arg)) return num && num[0];
  // default: a cleaned query — strip a leading imperative verb
  if (/query|q|search|location|city/.test(arg)) {
    return p.replace(/^[^a-z0-9]*(what'?s|what is|search( the web)?( for)?|look up|find|get|tell me)\s+/i, "")
            .replace(/[?.!]+$/, "").trim();
  }
  return (quoted && quoted[1]) || null;
}

// ---- single grounded call -------------------------------------------------
async function callOnce(query) {
  const ids = renderContext(query, []);
  const t0 = performance.now();
  const out = await forward(ids);
  const sel = selectTool(out.hidden, ids.length);
  const ms = performance.now() - t0;
  if (sel.isStop) return { abstain: true, conf: sel.conf, ms };
  return { tool: sel.name, args: groundArgs(sel.name, query), conf: sel.conf, ms };
}

// ---- planner rollout (port of plan_rollout) -------------------------------
async function planRollout(query, maxSteps = 4) {
  const steps = [];
  const t0 = performance.now();
  for (let i = 0; i < maxSteps; i++) {
    const ids = renderContext(query, steps);
    const out = await forward(ids);
    const sel = selectTool(out.hidden, ids.length);
    if (sel.isStop) break;
    const args = groundArgs(sel.name, query);
    steps.push({ tool: sel.name, args, conf: sel.conf, response: simResponse(sel.name, args) });
  }
  return { steps, ms: performance.now() - t0 };
}

// A compact simulated tool response so downstream steps have context (mirrors _sim_response).
function simResponse(tool, args) {
  if (/read_file|grep/.test(tool)) return Object.values(args)[0] || "ok";
  if (/search|news/.test(tool)) return "result: " + (args.query || "");
  return "ok";
}

// ---- UI -------------------------------------------------------------------
const $ = (id) => document.getElementById(id);

function setStatus(cls, text, backend) {
  const s = $("status");
  s.className = "status " + cls;
  $("status-text").textContent = text;
  const b = $("backend-badge");
  if (backend) { b.hidden = false; b.textContent = backend.toUpperCase(); }
}

function renderCall(step, idx) {
  const div = document.createElement("div");
  div.className = "call" + (step.abstain ? " abstain" : "");
  const conf = step.conf != null ? `<span class="conf">${(step.conf * 100).toFixed(0)}%</span>` : "";
  if (step.abstain) {
    div.innerHTML = `${conf}<span class="tool">— abstains (no tool needed)</span>`;
  } else {
    const ix = idx != null ? `<span class="step-index">${idx + 1}.</span>` : "";
    div.innerHTML = `${conf}${ix}<span class="tool">${step.tool}</span>` +
      `<pre>${JSON.stringify(step.args, null, 2)}</pre>`;
  }
  return div;
}

async function run() {
  const query = $("prompt").value.trim();
  if (!query || !SESSION) return;
  $("run").disabled = true;
  const res = $("result");
  res.hidden = false;
  res.innerHTML = '<div class="call"><span class="tool">…thinking</span></div>';
  try {
    if ($("plan-mode").checked) {
      const { steps, ms } = await planRollout(query);
      res.innerHTML = "";
      if (!steps.length) res.appendChild(renderCall({ abstain: true }));
      steps.forEach((s, i) => res.appendChild(renderCall(s, i)));
      const t = document.createElement("div");
      t.className = "timing";
      t.textContent = `${steps.length} step(s) · ${ms.toFixed(0)} ms · ${BACKEND}`;
      res.appendChild(t);
    } else {
      const out = await callOnce(query);
      res.innerHTML = "";
      res.appendChild(renderCall(out));
      const t = document.createElement("div");
      t.className = "timing";
      t.textContent = `${out.ms.toFixed(0)} ms · ${BACKEND}`;
      res.appendChild(t);
    }
  } catch (e) {
    res.innerHTML = `<div class="call abstain"><span class="tool">error</span><pre>${e}</pre></div>`;
  } finally {
    $("run").disabled = false;
  }
}

function wireUI() {
  $("run").addEventListener("click", run);
  $("prompt").addEventListener("keydown", (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "Enter") run();
  });
  document.querySelectorAll(".chip").forEach((c) => {
    c.addEventListener("click", () => {
      $("prompt").value = c.textContent;
      $("plan-mode").checked = c.dataset.plan === "1";
      run();
    });
  });
}

(async function main() {
  wireUI();
  try {
    setStatus("loading", "Loading model… (first load downloads & caches the weights)");
    await loadBundle();
    setStatus("ready", "Model ready — runs locally in your browser.", BACKEND);
    $("run").disabled = false;
  } catch (e) {
    console.error(e);
    setStatus("error", "Failed to load the model bundle: " + e.message);
  }
})();
