/* LocalAgent — in-browser tool calling on onnxruntime-web (WebGPU + WASM fallback).
 *
 * The transformer forward pass runs as an ONNX graph emitting `logits` and `hidden`. The GENERABLE
 * dispatch (route head -> dense two-tower selector -> pointer-copy args) is ported here from the
 * Python pipeline. Bundle contract (see localagent.inference.export):
 *   model.fp16.onnx      inputs: input_ids[int64, 1xT]  outputs: logits[1,T,256], hidden[1,T,d]
 *   dispatch_heads.json  { route_head:{weight:[5][d],bias:[5],routes:[5],stop_index},
 *                          dense_selector:{q_proj_weight:[p][d],q_proj_bias:[p],proj:p,
 *                                          tool_matrix:[N][p],tool_names:[N],normalize_query} }
 *   heads.json           { pointer_head:{arg_idx,arg_emb,start_W,end_W}, ... }   (args copy)
 *   meta.json            { d_model, markers:{...}, tools:[{name,args,schema}] }   (50 tools)
 *
 * Selection is NOT a fixed-N classifier: the dense selector scores every tool by its description
 * embedding, so adding/removing a tool is adding/removing a tool_matrix row.
 */

const MODEL_URL = "model.fp16.onnx";
let SESSION = null;
let HEADS = null;
let META = null;
let DISPATCH = null;
let BACKEND = "wasm";

// ---- bundle loading -------------------------------------------------------
async function loadBundle() {
  ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  [HEADS, META, DISPATCH] = await Promise.all([
    fetch("heads.json").then((r) => r.json()),
    fetch("meta.json").then((r) => r.json()),
    fetch("dispatch_heads.json").then((r) => r.json()),
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
function mark(name) { return META.markers[name].text; } // markers carry { text, ids }

// Render a user turn the way the model was trained.
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

// ---- generable dispatch: route head -> dense two-tower selector ------------
function lastHidden(hiddenTensor, T) {
  const d = META.d_model, H = hiddenTensor.data, off = (T - 1) * d;
  return H.subarray ? H.subarray(off, off + d) : Array.from(H).slice(off, off + d);
}
function linrow(W, b, x) {                 // W[o][d] · x[d] + b[o] -> [o]
  const o = W.length, out = new Float32Array(o);
  for (let i = 0; i < o; i++) { const Wi = W[i]; let a = b ? b[i] : 0; for (let k = 0; k < x.length; k++) a += Wi[k] * x[k]; out[i] = a; }
  return out;
}
function argmax(v) { let bi = 0; for (let i = 1; i < v.length; i++) if (v[i] > v[bi]) bi = i; return bi; }
function softmaxAt(v, i) { let m = -Infinity; for (const x of v) m = Math.max(m, x); let z = 0; for (const x of v) z += Math.exp(x - m); return Math.exp(v[i] - m) / z; }

function dispatchSelect(hiddenTensor, T) {
  const last = lastHidden(hiddenTensor, T);
  // 1. route head (5-way modality gate); the `text` route (stop_index) = abstain / direct answer.
  const R = DISPATCH.route_head;
  const rl = linrow(R.weight, R.bias, last);
  const ri = argmax(rl);
  if (ri === R.stop_index) return { isStop: true, route: R.routes[ri], conf: softmaxAt(rl, ri) };
  // 2. dense selector: q = normalize(q_proj(last)); score_j = q · tool_matrix[j]; argmax.
  const S = DISPATCH.dense_selector;
  const q = linrow(S.q_proj_weight, S.q_proj_bias, last);
  if (S.normalize_query) { let n = 0; for (const x of q) n += x * x; n = Math.sqrt(n) || 1; for (let i = 0; i < q.length; i++) q[i] /= n; }
  let bi = 0, bs = -Infinity;
  for (let j = 0; j < S.tool_names.length; j++) {
    const Tj = S.tool_matrix[j]; let a = 0; for (let i = 0; i < S.proj; i++) a += Tj[i] * q[i];
    if (a > bs) { bs = a; bi = j; }
  }
  return { name: S.tool_names[bi], route: R.routes[ri], conf: (bs + 1) / 2, isStop: false };
}

// ---- argument grounding via the learned pointer head (port of pointer_head) ----
//   q = arg_emb[arg_idx[arg]];  qs = start_W·q;  qe = end_W·q
//   start = argmax_t hidden[t]·qs;  end = argmax_{t>=start} hidden[t]·qe;  value = bytes[start..end]
function matvec(M, v) {
  const d = v.length, out = new Float32Array(d);
  for (let i = 0; i < d; i++) { const Mi = M[i]; let a = 0; for (let j = 0; j < d; j++) a += Mi[j] * v[j]; out[i] = a; }
  return out;
}
function dotAt(H, t, d, q) { const off = t * d; let a = 0; for (let k = 0; k < d; k++) a += H[off + k] * q[k]; return a; }
function pointerSpan(arg, ids, H, T) {
  const ph = HEADS.pointer_head, d = META.d_model;
  const ai = ph.arg_idx[arg];
  if (ai == null) return "";
  const qs = matvec(ph.start_W, ph.arg_emb[ai]);
  const qe = matvec(ph.end_W, ph.arg_emb[ai]);
  let s = 0, sb = -Infinity;
  for (let t = 0; t < T; t++) { const v = dotAt(H, t, d, qs); if (v > sb) { sb = v; s = t; } }
  let e = s, eb = -Infinity;
  for (let t = s; t < T; t++) { const v = dotAt(H, t, d, qe); if (v > eb) { eb = v; e = t; } }
  try { return new TextDecoder().decode(new Uint8Array(ids.slice(s, e + 1))); } catch { return ""; }
}
function groundArgs(tool, ids, hiddenTensor, T) {
  const spec = (META.tools || []).find((t) => t.name === tool);
  const args = {};
  if (!spec) return args;
  const H = hiddenTensor.data;
  for (const arg of spec.args || []) {
    args[arg] = HEADS.pointer_head.arg_idx[arg] != null ? pointerSpan(arg, ids, H, T) : "";
  }
  return args;
}

// ---- single grounded call -------------------------------------------------
async function callOnce(query) {
  const ids = renderContext(query, []);
  const t0 = performance.now();
  const out = await forward(ids);
  const sel = dispatchSelect(out.hidden, ids.length);
  const ms = performance.now() - t0;
  if (sel.isStop) return { abstain: true, route: sel.route, conf: sel.conf, ms };
  return { tool: sel.name, route: sel.route, args: groundArgs(sel.name, ids, out.hidden, ids.length), conf: sel.conf, ms };
}

// ---- planner rollout ------------------------------------------------------
async function planRollout(query, maxSteps = 4) {
  const steps = [];
  const t0 = performance.now();
  for (let i = 0; i < maxSteps; i++) {
    const ids = renderContext(query, steps);
    const out = await forward(ids);
    const sel = dispatchSelect(out.hidden, ids.length);
    if (sel.isStop) break;
    const args = groundArgs(sel.name, ids, out.hidden, ids.length);
    steps.push({ tool: sel.name, route: sel.route, args, conf: sel.conf, response: simResponse(sel.name, args) });
  }
  return { steps, ms: performance.now() - t0 };
}

// A compact simulated tool response so downstream steps have context.
function simResponse(tool, args) {
  if (/read_file|grep|list_dir|find/.test(tool)) return Object.values(args)[0] || "ok";
  if (/search|news|http|open_url|define/.test(tool)) return "result: " + (Object.values(args)[0] || "");
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
  const route = step.route ? `<span class="route">${step.route}</span>` : "";
  if (step.abstain) {
    div.innerHTML = `${conf}${route}<span class="tool">— abstains (no tool needed)</span>`;
  } else {
    const ix = idx != null ? `<span class="step-index">${idx + 1}.</span>` : "";
    div.innerHTML = `${conf}${route}${ix}<span class="tool">${step.tool}</span>` +
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
