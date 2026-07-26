// Decisive per-box diagnostic: for EVERY det candidate (low area floor) on each
// data/ page, run the REAL rec and print geometry + box_score + aspect + minDim
// + decoded text + rec confidence. Reading this table tells us empirically which
// feature separates real manga text from border/noise blobs — no guessing.
import ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const DET = join(ROOT, "models/ppocrv6-det.onnx");
const REC = join(ROOT, "models/ppocrv6-rec.onnx");
const DICT = join(ROOT, "models/ppocrv6_dict.txt");
const DATA = join(ROOT, "data");
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const THRESH = 0.3, REC_H = 48, REC_W = 320;

const dict = readFileSync(DICT, "utf8").split("\n").map(l => l.replace(/\r$/, "")).filter((l, i, a) => !(i === a.length - 1 && l === ""));
function ctcDecode(logits, T, C) {
  const idx = []; const conf = []; let prev = -1;
  for (let t = 0; t < T; t++) {
    let bi = 0, bv = -Infinity; const off = t * C;
    for (let c = 0; c < C; c++) { const v = logits[off + c]; if (v > bv) { bv = v; bi = c; } }
    if (bi !== 0 && bi !== prev) { idx.push(bi); conf.push(bv); }
    prev = bi;
  }
  let text = "";
  for (const i of idx) { const ci = i - 1; if (ci >= 0 && ci < dict.length) text += dict[ci]; else if (ci === dict.length) text += " "; }
  const c = conf.length ? conf.reduce((s, v) => s + v, 0) / conf.length : 0;
  return { text, conf: c };
}

const detSess = await ort.InferenceSession.create(DET, { executionProviders: ["cpu"] });
const recSess = await ort.InferenceSession.create(REC, { executionProviders: ["cpu"] });

async function recCrop(rgba, W, H, bx, by, bw, bh) {
  const cx = Math.max(0, Math.floor(bx)), cy = Math.max(0, Math.floor(by));
  const cw = Math.min(W - cx, Math.ceil(bw)), ch = Math.min(H - cy, Math.ceil(bh));
  if (cw < 2 || ch < 2) return null;
  const rr = REC_H / ch; const tw = Math.min(REC_W, Math.max(1, Math.round(cw * rr)));
  const plane = REC_H * REC_W; const t = new Float32Array(3 * plane);
  t.fill((0 - MEAN[0]) / STD[0], 0, plane); t.fill((0 - MEAN[1]) / STD[1], plane, 2 * plane); t.fill((0 - MEAN[2]) / STD[2], 2 * plane, 3 * plane);
  for (let y = 0; y < REC_H; y++) { const sy = Math.min(ch - 1, Math.floor(y / rr));
    for (let x = 0; x < tw; x++) { const sx = Math.min(cw - 1, Math.floor(x / rr));
      const sp = ((cy + sy) * W + (cx + sx)) * 4, di = y * REC_W + x;
      t[di] = (rgba[sp] / 255 - MEAN[0]) / STD[0]; t[plane + di] = (rgba[sp + 1] / 255 - MEAN[1]) / STD[1]; t[2 * plane + di] = (rgba[sp + 2] / 255 - MEAN[2]) / STD[2]; } }
  const res = await recSess.run({ [recSess.inputNames[0]]: new ort.Tensor("float32", t, [1, 3, REC_H, REC_W]) });
  const o = res[recSess.outputNames[0]]; const d = o.dims.map(Number);
  return ctcDecode(o.data, d[1], d[2]);
}

const files = readFileSync ? (await import("node:fs")).readdirSync(DATA).filter(f => /\.webp$/i.test(f)).sort() : [];
for (const file of files) {
  const { data, info } = await sharp(readFileSync(join(DATA, file))).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height;
  const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
  let ratio = 1; if (Math.max(W, H) > 960) ratio = 960 / Math.max(W, H);
  const nH = Math.max(32, Math.round(Math.round(H * ratio) / 32) * 32);
  const nW = Math.max(32, Math.round(Math.round(W * ratio) / 32) * 32);
  const sW = nW / W, sH = nH / H;
  const dPlane = nH * nW; const detT = new Float32Array(3 * dPlane);
  for (let y = 0; y < nH; y++) { const sy = Math.min(H - 1, Math.floor(y / sH));
    for (let x = 0; x < nW; x++) { const sx = Math.min(W - 1, Math.floor(x / sW));
      const sp = (sy * W + sx) * 4, di = y * nW + x;
      detT[di] = (rgba[sp] / 255 - MEAN[0]) / STD[0]; detT[dPlane + di] = (rgba[sp + 1] / 255 - MEAN[1]) / STD[1]; detT[2 * dPlane + di] = (rgba[sp + 2] / 255 - MEAN[2]) / STD[2]; } }
  const res = await detSess.run({ [detSess.inputNames[0]]: new ort.Tensor("float32", detT, [1, 3, nH, nW]) });
  const prob = res[detSess.outputNames[0]].data; const dd = res[detSess.outputNames[0]].dims.map(Number);
  const mapH = dd[dd.length - 2], mapW = dd[dd.length - 1];
  const bin = new Uint8Array(mapH * mapW); for (let i = 0; i < prob.length; i++) bin[i] = prob[i] > THRESH ? 1 : 0;
  const labels = new Int32Array(mapH * mapW); let next = 1; const parent = new Int32Array(mapH * mapW + 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb); };
  const st = new Map();
  for (let y = 0; y < mapH; y++) for (let x = 0; x < mapW; x++) {
    const idx = y * mapW + x; if (!bin[idx]) continue;
    const top = y > 0 ? labels[idx - mapW] : 0, left = x > 0 ? labels[idx - 1] : 0; let lab;
    if (!top && !left) lab = next++; else if (top && !left) lab = top; else if (!top && left) lab = left; else { lab = Math.min(top, left); union(top, left); }
    labels[idx] = lab; const r = find(lab); let s = st.get(r);
    if (!s) { s = { x1: x, y1: y, x2: x, y2: y, area: 0, sump: 0 }; st.set(r, s); }
    s.x1 = Math.min(s.x1, x); s.y1 = Math.min(s.y1, y); s.x2 = Math.max(s.x2, x); s.y2 = Math.max(s.y2, y); s.area++; s.sump += prob[idx];
  }
  // candidates: low floor so we see what we've been dropping
  const cands = [...st.values()].filter(s => s.area >= 80).map(s => {
    const w = s.x2 - s.x1 + 1, h = s.y2 - s.y1 + 1;
    return { ix: s.x1 / sW, iy: s.y1 / sH, iw: w / sW, ih: h / sH, area: s.area, bs: s.sump / s.area };
  }).sort((a, b) => b.area - a.area).slice(0, 40);

  console.log(`\n========== ${file} (${W}x${H}) — ${cands.length} candidates (area>=80 map px) ==========`);
  console.log(" ix   iy   iw   ih  | area  bScore asp  minD | recConf  recText");
  for (const c of cands) {
    const asp = (Math.max(c.iw, c.ih) / Math.max(1, Math.min(c.iw, c.ih))).toFixed(1);
    const minD = Math.round(Math.min(c.iw, c.ih));
    const r = await recCrop(rgba, W, H, c.ix, c.iy, c.iw, c.ih);
    const txt = r ? JSON.stringify(r.text).slice(0, 22) : "null";
    const rc = r ? r.conf.toFixed(2) : "  - ";
    console.log(`${String(Math.round(c.ix)).padStart(4)} ${String(Math.round(c.iy)).padStart(4)} ${String(Math.round(c.iw)).padStart(4)} ${String(Math.round(c.ih)).padStart(4)} | ${String(c.area).padStart(5)} ${c.bs.toFixed(2).padStart(6)} ${asp.padStart(4)} ${String(minD).padStart(4)} | ${rc}  ${txt}`);
  }
}
