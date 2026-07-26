// Det-only diagnostic: run REAL PP-OCRv6 det on data/1.webp, redo connected
// components while ACCUMULATING the prob sum per component, then print every
// candidate's bbox (image coords) + area + box_score (=mean prob inside the
// component). This reveals the true score distribution so we can pick a
// box_thresh that keeps real text bubbles and drops border/noise blobs.
import ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const DET = join(ROOT, "models/ppocrv6-det.onnx");
const IMG = join(ROOT, "data/1.webp");
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const THRESH = 0.3;

const { data, info } = await sharp(readFileSync(IMG)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const W = info.width, H = info.height;
const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);

let ratio = 1; const maxDim = Math.max(W, H);
if (maxDim > 960) ratio = 960 / maxDim;
const nH = Math.max(32, Math.round(Math.round(H * ratio) / 32) * 32);
const nW = Math.max(32, Math.round(Math.round(W * ratio) / 32) * 32);
const sW = nW / W, sH = nH / H;
const dPlane = nH * nW;
const detT = new Float32Array(3 * dPlane);
for (let y = 0; y < nH; y++) {
  const sy = Math.min(H - 1, Math.floor(y / sH));
  for (let x = 0; x < nW; x++) {
    const sx = Math.min(W - 1, Math.floor(x / sW));
    const sp = (sy * W + sx) * 4, di = y * nW + x;
    detT[di] = (rgba[sp] / 255 - MEAN[0]) / STD[0];
    detT[dPlane + di] = (rgba[sp + 1] / 255 - MEAN[1]) / STD[1];
    detT[2 * dPlane + di] = (rgba[sp + 2] / 255 - MEAN[2]) / STD[2];
  }
}

const sess = await ort.InferenceSession.create(DET, { executionProviders: ["cpu"] });
const res = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", detT, [1, 3, nH, nW]) });
const prob = res[sess.outputNames[0]].data;
const dd = res[sess.outputNames[0]].dims.map(Number);
const mapH = dd[dd.length - 2], mapW = dd[dd.length - 1];

const bin = new Uint8Array(mapH * mapW);
for (let i = 0; i < prob.length; i++) bin[i] = prob[i] > THRESH ? 1 : 0;

// two-pass CCL, accumulating prob sum + bounds per root
const labels = new Int32Array(mapH * mapW);
let next = 1;
const parent = new Int32Array(mapH * mapW + 1);
for (let i = 0; i < parent.length; i++) parent[i] = i;
const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb); };
const stats = new Map(); // root -> {x1,y1,x2,y2,area,sump}
for (let y = 0; y < mapH; y++) for (let x = 0; x < mapW; x++) {
  const idx = y * mapW + x;
  if (!bin[idx]) continue;
  const top = y > 0 ? labels[idx - mapW] : 0;
  const left = x > 0 ? labels[idx - 1] : 0;
  let lab;
  if (!top && !left) lab = next++;
  else if (top && !left) lab = top;
  else if (!top && left) lab = left;
  else { lab = Math.min(top, left); union(top, left); }
  labels[idx] = lab;
  const r = find(lab);
  let s = stats.get(r);
  if (!s) { s = { x1: x, y1: y, x2: x, y2: y, area: 0, sump: 0 }; stats.set(r, s); }
  s.x1 = Math.min(s.x1, x); s.y1 = Math.min(s.y1, y);
  s.x2 = Math.max(s.x2, x); s.y2 = Math.max(s.y2, y);
  s.area++; s.sump += prob[idx];
}

const rows = [];
for (const [, s] of stats) {
  const boxScore = s.sump / s.area;
  const w = s.x2 - s.x1 + 1, h = s.y2 - s.y1 + 1;
  rows.push({
    ix: Math.round(s.x1 / sW), iy: Math.round(s.y1 / sH),
    iw: Math.round(w / sW), ih: Math.round(h / sH),
    area: s.area, boxScore: +boxScore.toFixed(3),
  });
}
rows.sort((a, b) => b.area - a.area);
console.log(`total components = ${rows.length}`);
console.log("top 25 by area (image coords):  ix iy iw ih | area  boxScore");
for (const r of rows.slice(0, 25)) {
  console.log(`  ${String(r.ix).padStart(4)} ${String(r.iy).padStart(4)} ${String(r.iw).padStart(4)} ${String(r.ih).padStart(4)} | ${String(r.area).padStart(6)}  ${r.boxScore}`);
}
// score histogram
const buckets = { "<0.3": 0, "0.3-0.5": 0, "0.5-0.7": 0, "0.7-0.9": 0, ">=0.9": 0 };
for (const r of rows) {
  const b = r.boxScore;
  if (b < 0.3) buckets["<0.3"]++;
  else if (b < 0.5) buckets["0.3-0.5"]++;
  else if (b < 0.7) buckets["0.5-0.7"]++;
  else if (b < 0.9) buckets["0.7-0.9"]++;
  else buckets[">=0.9"]++;
}
console.log("\nbox_score histogram:", JSON.stringify(buckets));
console.log("components with boxScore>=0.6:", rows.filter(r => r.boxScore >= 0.6).length);
console.log("components with boxScore>=0.5:", rows.filter(r => r.boxScore >= 0.5).length);
