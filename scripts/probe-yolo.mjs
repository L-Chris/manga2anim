// Headless real-inference probe: load the REAL exported YOLO26s-seg ONNX with
// onnxruntime-node, run it on data/1.webp, and PRINT the true output tensor
// shapes + decode under both [1,C,N] and [1,N,C] layouts so we can see which
// matches Ultralytics' actual export. This is a diagnostic, not product code.
import ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const SEG = join(ROOT, "models/yolo26s-manga-seg.onnx");
const IMG = join(ROOT, "data/1.webp");
const INPUT = 1280;

const { data, info } = await sharp(readFileSync(IMG)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const W = info.width, H = info.height;
const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
console.log(`image ${W}x${H}`);

// letterbox (matches onnxProvider.ts)
const scale = Math.min(INPUT / W, INPUT / H);
const nW = Math.round(W * scale), nH = Math.round(H * scale);
const padX = Math.floor((INPUT - nW) / 2), padY = Math.floor((INPUT - nH) / 2);
const plane = INPUT * INPUT;
const tensor = new Float32Array(3 * plane).fill(114 / 255);
for (let y = 0; y < nH; y++) {
  const sy = Math.min(H - 1, Math.floor(y / scale));
  for (let x = 0; x < nW; x++) {
    const sx = Math.min(W - 1, Math.floor(x / scale));
    const sp = (sy * W + sx) * 4, di = (padY + y) * INPUT + (padX + x);
    tensor[di] = rgba[sp] / 255;
    tensor[plane + di] = rgba[sp + 1] / 255;
    tensor[2 * plane + di] = rgba[sp + 2] / 255;
  }
}

const sess = await ort.InferenceSession.create(SEG, { executionProviders: ["cpu"] });
console.log("inputs:", sess.inputNames);
const res = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, INPUT, INPUT]) });
console.log("outputs:", sess.outputNames);
for (const n of sess.outputNames) {
  const t = res[n];
  console.log(`  ${n}: dims=[${t.dims.join(",")}] type=${t.type}`);
}

const det = res[sess.outputNames[0]];
const dims = det.dims.map(Number);
const d = det.data;
// Determine layout: Ultralytics modern export is [1, N, 4+nc+nm] (row-major per
// detection). The probe prints both interpretations' first-candidate slices so
// we can eyeball which has sane cx,cy,w,h (within [0,1280]) and a one-hot-ish
// class score.
const [, A, B] = dims;
function rowMajorCandidate(i) { // [1,N,C]: row i = C values
  const off = i * B;
  return { cx: d[off], cy: d[off + 1], w: d[off + 2], h: d[off + 3], cls: [d[off + 4], d[off + 5], d[off + 6]] };
}
function colMajorCandidate(i) { // [1,C,N]: column i
  const N = B;
  return { cx: d[0 * N + i], cy: d[1 * N + i], w: d[2 * N + i], h: d[3 * N + i], cls: [d[4 * N + i], d[5 * N + i], d[6 * N + i]] };
}
console.log(`\nlayout test: A=${A} B=${B}`);
console.log("  if [1,N,C] (N=A,C=B): first 3 candidates (row-major):");
for (let i = 0; i < 3; i++) { const c = rowMajorCandidate(i); console.log(`    #${i} cx=${c.cx.toFixed(1)} cy=${c.cy.toFixed(1)} w=${c.w.toFixed(1)} h=${c.h.toFixed(1)} cls=[${c.cls.map(v => v.toFixed(2)).join(",")}]`); }
console.log("  if [1,C,N] (C=A,N=B): first 3 candidates (col-major):");
for (let i = 0; i < 3; i++) { const c = colMajorCandidate(i); console.log(`    #${i} cx=${c.cx.toFixed(1)} cy=${c.cy.toFixed(1)} w=${c.w.toFixed(1)} h=${c.h.toFixed(1)} cls=[${c.cls.map(v => v.toFixed(2)).join(",")}]`); }

// Count candidates with a strong class score under each layout, and how many
// have bbox coords inside the input canvas — the correct layout yields many
// in-range boxes with one dominant class score.
function stats(getter, N) {
  let inRange = 0, strong = 0;
  for (let i = 0; i < N; i++) {
    const c = getter(i);
    const maxCls = Math.max(...c.cls);
    if (maxCls > 0.25) strong++;
    if (c.cx >= 0 && c.cx <= INPUT && c.cy >= 0 && c.cy <= INPUT && c.w > 0 && c.w <= INPUT && c.h > 0 && c.h <= INPUT) inRange++;
  }
  return { N, strong, inRange };
}
console.log("\nstats (strong = max class score>0.25; inRange = bbox within canvas):");
console.log("  [1,N,C]:", JSON.stringify(stats(rowMajorCandidate, A)));
console.log("  [1,C,N]:", JSON.stringify(stats(colMajorCandidate, B)));

// --- structural dump: decide what the 34 columns after xywh actually are ---
console.log("\n=== candidate #0 full 38 columns (row-major [1,N,C]) ===");
const row0 = [];
for (let c = 0; c < B; c++) row0.push(d[0 * B + c]);
console.log(row0.map((v, i) => `${i}:${v.toFixed(2)}`).join("  "));

// Per-column stats over all 300 candidates, for columns 4..37.
// class_id column → values cluster near integers {0,1,2}; score column → [0,1];
// one-hot class logit columns → mostly negative with a few large positives;
// mask coeff columns → roughly zero-mean, mixed sign, no integer clustering.
console.log("\n=== per-column stats (col : min / max / mean / fracNearInt) over 300 cands ===");
for (let c = 4; c < B; c++) {
  let mn = Infinity, mx = -Infinity, sum = 0, nearInt = 0;
  for (let i = 0; i < A; i++) {
    const v = d[i * B + c];
    if (v < mn) mn = v; if (v > mx) mx = v; sum += v;
    if (Math.abs(v - Math.round(v)) < 0.02) nearInt++;
  }
  const mean = sum / A;
  console.log(`  col${String(c).padStart(2)}: min=${mn.toFixed(2).padStart(7)} max=${mx.toFixed(2).padStart(7)} mean=${mean.toFixed(2).padStart(6)} nearInt=${nearInt}/${A}`);
}
