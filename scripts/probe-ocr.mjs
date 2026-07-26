// Headless probe: load the REAL PP-OCRv6 det+rec ONNX with onnxruntime-node,
// run on data/1.webp, and PRINT the true det/rec output tensor shapes so we can
// calibrate ppocr.ts against reality (esp. rec numClasses vs dictionary length).
import ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const DET = join(ROOT, "models/ppocrv6-det.onnx");
const REC = join(ROOT, "models/ppocrv6-rec.onnx");
const DICT = join(ROOT, "models/ppocr_keys_v1.txt");
const IMG = join(ROOT, "data/1.webp");

const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

const { data, info } = await sharp(readFileSync(IMG)).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
const W = info.width, H = info.height;
const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
console.log(`image ${W}x${H}`);

// --- det preprocess (mirror of ppocr.ts detPreprocess) ---
let ratio = 1; const maxDim = Math.max(W, H);
if (maxDim > 960) ratio = 960 / maxDim;
let nH = Math.max(32, Math.round(Math.round(H * ratio) / 32) * 32);
let nW = Math.max(32, Math.round(Math.round(W * ratio) / 32) * 32);
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

const detSess = await ort.InferenceSession.create(DET, { executionProviders: ["cpu"] });
console.log("DET inputs:", detSess.inputNames, "outputs:", detSess.outputNames);
const detRes = await detSess.run({ [detSess.inputNames[0]]: new ort.Tensor("float32", detT, [1, 3, nH, nW]) });
const detOut = detRes[detSess.outputNames[0]];
const dd = detOut.dims.map(Number);
console.log(`DET output dims=[${dd.join(",")}]  (detPreprocess newH=${nH} newW=${nW})`);
const flat = detOut.data;
let mx = -Infinity, gt = 0;
for (let i = 0; i < flat.length; i++) { if (flat[i] > mx) mx = flat[i]; if (flat[i] > 0.3) gt++; }
console.log(`DET prob map: max=${mx.toFixed(3)}  pixels>0.3 = ${gt} / ${flat.length}`);

// --- pick a text box via simple threshold+scan (first connected run) ---
// We just need ONE crop to probe rec; find first row/col block above threshold.
const mapH = dd[dd.length - 2], mapW = dd[dd.length - 1];
let y0 = -1, y1 = -1, x0 = -1, x1 = -1;
outer: for (let y = 0; y < mapH; y++) for (let x = 0; x < mapW; x++) {
  if (flat[y * mapW + x] > 0.3) { y0 = y; break outer; }
}
if (y0 >= 0) {
  for (let y = y0; y < mapH; y++) { let any = false; for (let x = 0; x < mapW; x++) if (flat[y * mapW + x] > 0.3) { any = true; if (x0 < 0) x0 = x; x1 = x; } if (!any) { y1 = y; break; } if (y === mapH - 1) y1 = mapH; }
}
console.log(`first det blob (map coords): y[${y0},${y1}) x[${x0},${x1})`);
// map -> original image coords
const bx = x0 / sW, by = y0 / sH, bw = (x1 - x0) / sW, bh = (y1 - y0) / sH;
console.log(`first det blob (image coords): x=${bx.toFixed(0)} y=${by.toFixed(0)} w=${bw.toFixed(0)} h=${bh.toFixed(0)}`);

// --- rec preprocess on that crop (mirror of ppocr.ts recPreprocess) ---
const REC_H = 48, REC_W = 320;
const cx = Math.max(0, Math.floor(bx)), cy = Math.max(0, Math.floor(by));
const cw = Math.min(W - cx, Math.ceil(bw)), ch = Math.min(H - cy, Math.ceil(bh));
const rr = REC_H / ch;
const targetW = Math.min(REC_W, Math.max(1, Math.round(cw * rr)));
const rPlane = REC_H * REC_W;
const recT = new Float32Array(3 * rPlane);
for (let y = 0; y < REC_H; y++) {
  const sy = Math.min(ch - 1, Math.floor(y / rr));
  for (let x = 0; x < targetW; x++) {
    const sx = Math.min(cw - 1, Math.floor(x / rr));
    const sp = ((cy + sy) * W + (cx + sx)) * 4, di = y * REC_W + x;
    recT[di] = (rgba[sp] / 255 - MEAN[0]) / STD[0];
    recT[rPlane + di] = (rgba[sp + 1] / 255 - MEAN[1]) / STD[1];
    recT[2 * rPlane + di] = (rgba[sp + 2] / 255 - MEAN[2]) / STD[2];
  }
}

const recSess = await ort.InferenceSession.create(REC, { executionProviders: ["cpu"] });
console.log("\nREC inputs:", recSess.inputNames, "outputs:", recSess.outputNames);
const recRes = await recSess.run({ [recSess.inputNames[0]]: new ort.Tensor("float32", recT, [1, 3, REC_H, REC_W]) });
const recOut = recRes[recSess.outputNames[0]];
const rd = recOut.dims.map(Number);
console.log(`REC output dims=[${rd.join(",")}]  -> T=${rd[1]} numClasses=${rd[2]}`);

const dictLines = readFileSync(DICT, "utf8").split("\n").map(l => l.replace(/\r$/, ""));
const dictLen = dictLines[dictLines.length - 1] === "" ? dictLines.length - 1 : dictLines.length;
console.log(`\nppocr_keys_v1.txt entries = ${dictLen}`);
console.log(`numClasses from model = ${rd[2]}  (need dictLen+1 == numClasses for ctcDecode to align)`);
console.log(`MATCH (zh v1 dict) = ${dictLen + 1 === rd[2]}`);
