// Validates the orientation-aware crop-rec fix: run REAL YOLO to get text(1) +
// bubble(2) bboxes, crop each from the page, run REAL rec in all four 90°
// rotations, print orientation -> (conf, text). If rotating vertical dialogue
// yields readable Japanese, the crop+orientation architecture is proven.
import ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const SEG = join(ROOT, "models/yolo26s-manga-seg.onnx");
const REC = join(ROOT, "models/ppocrv6-rec.onnx");
const DICT = join(ROOT, "models/ppocrv6_dict.txt");
const DATA = join(ROOT, "data");
const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];
const INPUT = 1280, REC_H = 48, REC_W = 320;

const dict = readFileSync(DICT, "utf8").split("\n").map(l => l.replace(/\r$/, "")).filter((l, i, a) => !(i === a.length - 1 && l === ""));
function ctcDecode(logits, T, C) {
  const idx = [], conf = []; let prev = -1;
  for (let t = 0; t < T; t++) { let bi = 0, bv = -Infinity; const off = t * C;
    for (let c = 0; c < C; c++) { const v = logits[off + c]; if (v > bv) { bv = v; bi = c; } }
    if (bi !== 0 && bi !== prev) { idx.push(bi); conf.push(bv); } prev = bi; }
  let text = ""; for (const i of idx) { const ci = i - 1; if (ci >= 0 && ci < dict.length) text += dict[ci]; else if (ci === dict.length) text += " "; }
  return { text, conf: conf.length ? conf.reduce((s, v) => s + v, 0) / conf.length : 0 };
}

// rotate RGBA crop by k*90° CW. returns {w,h,data}
function rot90cw(img) {
  const { w, h, data } = img; const nw = h, nh = w; const out = new Uint8ClampedArray(nw * nh * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const X = h - 1 - y, Y = x; // new coords
    const si = (y * w + x) * 4, di = (X * nw + Y) * 4;
    out[di] = data[si]; out[di + 1] = data[si + 1]; out[di + 2] = data[si + 2]; out[di + 3] = data[si + 3];
  }
  return { w: nw, h: nh, data: out };
}
function rotate(img, k) { let r = img; for (let i = 0; i < (k & 3); i++) r = rot90cw(r); return r; }

function recTensor(img) {
  const { w, h, data } = img; const rr = REC_H / h; const tw = Math.min(REC_W, Math.max(1, Math.round(w * rr)));
  const plane = REC_H * REC_W; const t = new Float32Array(3 * plane);
  t.fill((0 - MEAN[0]) / STD[0], 0, plane); t.fill((0 - MEAN[1]) / STD[1], plane, 2 * plane); t.fill((0 - MEAN[2]) / STD[2], 2 * plane, 3 * plane);
  for (let y = 0; y < REC_H; y++) { const sy = Math.min(h - 1, Math.floor(y / rr));
    for (let x = 0; x < tw; x++) { const sx = Math.min(w - 1, Math.floor(x / rr));
      const sp = (sy * w + sx) * 4, di = y * REC_W + x;
      t[di] = (data[sp] / 255 - MEAN[0]) / STD[0]; t[plane + di] = (data[sp + 1] / 255 - MEAN[1]) / STD[1]; t[2 * plane + di] = (data[sp + 2] / 255 - MEAN[2]) / STD[2]; } }
  return t;
}

const segSess = await ort.InferenceSession.create(SEG, { executionProviders: ["cpu"] });
const recSess = await ort.InferenceSession.create(REC, { executionProviders: ["cpu"] });
async function rec(img) {
  const t = recTensor(img); const res = await recSess.run({ [recSess.inputNames[0]]: new ort.Tensor("float32", t, [1, 3, REC_H, REC_W]) });
  const o = res[recSess.outputNames[0]]; const d = o.dims.map(Number); return ctcDecode(o.data, d[1], d[2]);
}

const files = readFileSync ? (await import("node:fs")).readdirSync(DATA).filter(f => /\.webp$/i.test(f)).sort() : [];
for (const file of files) {
  const { data, info } = await sharp(readFileSync(join(DATA, file))).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height; const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
  // letterbox + YOLO
  const scale = Math.min(INPUT / W, INPUT / H); const nW = Math.round(W * scale), nH = Math.round(H * scale);
  const padX = Math.floor((INPUT - nW) / 2), padY = Math.floor((INPUT - nH) / 2);
  const plane = INPUT * INPUT; const tensor = new Float32Array(3 * plane).fill(114 / 255);
  for (let y = 0; y < nH; y++) { const sy = Math.min(H - 1, Math.floor(y / scale));
    for (let x = 0; x < nW; x++) { const sx = Math.min(W - 1, Math.floor(x / scale));
      const sp = (sy * W + sx) * 4, di = (padY + y) * INPUT + (padX + x);
      tensor[di] = rgba[sp] / 255; tensor[plane + di] = rgba[sp + 1] / 255; tensor[2 * plane + di] = rgba[sp + 2] / 255; } }
  const sres = await segSess.run({ [segSess.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, INPUT, INPUT]) });
  const det = sres[sess_out0(segSess)]; const dd = det.dims.map(Number); const N = dd[1], C = dd[2]; const d = det.data;
  const regions = [];
  for (let i = 0; i < N; i++) { const row = i * C; const conf = d[row + 4]; if (conf < 0.25) continue;
    const cls = Math.round(d[row + 5]); if (cls !== 1 && cls !== 2) continue; // text / bubble only
    const cx = d[row], cy = d[row + 1], w = d[row + 2], h = d[row + 3];
    const x = (cx - w / 2 - padX) / scale, y = (cy - h / 2 - padY) / scale;
    regions.push({ cls, conf, x: Math.max(0, x), y: Math.max(0, y), w: Math.min(W - Math.max(0, x), w / scale), h: Math.min(H - Math.max(0, y), h / scale) });
  }
  console.log(`\n========== ${file}: ${regions.length} YOLO text/bubble regions ==========`);
  for (const r of regions) {
    const cx0 = Math.max(0, Math.floor(r.x)), cy0 = Math.max(0, Math.floor(r.y));
    const cw = Math.max(2, Math.floor(r.w)), ch = Math.max(2, Math.floor(r.h));
    const crop = { w: cw, h: ch, data: new Uint8ClampedArray(cw * ch * 4) };
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const si = ((cy0 + y) * W + (cx0 + x)) * 4, di = (y * cw + x) * 4;
      crop.data[di] = rgba[si]; crop.data[di + 1] = rgba[si + 1]; crop.data[di + 2] = rgba[si + 2]; crop.data[di + 3] = 255;
    }
    const results = [];
    for (let k = 0; k < 4; k++) { const rr = rec(rotate(crop, k)); results.push({ k, ...(await rr) }); }
    results.sort((a, b) => (b.conf - a.conf) || (b.text.length - a.text.length));
    const best = results[0];
    const tag = r.cls === 1 ? "TXT" : "BUB";
    console.log(`  [${tag} conf=${r.conf.toFixed(2)} box=${Math.round(r.w)}x${Math.round(r.h)}] best rot=${best.k*90}° conf=${best.conf.toFixed(2)} text=${JSON.stringify(best.text)}  | all: ${results.map(x => `${x.k*90}°:${x.conf.toFixed(2)}${JSON.stringify(x.text)}`).join("  ")}`);
  }
}
function sess_out0(s) { return s.outputNames[0]; }
