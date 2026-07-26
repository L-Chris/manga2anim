// Definitive test of the spec's intended flow: YOLO region -> crop -> PP-OCR det
// INSIDE the crop (to isolate vertical columns; no panel borders inside a bubble)
// -> rotate each inner column 90° -> rec. If vertical dialogue emerges here, the
// crop->inner-det->oriented-rec architecture is proven and worth implementing.
import ort from "onnxruntime-node";
import sharp from "sharp";
import { readFileSync, readdirSync } from "node:fs";
import { join } from "node:path";

const ROOT = process.cwd();
const SEG = join(ROOT, "models/yolo26s-manga-seg.onnx");
const DET = join(ROOT, "models/ppocrv6-det.onnx");
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
function rot90cw(img) {
  const { w, h, data } = img; const nw = h, nh = w; const out = new Uint8ClampedArray(nw * nh * 4);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const X = h - 1 - y, Y = x;
    const si = (y * w + x) * 4, di = (X * nw + Y) * 4;
    out[di] = data[si]; out[di + 1] = data[si + 1]; out[di + 2] = data[si + 2]; out[di + 3] = data[si + 3]; }
  return { w: nw, h: nh, data: out };
}
function rotate(img, k) { let r = img; for (let i = 0; i < (k & 3); i++) r = rot90cw(r); return r; }
function normTensor(img, maxSide) {
  const { w, h, data } = img; let ratio = 1; if (Math.max(w, h) > maxSide) ratio = maxSide / Math.max(w, h);
  const nH = Math.max(32, Math.round(Math.round(h * ratio) / 32) * 32), nW = Math.max(32, Math.round(Math.round(w * ratio) / 32) * 32);
  const sW = nW / w, sH = nH / h; const plane = nH * nW; const t = new Float32Array(3 * plane);
  for (let y = 0; y < nH; y++) { const sy = Math.min(h - 1, Math.floor(y / sH));
    for (let x = 0; x < nW; x++) { const sx = Math.min(w - 1, Math.floor(x / sW));
      const sp = (sy * w + sx) * 4, di = y * nW + x;
      t[di] = (data[sp] / 255 - MEAN[0]) / STD[0]; t[plane + di] = (data[sp + 1] / 255 - MEAN[1]) / STD[1]; t[2 * plane + di] = (data[sp + 2] / 255 - MEAN[2]) / STD[2]; } }
  return { t, nH, nW, sW, sH };
}
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
const detSess = await ort.InferenceSession.create(DET, { executionProviders: ["cpu"] });
const recSess = await ort.InferenceSession.create(REC, { executionProviders: ["cpu"] });
async function recAt(img, k) { const r = recSess.run({ [recSess.inputNames[0]]: new ort.Tensor("float32", recTensor(rotate(img, k)), [1, 3, REC_H, REC_W]) }); const o = (await r)[recSess.outputNames[0]]; const d = o.dims.map(Number); return ctcDecode(o.data, d[1], d[2]); }

// inner det on a crop -> list of inner boxes (crop-local coords)
async function innerDet(img) {
  const { t, nH, nW, sW, sH } = normTensor(img, 960);
  const r = await detSess.run({ [detSess.inputNames[0]]: new ort.Tensor("float32", t, [1, 3, nH, nW]) });
  const prob = r[detSess.outputNames[0]].data; const dd = r[detSess.outputNames[0]].dims.map(Number);
  const mapH = dd[dd.length - 2], mapW = dd[dd.length - 1];
  const bin = new Uint8Array(mapH * mapW); for (let i = 0; i < prob.length; i++) bin[i] = prob[i] > 0.3 ? 1 : 0;
  const labels = new Int32Array(mapH * mapW); let next = 1; const parent = new Int32Array(mapH * mapW + 1);
  for (let i = 0; i < parent.length; i++) parent[i] = i;
  const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const union = (a, b) => { const ra = find(a), rb = find(b); if (ra !== rb) parent[Math.max(ra, rb)] = Math.min(ra, rb); };
  const st = new Map();
  for (let y = 0; y < mapH; y++) for (let x = 0; x < mapW; x++) {
    const idx = y * mapW + x; if (!bin[idx]) continue;
    const top = y > 0 ? labels[idx - mapW] : 0, left = x > 0 ? labels[idx - 1] : 0; let lab;
    if (!top && !left) lab = next++; else if (top && !left) lab = top; else if (!top && left) lab = left; else { lab = Math.min(top, left); union(top, left); }
    labels[idx] = lab; const rr = find(lab); let s = st.get(rr);
    if (!s) { s = { x1: x, y1: y, x2: x, y2: y, area: 0, sump: 0 }; st.set(rr, s); }
    s.x1 = Math.min(s.x1, x); s.y1 = Math.min(s.y1, y); s.x2 = Math.max(s.x2, x); s.y2 = Math.max(s.y2, y); s.area++; s.sump += prob[idx];
  }
  const out = [];
  for (const [, s] of st) {
    const bs = s.sump / s.area; if (bs < 0.5 || s.area < 20) continue;
    const w = s.x2 - s.x1 + 1, h = s.y2 - s.y1 + 1;
    out.push({ x: s.x1 / sW, y: s.y1 / sH, w: w / sW, h: h / sH, bs, area: s.area });
  }
  return out;
}

function cropOf(rgba, W, H, x, y, w, h) {
  const cx = Math.max(0, Math.floor(x)), cy = Math.max(0, Math.floor(y));
  const cw = Math.max(2, Math.min(W - cx, Math.ceil(w))), ch = Math.max(2, Math.min(H - cy, Math.ceil(h)));
  const data = new Uint8ClampedArray(cw * ch * 4);
  for (let yy = 0; yy < ch; yy++) for (let xx = 0; xx < cw; xx++) {
    const si = ((cy + yy) * W + (cx + xx)) * 4, di = (yy * cw + xx) * 4;
    data[di] = rgba[si]; data[di + 1] = rgba[si + 1]; data[di + 2] = rgba[si + 2]; data[di + 3] = 255;
  }
  return { w: cw, h: ch, data };
}

// Only first two pages to bound time; only bubble(2)+text(1) regions.
const files = readdirSync(DATA).filter(f => /\.webp$/i.test(f)).sort().slice(0, 2);
for (const file of files) {
  const { data, info } = await sharp(readFileSync(join(DATA, file))).ensureAlpha().raw().toBuffer({ resolveWithObject: true });
  const W = info.width, H = info.height; const rgba = new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength);
  const scale = Math.min(INPUT / W, INPUT / H); const nW = Math.round(W * scale), nH = Math.round(H * scale);
  const padX = Math.floor((INPUT - nW) / 2), padY = Math.floor((INPUT - nH) / 2);
  const plane = INPUT * INPUT; const tensor = new Float32Array(3 * plane).fill(114 / 255);
  for (let y = 0; y < nH; y++) { const sy = Math.min(H - 1, Math.floor(y / scale));
    for (let x = 0; x < nW; x++) { const sx = Math.min(W - 1, Math.floor(x / scale));
      const sp = (sy * W + sx) * 4, di = (padY + y) * INPUT + (padX + x);
      tensor[di] = rgba[sp] / 255; tensor[plane + di] = rgba[sp + 1] / 255; tensor[2 * plane + di] = rgba[sp + 2] / 255; } }
  const sres = await segSess.run({ [segSess.inputNames[0]]: new ort.Tensor("float32", tensor, [1, 3, INPUT, INPUT]) });
  const det = sres[segSess.outputNames[0]]; const dd = det.dims.map(Number); const N = dd[1], C = dd[2]; const d = det.data;
  const regions = [];
  for (let i = 0; i < N; i++) { const row = i * C; if (d[row + 4] < 0.25) continue; const cls = Math.round(d[row + 5]); if (cls !== 1 && cls !== 2) continue;
    const cx = d[row], cy = d[row + 1], w = d[row + 2], h = d[row + 3];
    regions.push({ cls, x: Math.max(0, (cx - w / 2 - padX) / scale), y: Math.max(0, (cy - h / 2 - padY) / scale), w: w / scale, h: h / scale });
  }
  console.log(`\n========== ${file}: ${regions.length} regions -> inner-det -> oriented-rec ==========`);
  let regionIdx = 0;
  for (const r of regions) {
    regionIdx++;
    const regionImg = cropOf(rgba, W, H, r.x, r.y, r.w, r.h);
    const inner = await innerDet(regionImg);
    if (inner.length === 0) { console.log(`  R${regionIdx} [${r.cls === 1 ? "TXT" : "BUB"} ${Math.round(r.w)}x${Math.round(r.h)}] inner-det: 0 columns`); continue; }
    console.log(`  R${regionIdx} [${r.cls === 1 ? "TXT" : "BUB"} ${Math.round(r.w)}x${Math.round(r.h)}] inner-det: ${inner.length} column(s)`);
    for (let j = 0; j < inner.length; j++) {
      const ib = inner[j];
      const colImg = cropOf(regionImg.data, regionImg.w, regionImg.h, ib.x, ib.y, ib.w, ib.h);
      // try 0 and 90 (vertical column -> horizontal)
      const r0 = await recAt(colImg, 0); const r90 = await recAt(colImg, 1);
      const pick = (r90.text.length && r90.conf >= r0.conf) ? { rot: 90, ...r90 } : { rot: 0, ...r0 };
      console.log(`     col${j} ${Math.round(ib.w)}x${Math.round(ib.h)} bs=${ib.bs.toFixed(2)} | 0°=${r0.conf.toFixed(2)}${JSON.stringify(r0.text)} 90°=${r90.conf.toFixed(2)}${JSON.stringify(r90.text)}  => pick ${pick.rot}° ${pick.conf.toFixed(2)} ${JSON.stringify(pick.text)}`);
    }
  }
}
