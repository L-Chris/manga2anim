// Diagnostic: run the demo pipeline on data/ pages and report panel structure.
// Usage: node --experimental-strip-types scripts/diagnose.mjs  (or via tsx)
import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import sharp from "sharp";

const DATA_DIR = join(process.cwd(), "data");

async function decode(path) {
  const { data, info } = await sharp(readFileSync(path))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
  return {
    width: info.width,
    height: info.height,
    data: new Uint8ClampedArray(data.buffer, data.byteOffset, data.byteLength),
  };
}

// --- inline copy of the demo gutter segmentation logic (kept in sync) ---
function toGray(img) {
  const { width, height, data } = img;
  const gray = new Float32Array(width * height);
  for (let i = 0, p = 0; i < gray.length; i++, p += 4)
    gray[i] = 0.299 * data[p] + 0.587 * data[p + 1] + 0.114 * data[p + 2];
  return gray;
}
function rowFrac(gray, w, h, t) {
  const o = new Float32Array(h);
  for (let y = 0; y < h; y++) {
    let l = 0; const b = y * w;
    for (let x = 0; x < w; x++) if (gray[b + x] >= t) l++;
    o[y] = l / w;
  }
  return o;
}
function colFracInBand(gray, w, y0, y1, t) {
  const bandH = Math.max(1, y1 - y0);
  const o = new Float32Array(w);
  for (let x = 0; x < w; x++) {
    let l = 0;
    for (let y = y0; y < y1; y++) if (gray[y * w + x] >= t) l++;
    o[x] = l / bandH;
  }
  return o;
}
function gaps(frac, gf, minGap) {
  const g = []; let s = -1;
  for (let i = 0; i < frac.length; i++) {
    const isG = frac[i] >= gf;
    if (isG && s === -1) s = i;
    if ((!isG || i === frac.length - 1) && s !== -1) {
      const e = isG ? i + 1 : i;
      if (e - s >= minGap) g.push([s, e]);
      s = -1;
    }
  }
  return g;
}
function segs(len, g, m) {
  const out = []; let c = m;
  for (const [a, b] of g) { if (a - c > m) out.push([c, a]); c = b; }
  if (len - c > m) out.push([c, len - m]);
  if (!out.length) out.push([m, len - m]);
  return out;
}

const files = readdirSync(DATA_DIR).filter((f) => /\.webp$/i.test(f)).sort();
for (const f of files) {
  const img = await decode(join(DATA_DIR, f));
  const { width, height } = img;
  const gray = toGray(img);
  const lt = 235, gf = 0.97;
  const minGap = Math.max(4, Math.round(Math.min(width, height) * 0.006));
  const m = Math.max(2, Math.round(Math.min(width, height) * 0.01));
  const hg = gaps(rowFrac(gray, width, height, lt), gf, minGap);
  const rows = segs(height, hg, m);
  // Row-first XY-cut: per-band column gutters
  let totalPanels = 0;
  const rowDetail = [];
  for (const [y0, y1] of rows) {
    const vg = gaps(colFracInBand(gray, width, y0, y1, lt), gf, minGap);
    const cols = segs(width, vg, m);
    totalPanels += cols.length;
    rowDetail.push(`${y0}-${y1}: ${cols.length}col`);
  }
  console.log(
    `${f}: ${width}x${height}  hGaps=${hg.length} rows=${rows.length}  ` +
      `panels=${totalPanels}  [${rowDetail.join(", ")}]`
  );
}
