// CLIP's image input, prepared where a picture is rendered: the whole picture squashed to SIZE x SIZE (SCORER's;
// 224 is CLIP's own) by an area average (no crop), then normalized as CLIP's processor does, as planar floats. No
// model code, so the render workers can import it.
import { SCORER } from "./objective.mjs";

export const SIZE = SCORER.size;
export const MEAN = [0.48145466, 0.4578275, 0.40821073], STD = [0.26862954, 0.26130258, 0.27577711];
const PLANE = SIZE * SIZE;

// For each output index along one axis of `from` pixels, the source indices it averages and their weights
// (summing to 1). The area average is separable, so a picture is resized along x, then along y.
const axes = new Map();
function taps(from) {
  if (!axes.has(from)) {
    const s = from / SIZE, index = [], weight = [];
    for (let x = 0; x < SIZE; x++) {
      const x0 = x * s, x1 = x0 + s, i = [], w = [];
      // Clamped: x1 can land a rounding error past the edge.
      for (let k = Math.floor(x0), end = Math.min(Math.ceil(x1), from); k < end; k++) { i.push(k); w.push(Math.min(x1, k + 1) - Math.max(x0, k)); }
      const n = w.reduce((a, b) => a + b, 0);
      index.push(i); weight.push(w.map((v) => v / n));
    }
    axes.set(from, { index, weight });
  }
  return axes.get(from);
}

let rowsBuffer = new Float32Array(0);

// An image ({ data, width, height }, RGB floats in [0, 1]) as CLIP's pixel values: 3 planes of SIZE x SIZE.
export function pixelValues({ data, width, height }) {
  const tx = taps(width), ty = taps(height), out = new Float32Array(3 * PLANE);
  if (rowsBuffer.length < height * SIZE * 3) rowsBuffer = new Float32Array(height * SIZE * 3);
  const rows = rowsBuffer;
  // Along x: every source row to SIZE columns.
  for (let y = 0; y < height; y++) {
    const from = y * width * 3, to = y * SIZE * 3;
    for (let x = 0; x < SIZE; x++) {
      const index = tx.index[x], weight = tx.weight[x];
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < index.length; k++) {
        const o = from + 3 * index[k], w = weight[k];
        r += w * data[o]; g += w * data[o + 1]; b += w * data[o + 2];
      }
      rows[to + 3 * x] = r; rows[to + 3 * x + 1] = g; rows[to + 3 * x + 2] = b;
    }
  }
  // Along y, then normalized into planes.
  const a0 = 1 / STD[0], a1 = 1 / STD[1], a2 = 1 / STD[2], b0 = MEAN[0] / STD[0], b1 = MEAN[1] / STD[1], b2 = MEAN[2] / STD[2];
  for (let y = 0; y < SIZE; y++) {
    const index = ty.index[y], weight = ty.weight[y];
    for (let x = 0; x < SIZE; x++) {
      let r = 0, g = 0, b = 0;
      for (let k = 0; k < index.length; k++) {
        const o = (index[k] * SIZE + x) * 3, w = weight[k];
        r += w * rows[o]; g += w * rows[o + 1]; b += w * rows[o + 2];
      }
      const p = y * SIZE + x;
      out[p] = r * a0 - b0; out[PLANE + p] = g * a1 - b1; out[2 * PLANE + p] = b * a2 - b2;
    }
  }
  return out;
}
