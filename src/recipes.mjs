// Random recipes over the operation table: the page's proposals, and fresh ones beside a kept variant's children.
import { OPS, random } from "./effects.mjs";

const NAMES = Object.keys(OPS);

// A node with every parameter drawn uniformly within its bounds (among those the operation's fits allows), a
// strength in [0.4, 1] (from its least, if higher), and in a fifth of cases a soft region that confines it. A blind
// reviewer read small hard-edged patches on an otherwise unchanged photo as rendering defects, so borders stay soft.
// An operation with `whole` keeps only that share of its draws over the whole frame, the others drawn again.
export function randomNode(rnd) {
  for (;;) {
    const op = NAMES[Math.floor(rnd() * NAMES.length)], { params, int = [], least = 0.4, fits, whole = 1 } = OPS[op];
    const p = {};
    do {
      for (const [k, [min, max]] of Object.entries(params)) {
        p[k] = int.includes(k) ? min + Math.floor(rnd() * (max - min + 1)) : round(min + rnd() * (max - min));
      }
    } while (fits && !fits(p));
    const node = { op, params: p, strength: round(least + (1 - least) * rnd()) };
    if (rnd() < 1 / 5) node.region = randomRegion(rnd);
    if (node.region || whole === 1 || rnd() < whole) return node;
  }
}

// Boxes and ovals in two cases of five, half-planes (soft gradients) in one, and bands, rings, noise patches and the
// picture's own light in one each of ten. Sizes from a fifth of the picture, soft borders, and the outside instead in
// a case of four.
const SHAPES = [0, 0, 1, 1, 2, 3, 3, 4, 5, 6];
export function randomRegion(rnd) {
  const w = 0.2 + 0.8 * rnd(), h = 0.2 + 0.8 * rnd(), shape = SHAPES[Math.floor(rnd() * SHAPES.length)];
  return {
    x: round(rnd() * (1 - w)), y: round(rnd() * (1 - h)), w: round(w), h: round(h),
    feather: round(0.05 + 0.2 * rnd()), shape, angle: round(-180 + 360 * rnd()), invert: rnd() < 0.25 ? 1 : 0,
    level: round(0.25 + 0.5 * rnd()), seed: Math.floor(rnd() * 10000),
  };
}

// Two to six nodes.
export function randomRecipe(rnd) {
  return tidy(Array.from({ length: 2 + Math.floor(rnd() * 5) }, () => randomNode(rnd)));
}

// The recipe without repeats of an operation marked once (ink lines drawn from ink lines turn into black blobs).
export const tidy = (recipe) => recipe.filter((n, i) => !OPS[n.op].once || recipe.findIndex((m) => m.op === n.op) === i);

export const round = (v) => Math.round(v * 1e4) / 1e4;
