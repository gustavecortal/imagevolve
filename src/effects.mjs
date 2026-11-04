// The image operations a recipe chains, and the checks that drop flat or barely changed results.
// An image is { data: Float32Array of RGB in [0, 1], sRGB-encoded, row-major, width, height }.
// Positions, sizes and radii are fractions of the longest edge (L), so a recipe looks alike at any resolution.
// A recipe is a list of nodes { op, params, strength, region? }, each applied to the previous output.

const TAU = 2 * Math.PI;
const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);
const luma = (r, g, b) => 0.2126 * r + 0.7152 * g + 0.0722 * b;

// Longest edge of candidates during search, in pixels. evolve.mjs renders the kept recipes larger.
export const WORKING = 320;

// Each operation lists its parameters as [min, max, default], whether its value is an integer (int: [...names]),
// whether a recipe may hold it only once (once), and an apply(src, dst, p, ctx) that writes every pixel of dst from src. ctx: { w, h, L, scratch(name, length?) }.
// whole: the share of random draws kept when the operation covers the whole frame (recipes.mjs).
// Node strength mixes the result with the operation's input, so strength 0 is the identity. Where a mix or some
// parameters cancel the picture into a flat field, random draws avoid them (recipes.mjs): least, the lowest strength
// drawn, and fits(params), whether parameters may be drawn.
export const OPS = {
  // Exposure gain, contrast around mid-grey, then gamma.
  tone: {
    params: { gain: [0.4, 2.5, 1], contrast: [0.4, 2.5, 1], gamma: [0.4, 2.5, 1] },
    apply(s, d, { gain, contrast, gamma }) {
      const e = 1 / gamma;
      for (let i = 0; i < s.length; i++) {
        const v = clamp01((s[i] * gain - 0.5) * contrast + 0.5);
        d[i] = e === 1 ? v : Math.pow(v, e);
      }
    },
  },
  // Hue rotation in degrees, saturation, and a warm (positive) or cool (negative) shift.
  color: {
    params: { hue: [-180, 180, 0], saturation: [0, 3, 1], warmth: [-1, 1, 0] },
    apply(s, d, { hue, saturation: k, warmth: t }) {
      const [m0, m1, m2, m3, m4, m5, m6, m7, m8] = hueMatrix(hue);
      for (let i = 0; i < s.length; i += 3) {
        const r0 = s[i], g0 = s[i + 1], b0 = s[i + 2];
        const r = m0 * r0 + m1 * g0 + m2 * b0, g = m3 * r0 + m4 * g0 + m5 * b0, b = m6 * r0 + m7 * g0 + m8 * b0;
        const y = luma(r, g, b);
        d[i] = y + (r - y) * k + 0.15 * t; d[i + 1] = y + (g - y) * k + 0.03 * t; d[i + 2] = y + (b - y) * k - 0.15 * t;
      }
    },
  },
  // Luminance mapped from a shadow colour to a highlight colour (RGB in [0, 1]). Drawn with the highlight brighter
  // by 0.2 at least: a darker one, mixed with the picture, rendered flat in about half the draws, a brighter one never.
  duotone: {
    params: {
      shadowR: [0, 1, 0], shadowG: [0, 1, 0], shadowB: [0, 1, 0],
      lightR: [0, 1, 1], lightG: [0, 1, 1], lightB: [0, 1, 1],
    },
    fits: (p) => luma(p.lightR, p.lightG, p.lightB) - luma(p.shadowR, p.shadowG, p.shadowB) >= 0.2,
    apply(s, d, p) {
      const dr = p.lightR - p.shadowR, dg = p.lightG - p.shadowG, db = p.lightB - p.shadowB;
      for (let i = 0; i < s.length; i += 3) {
        const y = luma(s[i], s[i + 1], s[i + 2]);
        d[i] = p.shadowR + dr * y; d[i + 1] = p.shadowG + dg * y; d[i + 2] = p.shadowB + db * y;
      }
    },
  },
  // Each channel rounded to a number of levels.
  posterize: {
    params: { levels: [2, 16, 4] }, int: ["levels"],
    apply(s, d, { levels }) {
      const n = levels - 1;
      for (let i = 0; i < s.length; i++) d[i] = Math.round(s[i] * n) / n;
    },
  },
  // Black or white by luminance around a level. Softness widens the ramp between them.
  threshold: {
    params: { level: [0.1, 0.9, 0.5], softness: [0.005, 0.3, 0.05] },
    apply(s, d, { level, softness }) {
      for (let i = 0; i < s.length; i += 3) {
        const y = luma(s[i], s[i + 1], s[i + 2]);
        d[i] = d[i + 1] = d[i + 2] = clamp01((y - level) / softness + 0.5);
      }
    },
  },
  // Drawn at strength 0.7 at least: about half strength mixes the picture with its negative into grey.
  invert: {
    params: {}, least: 0.7,
    apply(s, d) {
      for (let i = 0; i < s.length; i++) d[i] = 1 - s[i];
    },
  },
  // Monochrome film grain from seeded value noise. Size is the grain cell, at least about a pixel at the
  // 320 px working size, so the search's grain is the saved picture's, smoothly interpolated.
  grain: {
    params: { amount: [0, 0.5, 0.1], size: [0.003, 0.02, 0.004], seed: [0, 9999, 1] }, int: ["seed"],
    apply(s, d, { amount, size, seed }, { w, h, L }) {
      const cell = size * L;
      for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i += 3) {
        const n = amount * valueNoise((x + 0.5) / cell, (y + 0.5) / cell, seed);
        d[i] = s[i] + n; d[i + 1] = s[i + 1] + n; d[i + 2] = s[i + 2] + n;
      }
    },
  },
  // Blur (amount -1) or sharpen (amount > 0) by unsharp masking.
  detail: {
    params: { radius: [0, 0.05, 0.005], amount: [-1, 3, 0] },
    apply(s, d, { radius, amount }, ctx) {
      const b = blur(s, radius, ctx);
      for (let i = 0; i < s.length; i++) d[i] = s[i] + amount * (s[i] - b[i]);
    },
  },
  // Dark lines where luminance changes, on white: an ink drawing of the edges. Gain scales line darkness. Above 3
  // the lines merge into thick black blobs, which blind reviewers read as a broken filter. Over the whole frame it
  // reads as a literal sketch of the photo (blind reviews), so it is mostly drawn inside a region.
  // The gradient is measured after a blur that scales with the image and multiplied by its width, so a line
  // keeps its darkness and relative thickness at any resolution.
  edges: {
    params: { gain: [0.5, 3, 1.5] }, once: true, whole: 1 / 3,
    apply(s, d, { gain }, ctx) {
      const { w, h, L } = ctx, r = EDGE_BLUR * L, b = blur(s, EDGE_BLUR, ctx), y = ctx.scratch("luma", w * h);
      for (let i = 0, j = 0; j < y.length; i += 3, j++) y[j] = luma(b[i], b[i + 1], b[i + 2]);
      const k = gain * Math.max(1, r);
      for (let row = 0; row < h; row++) {
        const up = (row > 0 ? row - 1 : 0) * w, mid = row * w, dn = (row < h - 1 ? row + 1 : row) * w;
        for (let x = 0; x < w; x++) {
          const l = x > 0 ? x - 1 : 0, rt = x < w - 1 ? x + 1 : x;
          const gx = y[up + rt] + 2 * y[mid + rt] + y[dn + rt] - y[up + l] - 2 * y[mid + l] - y[dn + l];
          const gy = y[dn + l] + 2 * y[dn + x] + y[dn + rt] - y[up + l] - 2 * y[up + x] - y[up + rt];
          const v = clamp01(1 - k * Math.sqrt(gx * gx + gy * gy)), o = 3 * (mid + x);
          d[o] = d[o + 1] = d[o + 2] = v;
        }
      }
    },
  },

  // Geometry: each output pixel samples the input at a mapped position (bilinear, mirrored at the borders).
  // A map receives a row's v and every column's u, and writes the source positions into xs and ys, all in units of
  // the longest edge: a row at a time, so no function is called per pixel.
  // Rotation in degrees and scale around a centre, then a shift. Flip mirrors horizontally.
  transform: {
    params: { angle: [-180, 180, 0], scale: [0.3, 4, 1], cx: [0, 1, 0.5], cy: [0, 1, 0.5], dx: [-0.5, 0.5, 0], dy: [-0.5, 0.5, 0], flip: [0, 1, 0] }, int: ["flip"],
    apply(s, d, p, ctx) {
      const a = (-p.angle * Math.PI) / 180, c = Math.cos(a) / p.scale, n = Math.sin(a) / p.scale;
      const [px, py] = point(p.cx, p.cy, ctx), f = p.flip ? -1 : 1;
      warp(s, d, ctx, (v, us, xs, ys) => {
        const y = v - py - p.dy;
        for (let k = 0; k < us.length; k++) {
          const x = f * (us[k] - px - p.dx);
          xs[k] = px + c * x - n * y; ys[k] = py + n * x + c * y;
        }
      });
    },
  },
  // A square of the image (centre sx, sy, side size) copied to another place, scaled and turned.
  paste: {
    params: { sx: [0, 1, 0.3], sy: [0, 1, 0.3], size: [0.05, 0.6, 0.25], dx: [0, 1, 0.7], dy: [0, 1, 0.7], scale: [0.3, 3, 1], angle: [-180, 180, 0] },
    apply(s, d, p, ctx) {
      const a = (-p.angle * Math.PI) / 180, c = Math.cos(a) / p.scale, n = Math.sin(a) / p.scale, half = p.size / 2;
      const [sx, sy] = point(p.sx, p.sy, ctx), [tx, ty] = point(p.dx, p.dy, ctx);
      warp(s, d, ctx, (v, us, xs, ys) => {
        const nv = n * (v - ty), cv = c * (v - ty);
        for (let k = 0; k < us.length; k++) {
          const u = us[k], x = c * (u - tx) - nv, y = n * (u - tx) + cv;
          const inside = Math.abs(x) <= half && Math.abs(y) <= half;
          xs[k] = inside ? sx + x : u; ys[k] = inside ? sy + y : v;
        }
      });
    },
  },
  // A grid of tiles, a share of them swapped among themselves and some flipped. Over the whole frame it reads as a
  // puzzle of the photo (blind reviews), so it is mostly drawn inside a region, as ink lines are.
  tiles: {
    params: { cols: [2, 12, 4], rows: [2, 12, 3], shuffle: [0, 1, 1], turn: [0, 1, 0], seed: [0, 9999, 1] }, int: ["cols", "rows", "seed"], whole: 1 / 3,
    apply(s, d, p, ctx) {
      const { order, flips } = tileTables(p);
      const tw = ctx.w / ctx.L / p.cols, th = ctx.h / ctx.L / p.rows;
      warp(s, d, ctx, (v, us, xs, ys) => {
        const row = Math.min(p.rows - 1, Math.floor(v / th));
        for (let k = 0; k < us.length; k++) {
          const u = us[k], col = Math.min(p.cols - 1, Math.floor(u / tw));
          const t = row * p.cols + col, from = order[t], fl = flips[t];
          let x = u - col * tw, y = v - row * th;
          if (fl & 1) x = tw - x;
          if (fl & 2) y = th - y;
          xs[k] = (from % p.cols) * tw + x; ys[k] = Math.floor(from / p.cols) * th + y;
        }
      });
    },
  },
  // The image cut into strips, each shifted along its length by a seeded amount.
  strips: {
    params: { count: [2, 64, 12], shift: [0, 0.5, 0.1], vertical: [0, 1, 0], seed: [0, 9999, 1] }, int: ["count", "vertical", "seed"],
    apply(s, d, p, ctx) {
      const offsets = stripOffsets(p);
      const W = ctx.w / ctx.L, H = ctx.h / ctx.L;
      warp(s, d, ctx, p.vertical
        ? (v, us, xs, ys) => { for (let k = 0; k < us.length; k++) { xs[k] = us[k]; ys[k] = v + offsets[Math.min(p.count - 1, Math.floor((us[k] / W) * p.count))] * H; } }
        : (v, us, xs, ys) => {
          const across = offsets[Math.min(p.count - 1, Math.floor((v / H) * p.count))] * W;
          for (let k = 0; k < us.length; k++) { xs[k] = us[k] + across; ys[k] = v; }
        });
    },
  },
  // Mirror segments around a centre, as a kaleidoscope with a number of segments.
  kaleido: {
    params: { segments: [2, 12, 6], angle: [-180, 180, 0], cx: [0, 1, 0.5], cy: [0, 1, 0.5] }, int: ["segments"],
    apply(s, d, p, ctx) {
      const [px, py] = point(p.cx, p.cy, ctx), wedge = TAU / p.segments, a0 = (p.angle * Math.PI) / 180;
      warp(s, d, ctx, (v, us, xs, ys) => {
        const y = v - py;
        for (let k = 0; k < us.length; k++) {
          const x = us[k] - px, r = Math.sqrt(x * x + y * y);
          let t = (((Math.atan2(y, x) - a0) % wedge) + wedge) % wedge;
          if (t > wedge / 2) t = wedge - t;
          xs[k] = px + r * Math.cos(t + a0); ys[k] = py + r * Math.sin(t + a0);
        }
      });
    },
  },
  // Rotation that grows towards a centre: a turn of `angle` degrees there, none beyond `radius`.
  swirl: {
    params: { angle: [-720, 720, 180], radius: [0.05, 1, 0.4], cx: [0, 1, 0.5], cy: [0, 1, 0.5] },
    apply(s, d, p, ctx) {
      const [px, py] = point(p.cx, p.cy, ctx), a = (p.angle * Math.PI) / 180;
      warp(s, d, ctx, (v, us, xs, ys) => {
        const y = v - py;
        for (let k = 0; k < us.length; k++) {
          const u = us[k], x = u - px, r = Math.sqrt(x * x + y * y);
          if (r >= p.radius) { xs[k] = u; ys[k] = v; continue; }
          const t = a * (1 - r / p.radius) ** 2, c = Math.cos(t), n = Math.sin(t);
          xs[k] = px + c * x - n * y; ys[k] = py + n * x + c * y;
        }
      });
    },
  },
  // The image wrapped round a centre: angle across, distance from the centre down (inside out when inverse).
  polar: {
    params: { cx: [0, 1, 0.5], cy: [0, 1, 0.5], turns: [1, 4, 1], inverse: [0, 1, 0] }, int: ["turns", "inverse"],
    apply(s, d, p, ctx) {
      const [px, py] = point(p.cx, p.cy, ctx), W = ctx.w / ctx.L, H = ctx.h / ctx.L, R = Math.hypot(W, H) / 2;
      // The inverse map's angle depends only on the column.
      const angles = p.inverse ? Array.from({ length: ctx.w }, (_, k) => (((k + 0.5) / ctx.L) / W) * TAU) : [];
      const cos = angles.map(Math.cos), sin = angles.map(Math.sin);
      warp(s, d, ctx, p.inverse
        ? (v, us, xs, ys) => {  // output columns are angles, rows distances: the image unwrapped around the centre
          const r = (v / H) * R;
          for (let k = 0; k < us.length; k++) { xs[k] = px + r * cos[k]; ys[k] = py + r * sin[k]; }
        }
        : (v, us, xs, ys) => {
          const y = v - py, y2 = y ** 2;
          for (let k = 0; k < us.length; k++) {
            const u = us[k];
            xs[k] = (((Math.atan2(y, u - px) / TAU + 1) * p.turns) % 1) * W; ys[k] = (Math.sqrt((u - px) ** 2 + y2) / R) * H;
          }
        });
    },
  },
  // Every pixel pushed by smooth seeded noise: amount is the largest push, size the noise's feature size.
  displace: {
    params: { amount: [0, 0.2, 0.03], size: [0.02, 0.5, 0.1], seed: [0, 9999, 1] }, int: ["seed"],
    apply(s, d, p, ctx) {
      warp(s, d, ctx, (v, us, xs, ys) => {
        const y = v / p.size;
        for (let k = 0; k < us.length; k++) {
          const u = us[k];
          xs[k] = u + p.amount * valueNoise(u / p.size, y, p.seed);
          ys[k] = v + p.amount * valueNoise(u / p.size, y, p.seed + 7919);
        }
      });
    },
  },
};

// Hue rotation about the grey axis (a luminance-preserving matrix, row by row), for a hue in degrees.
function hueMatrix(hue) {
  const a = (hue * Math.PI) / 180, c = Math.cos(a), n = Math.sin(a);
  return [0.213 + c * 0.787 - n * 0.213, 0.715 - c * 0.715 - n * 0.715, 0.072 - c * 0.072 + n * 0.928,
    0.213 - c * 0.213 + n * 0.143, 0.715 + c * 0.285 + n * 0.14, 0.072 - c * 0.072 - n * 0.283,
    0.213 - c * 0.213 - n * 0.787, 0.715 - c * 0.715 + n * 0.715, 0.072 + c * 0.928 + n * 0.072];
}

// The tiles' seeded tables: each tile's source tile, and its flips (1 across, 2 down, 3 both).
function tileTables(p) {
  const n = p.cols * p.rows, rnd = random(p.seed), order = [...Array(n).keys()];
  const moving = order.filter(() => rnd() < p.shuffle), shuffled = moving.slice();
  shuffle(shuffled, rnd);
  moving.forEach((t, k) => (order[t] = shuffled[k]));
  return { order, flips: order.map(() => (rnd() < p.turn ? 1 + Math.floor(rnd() * 3) : 0)) };
}

// Each strip's seeded shift.
const stripOffsets = (p) => { const rnd = random(p.seed); return Array.from({ length: p.count }, () => (rnd() * 2 - 1) * p.shift); };

// Regions that confine a node. x, y, w, h are fractions of width and height: shape 0 is that box, 1 an oval inside
// it. The other shapes are placed at the box's centre and sized in units of the longest edge: 2 a band at angle, h
// thick; 3 the half-plane on one side of a line at angle; 4 a ring w across and h/4 thick; 5 patches of seeded noise
// of feature size w covering about a share level of the picture; 6 the parts brighter than level (the picture's own
// light). invert swaps inside and outside. feather is the width of the soft border.
const REGION = {
  x: [0, 1, 0], y: [0, 1, 0], w: [0.05, 1, 1], h: [0.05, 1, 1], feather: [0, 0.3, 0.02],
  shape: [0, 6, 0], angle: [-180, 180, 0], invert: [0, 1, 0], level: [0, 1, 0.5], seed: [0, 9999, 1],
};

// Scratch buffers kept between renders (each is written whole before it is read), so the search does not allocate
// several picture-sized arrays per candidate.
// Named: "fx" each operation's output in render, "blurA" and "blurB" the blur's passes, "luma" the ink lines'.
const scratchPool = new Map();

// Apply a recipe to an image. Unknown operations or parameters and non-finite values throw. Values are clamped
// to their bounds. A node at strength 0 is skipped.
export function render(img, nodes) {
  const { width: w, height: h } = img, L = Math.max(w, h), size = img.data.length;
  const ctx = { w, h, L, scratch: (name, n = size) => { let b = scratchPool.get(name); if (b?.length !== n) scratchPool.set(name, (b = new Float32Array(n))); return b; } };
  // cur becomes the result, so it is a new array. fx, each operation's output, is reused: every operation writes all of it.
  const cur = new Float32Array(img.data), fx = ctx.scratch("fx");
  for (const { op, params, strength: k, region: r } of normalized(nodes)) {
    if (k === 0) continue;
    OPS[op].apply(cur, fx, params, ctx);
    if (r) {
      const fill = regionMask(r, ctx), row = new Float64Array(w);
      for (let y = 0, i = 0; y < h; y++) {
        fill((y + 0.5) / h, cur, i, row);
        for (let x = 0; x < w; x++, i += 3) {
          const m = k * row[x];
          cur[i] = clamp01(cur[i] + (fx[i] - cur[i]) * m);
          cur[i + 1] = clamp01(cur[i + 1] + (fx[i + 1] - cur[i + 1]) * m);
          cur[i + 2] = clamp01(cur[i + 2] + (fx[i + 2] - cur[i + 2]) * m);
        }
      }
    } else if (k === 1) {
      for (let i = 0; i < size; i++) cur[i] = clamp01(fx[i]);
    } else {
      for (let i = 0; i < size; i++) cur[i] = clamp01(cur[i] + (fx[i] - cur[i]) * k);
    }
  }
  return { width: w, height: h, data: cur };
}

// The recipe with every node complete (all parameters, strength, a full region or none) and within bounds.
// Throws, as render does, on unknown operations or parameters and non-finite values.
export function normalized(nodes) {
  return nodes.map((node) => {
    const def = OPS[node.op];
    if (!def) throw new Error(`unknown operation ${node.op}`);
    const strength = clamp01(Number(node.strength ?? 1));
    if (!Number.isFinite(strength)) throw new Error(`${node.op}.strength is not a finite number`);
    const out = { op: node.op, params: bounded(def.params, node.params ?? {}, def.int, node.op), strength };
    if (node.region) {
      out.region = bounded(REGION, node.region, ["shape", "invert", "seed"], "region");
    }
    return out;
  });
}

function bounded(spec, values, ints = [], what) {
  for (const k of Object.keys(values)) if (!(k in spec)) throw new Error(`unknown parameter ${what}.${k}`);
  const out = {};
  for (const [k, [min, max, initial]] of Object.entries(spec)) {
    let v = Number(values[k] ?? initial);
    if (!Number.isFinite(v)) throw new Error(`${what}.${k} is not a finite number`);
    if (ints.includes(k)) v = Math.round(v);
    out[k] = Math.min(max, Math.max(min, v));
  }
  return out;
}

// The region a row at a time, fill(v, img, i, m): for the row at v (a fraction of the height), whose first pixel is
// img at i, m gets each pixel's 1 inside, 0 outside, with a linear ramp `feather` wide at the border. Distances along
// an angle are in units of the longest edge, so a band or a half-plane keeps its angle on a wide picture.
function regionMask(region, ctx) {
  const { x, y, w, h, shape, invert, level, seed } = region, { w: W } = ctx, { f, cx, cy, sx, sy, nx, ny } = regionGeometry(region, ctx);
  const us = new Float64Array(W);
  for (let k = 0; k < W; k++) us[k] = (k + 0.5) / W;
  const fill = shape === 2 ? (v, img, i, m) => { const b = (v - cy) * sy * ny; for (let k = 0; k < W; k++) m[k] = clamp01((h / 2 - Math.abs((us[k] - cx) * sx * nx + b)) / f); }
    : shape === 3 ? (v, img, i, m) => { const b = (v - cy) * sy * ny; for (let k = 0; k < W; k++) m[k] = clamp01(((us[k] - cx) * sx * nx + b) / f + 0.5); }
    : shape === 4 ? (v, img, i, m) => {
      const b = ((v - cy) * sy) ** 2;
      for (let k = 0; k < W; k++) { const r = Math.sqrt(((us[k] - cx) * sx) ** 2 + b); m[k] = clamp01((h / 8 - Math.abs(r - w / 2)) / f); }
    }
    : shape === 5 ? (v, img, i, m) => { const b = (v * sy) / (w * 0.5); for (let k = 0; k < W; k++) m[k] = clamp01((valueNoise((us[k] * sx) / (w * 0.5), b, seed) * 0.5 + 0.5 - (1 - level)) / f + 0.5); }
    : shape === 6 ? (v, img, i, m) => { for (let k = 0; k < W; k++, i += 3) m[k] = clamp01((luma(img[i], img[i + 1], img[i + 2]) - level) / f + 0.5); }
    : shape === 1 ? (v, img, i, m) => {
      const dy = (v - cy) / (h / 2);
      for (let k = 0; k < W; k++) { const dx = (us[k] - cx) / (w / 2); m[k] = clamp01((1 - Math.sqrt(dx * dx + dy * dy)) * (Math.min(w, h) / 2) / f); }
    }
    : (v, img, i, m) => { const b = Math.min(v - y, y + h - v); for (let k = 0; k < W; k++) m[k] = clamp01(Math.min(us[k] - x, x + w - us[k], b) / f); };
  return invert ? (v, img, i, m) => { fill(v, img, i, m); for (let k = 0; k < W; k++) m[k] = 1 - m[k]; } : fill;
}

// A region's feather (above 0), centre, the picture's sides in units of the longest edge, and the unit normal of its
// angle, as regionMask reads them.
function regionGeometry({ x, y, w, h, feather, angle }, { w: W, h: H, L }) {
  const a = (angle * Math.PI) / 180;
  return { f: Math.max(feather, 1e-6), cx: x + w / 2, cy: y + h / 2, sx: W / L, sy: H / L, nx: -Math.sin(a), ny: Math.cos(a) };
}

// A point given in fractions of width and height, in units of the longest edge.
const point = (fx, fy, { w, h, L }) => [(fx * w) / L, (fy * h) / L];

// Fill d by sampling s where map(v, us, xs, ys) puts each output pixel of a row, all in units of the longest edge:
// bilinear at pixel coordinates, mirrored at the borders.
function warp(s, d, { w, h, L }, map) {
  const us = new Float64Array(w), xs = new Float64Array(w), ys = new Float64Array(w);
  for (let x = 0; x < w; x++) us[x] = (x + 0.5) / L;
  for (let y = 0, o = 0; y < h; y++) {
    map((y + 0.5) / L, us, xs, ys);
    for (let x = 0; x < w; x++, o += 3) {
      let fx = xs[x] * L - 0.5, fy = ys[x] * L - 0.5;
      if (!(fx >= 0 && fx <= w - 1)) fx = mirror(fx, w);
      if (!(fy >= 0 && fy <= h - 1)) fy = mirror(fy, h);
      const x0 = fx | 0, y0 = fy | 0, x1 = x0 < w - 1 ? x0 + 1 : x0, y1 = y0 < h - 1 ? y0 + 1 : y0;  // fx, fy >= 0: | 0 floors, as an integer
      const ax = fx - x0, ay = fy - y0;
      const i00 = 3 * (y0 * w + x0), i10 = 3 * (y0 * w + x1), i01 = 3 * (y1 * w + x0), i11 = 3 * (y1 * w + x1);
      let top = s[i00] + (s[i10] - s[i00]) * ax, bot = s[i01] + (s[i11] - s[i01]) * ax;
      d[o] = top + (bot - top) * ay;
      top = s[i00 + 1] + (s[i10 + 1] - s[i00 + 1]) * ax; bot = s[i01 + 1] + (s[i11 + 1] - s[i01 + 1]) * ax;
      d[o + 1] = top + (bot - top) * ay;
      top = s[i00 + 2] + (s[i10 + 2] - s[i00 + 2]) * ax; bot = s[i01 + 2] + (s[i11 + 2] - s[i01 + 2]) * ax;
      d[o + 2] = top + (bot - top) * ay;
    }
  }
}

function mirror(t, n) {
  const p = 2 * n - 2;
  if (p <= 0) return 0;
  t = ((t % p) + p) % p;
  return t > n - 1 ? p - t : t;
}

// Three box blurs approximate a Gaussian. Radius in fractions of the longest edge. A fractional pixel radius
// weights the two outermost samples, so small radii still blur. Borders clamp to the edge pixel.
// The blur radius of ink lines, as a fraction of the longest edge.
const EDGE_BLUR = 0.004;
// A blur radius in pixels, or 0 where it is too small to blur.
const blurPixels = (radius, L) => (radius * L < 0.05 ? 0 : radius * L);

function blur(s, radius, ctx) {
  const r = blurPixels(radius, ctx.L);
  if (!r) return s;
  const a = ctx.scratch("blurA"), b = ctx.scratch("blurB");
  for (let pass = 0; pass < 3; pass++) {
    boxRows(pass ? b : s, a, ctx.w, ctx.h, r);
    boxColumns(a, b, ctx.w, ctx.h, r);
  }
  return b;
}

function boxRows(src, out, w, h, radius) {
  const ri = Math.floor(radius), frac = radius - ri, norm = 1 / (2 * radius + 1);
  // The clamped indices are written out, so no function is called per pixel.
  const last = w - 1;
  for (let line = 0; line < h; line++) {
    const base = 3 * line * w;
    let s0 = 0, s1 = 0, s2 = 0;
    for (let k = -ri; k <= ri; k++) { const i = base + 3 * (k < 0 ? 0 : k > last ? last : k); s0 += src[i]; s1 += src[i + 1]; s2 += src[i + 2]; }
    for (let k = 0; k < w; k++) {
      const a = k - ri - 1, b = k + ri + 1, c = k - ri, o = base + 3 * k;
      const lo = base + 3 * (a < 0 ? 0 : a), hi = base + 3 * (b > last ? last : b), rem = base + 3 * (c < 0 ? 0 : c);
      out[o] = (s0 + frac * (src[lo] + src[hi])) * norm;
      out[o + 1] = (s1 + frac * (src[lo + 1] + src[hi + 1])) * norm;
      out[o + 2] = (s2 + frac * (src[lo + 2] + src[hi + 2])) * norm;
      s0 += src[hi] - src[rem]; s1 += src[hi + 1] - src[rem + 1]; s2 += src[hi + 2] - src[rem + 2];
    }
  }
}

// The same down the columns, a row at a time with a running sum per column, so memory is read in order.
function boxColumns(src, out, w, h, radius) {
  const ri = Math.floor(radius), frac = radius - ri, norm = 1 / (2 * radius + 1), row = 3 * w, sums = new Float64Array(row);
  const at = (k) => row * (k < 0 ? 0 : k >= h ? h - 1 : k);
  for (let k = -ri; k <= ri; k++) { const i = at(k); for (let c = 0; c < row; c++) sums[c] += src[i + c]; }
  for (let k = 0; k < h; k++) {
    const lo = at(k - ri - 1), hi = at(k + ri + 1), rem = at(k - ri), o = row * k;
    for (let c = 0; c < row; c++) {
      out[o + c] = (sums[c] + frac * (src[lo + c] + src[hi + c])) * norm;
      sums[c] += src[hi + c] - src[rem + c];
    }
  }
}

// Two measures of a picture on grids of cells along its longest edge, from each cell's luminance mean and variance.
// contrast: the standard deviation of the cell means on a grid of about 12. Under 0.03 a picture reads as empty: a
// flat field, specks on black, or a pale wash, which the scorer can still reward (a grey frame scored high for
// "loneliness"). featureless: the share of cells, on a grid of about 24, whose luminance hardly varies inside
// (standard deviation under 0.02). Above 0.75 a picture reads as empty too: a grey blank, a blur with one oval, a
// white page with a few fragments, which long searches for "loneliness" reached and a blind reviewer saw as the
// picture getting worse, while dark or blown-out takes on the phrase stay under 0.5.
const FLAT = { contrast: 0.03, featureless: 0.75 };

// The grid of cells about `across` along the longest edge: the cell side c in pixels, columns and rows, and each
// cell's pixel count.
function grid(width, height, across) {
  const c = Math.ceil(Math.max(width, height) / across), cols = Math.ceil(width / c), rows = Math.ceil(height / c);
  const n = Float64Array.from({ length: cols * rows }, (_, k) => Math.min(c, width - (k % cols) * c) * Math.min(c, height - Math.floor(k / cols) * c));
  return { c, cols, rows, n };
}

// Each cell's luminance mean and variance, from its sums of luminance and of its square (s, s2) and its count (n).
const fromSums = (s, s2, n) => Array.from(n, (count, k) => ({ mean: s[k] / count, variance: Math.max(0, s2[k] / count - (s[k] / count) ** 2) }));

function cells({ data, width, height }, across) {
  const { c, cols, n } = grid(width, height, across), s = new Float64Array(n.length), s2 = new Float64Array(n.length);
  const col = Int32Array.from({ length: width }, (_, x) => Math.floor(x / c));
  for (let y = 0, i = 0; y < height; y++) {
    const row = Math.floor(y / c) * cols;
    for (let x = 0; x < width; x++, i += 3) {
      const k = row + col[x], v = luma(data[i], data[i + 1], data[i + 2]);
      s[k] += v; s2[k] += v * v;
    }
  }
  return fromSums(s, s2, n);
}

const contrastOf = (all) => {
  const means = all.map((c) => c.mean), m = means.reduce((a, b) => a + b, 0) / means.length;
  return Math.sqrt(means.reduce((a, v) => a + (v - m) ** 2, 0) / means.length);
};
const featurelessOf = (all) => all.filter((c) => c.variance < 0.02 ** 2).length / all.length;
// The grids the two measures use.
const COARSE = 12, FINE = 24;

const contrast = (img) => contrastOf(cells(img, COARSE));
const featureless = (img) => featurelessOf(cells(img, FINE));

// A picture that reads as empty. The limits loosen for a source that is itself plain (a drawing on white), so its
// variants are not all rejected: half its contrast at most, and its featureless share plus 0.05 at least.
export const isFlat = (img, source) => {
  const l = limits(source);
  return contrast(img) < l.contrast || featureless(img) > l.featureless;
};

// How far a picture moved from the source: the mean change of a pixel (the mean of its channels' changes), and the
// share of pixels that changed by more than OVER.
const OVER = 0.3;
function change({ data }, source) {
  let sum = 0, moved = 0;
  for (let i = 0; i < data.length; i += 3) {
    const d = (Math.abs(data[i] - source.data[i]) + Math.abs(data[i + 1] - source.data[i + 1]) + Math.abs(data[i + 2] - source.data[i + 2])) / 3;
    sum += d;
    if (d > OVER) moved++;
  }
  const n = data.length / 3;
  return { mean: sum / n, moved: moved / n };
}

// A faint overlay on an unchanged photograph, which the scorer could still rank first (a ghost in the sky won "la
// mer en hiver": mean change 0.026, 2% of pixels past 0.3), while the variants a blind reviewer saw as changed
// moved 0.06 to 0.47 on average. A strong change inside a small region keeps its share of moved pixels.
const unchanged = (c) => c.mean < 0.05 && c.moved < 0.05;
export const barelyChanged = (img, source) => unchanged(change(img, source));

const known = new WeakMap();
function limits(source) {
  if (!known.has(source)) known.set(source, { contrast: Math.min(FLAT.contrast, contrast(source) / 2), featureless: Math.max(FLAT.featureless, featureless(source) + 0.05) });
  return known.get(source);
}

// The list shuffled in place (Fisher-Yates), the same order in every engine for a seeded rnd.
function shuffle(list, rnd) {
  for (let i = list.length - 1; i > 0; i--) {
    const j = Math.floor(rnd() * (i + 1));
    [list[i], list[j]] = [list[j], list[i]];
  }
  return list;
}

// A seeded generator of numbers in [0, 1) (mulberry32).
export function random(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
  };
}

// Smooth noise in [-1, 1] from a hash of the integer lattice and the seed.
function valueNoise(x, y, seed) {
  const x0 = Math.floor(x), y0 = Math.floor(y), fx = x - x0, fy = y - y0;
  const sx = fx * fx * (3 - 2 * fx), sy = fy * fy * (3 - 2 * fy);
  const a = hash(x0, y0, seed), b = hash(x0 + 1, y0, seed), c = hash(x0, y0 + 1, seed), e = hash(x0 + 1, y0 + 1, seed);
  return a + (b - a) * sx + (c - a) * sy + (a - b - c + e) * sx * sy;
}

function hash(x, y, seed) {
  let h = Math.imul(x, 0x27d4eb2d) ^ Math.imul(y, 0x165667b1) ^ Math.imul(seed, 0x9e3779b1);
  h = Math.imul(h ^ (h >>> 15), 0x85ebca6b);
  h = Math.imul(h ^ (h >>> 13), 0xc2b2ae35);
  return ((h ^ (h >>> 16)) >>> 0) / 2 ** 31 - 1;
}
