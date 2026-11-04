// Minimal probes of the recipe contract: identity, strength 0, determinism, order sensitivity, and validation.
// Usage: node test.mjs   (exits 1 on a failure)
import { OPS, random, render } from "./src/effects.mjs";
import { randomRecipe } from "./src/recipes.mjs";

const rnd = random(42), w = 64, h = 40;
const img = { width: w, height: h, data: Float32Array.from({ length: w * h * 3 }, () => rnd()) };
const same = (a, b) => a.data.every((v, i) => v === b.data[i]);
const throws = (f) => { try { f(); return false; } catch { return true; } };
const checks = {
  "empty recipe is the identity": same(render(img, []), img),
  "strength 0 is the identity for every operation": Object.keys(OPS).every((op) => same(render(img, [{ op, params: {}, strength: 0 }]), img)),
  "a recipe renders the same twice": (() => { const r = randomRecipe(random(7)); return same(render(img, r), render(img, r)); })(),
  "order matters (tiles then swirl differs from swirl then tiles)": !same(
    render(img, [{ op: "tiles", params: {} }, { op: "swirl", params: {} }]), render(img, [{ op: "swirl", params: {} }, { op: "tiles", params: {} }])),
  "a long recipe stays finite and in [0, 1]": (() => { const long = [...randomRecipe(random(9)), ...randomRecipe(random(10))].slice(0, 8), r = render(img, long); return r.data.every((v) => v >= 0 && v <= 1); })(),
  "an unknown operation throws": throws(() => render(img, [{ op: "nope" }])),
  "an unknown parameter throws": throws(() => render(img, [{ op: "tone", params: { satuation: 1 } }])),
  "a non-finite parameter throws": throws(() => render(img, [{ op: "tone", params: { gain: NaN } }])),
  "a non-finite strength throws": throws(() => render(img, [{ op: "tone", strength: "abc" }])),
};
for (const [name, ok] of Object.entries(checks)) console.log(`${ok ? "ok  " : "FAIL"} ${name}`);
process.exit(Object.values(checks).every(Boolean) ? 0 : 1);
