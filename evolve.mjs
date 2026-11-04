// Evolve a photograph toward a phrase, as the web page does, and save the eight variants it keeps.
//
//   node evolve.mjs photo.jpg "a storm at sea"                  random recipes, as before a tap on the page
//   node evolve.mjs photo.jpg "a storm at sea" --keep out/3.json  breed from variant 3, as after a tap
//
// Options: --proposals N (default 800), --seed N (default 1), --out DIR (default out), --size N, the longest edge of
// the saved pictures in pixels (default 2048).
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import sharp from "sharp";
import { loadClip } from "./src/clip.mjs";
import { promptDirection } from "./src/direction.mjs";
import { WORKING, normalized, random, render } from "./src/effects.mjs";
import { SCORER, dot, sub } from "./src/objective.mjs";
import { select, step } from "./src/search.mjs";

const POOL = 8, BATCH = 8;
const { values: opts, positionals: [photo, phrase] } = parseArgs({
  allowPositionals: true,
  options: {
    proposals: { type: "string", default: "800" }, seed: { type: "string", default: "1" },
    out: { type: "string", default: "out" }, size: { type: "string", default: "2048" }, keep: { type: "string" },
  },
});
const [proposals, seed, size] = [opts.proposals, opts.seed, opts.size].map(Number);
if (!photo || !phrase || ![proposals, seed, size].every(Number.isInteger) || proposals < 1 || size < 1) {
  console.error('Usage: node evolve.mjs <photo> "<phrase>" [--proposals N] [--seed N] [--out DIR] [--size N] [--keep FILE]');
  process.exit(1);
}

// A photograph oriented by its EXIF tag and fitted inside long x long pixels, as float RGB.
async function load(path, long) {
  const { data, info } = await sharp(path).rotate().resize(long, long, { fit: "inside", withoutEnlargement: true })
    .removeAlpha().raw().toBuffer({ resolveWithObject: true });
  return { data: Float32Array.from(data, (v) => v / 255), width: info.width, height: info.height };
}
const save = (img, path) =>
  sharp(Buffer.from(Uint8ClampedArray.from(img.data, (v) => v * 255)), { raw: { width: img.width, height: img.height, channels: 3 } })
    .jpeg({ quality: 92 }).toFile(path);

const scorer = await loadClip();
const source = await load(photo, WORKING), [orig] = await scorer.embedImages([source]);
const [neutral, words] = await scorer.embedTexts([SCORER.neutral, phrase]);
const direction = promptDirection(words, neutral);
const options = { source, orig, direction, embed: scorer.embedImages, rnd: random(seed), batch: BATCH, id: ((n) => () => n++)(0) };

// A kept variant, pinned: the search breeds three proposals in four from it.
let pool = [];
if (opts.keep) {
  const recipe = normalized(JSON.parse(readFileSync(opts.keep, "utf8")).recipe);
  const [emb] = await scorer.embedImages([render(source, recipe)]), delta = sub(emb, orig);
  pool = [{ id: -1, recipe, emb, delta, score: dot(delta, direction), pinned: true }];
}

const started = performance.now();
for (let n = 0; n < proposals; n += BATCH) {
  pool = select([...pool, ...(await step(pool, options))], POOL);
  const best = Math.max(...pool.map((entry) => entry.score));
  process.stderr.write(`\r${n + BATCH} proposals, best score ${pool.length ? best.toFixed(3) : "-"}`);
}
console.error(` in ${((performance.now() - started) / 1000).toFixed(0)} s`);

mkdirSync(opts.out, { recursive: true });
const full = await load(photo, size);
for (const [k, { recipe, score }] of pool.entries()) {
  await save(render(full, recipe), `${opts.out}/${k + 1}.jpg`);
  writeFileSync(`${opts.out}/${k + 1}.json`, JSON.stringify({ phrase, score: Math.round(score * 1e4) / 1e4, recipe }, null, 1));
}
console.log(`${pool.length} variants in ${opts.out}/`);
