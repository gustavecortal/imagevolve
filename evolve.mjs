// Evolve a photograph toward a phrase and save the eight variants the search keeps.
//
//   node evolve.mjs photo.jpg "a storm at sea"                    random recipes
//   node evolve.mjs photo.jpg "a storm at sea" --keep out/3.json  children of a kept variant, bred from its recipe
//
// Run with --help for the options.
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import sharp from "sharp";
import { SCORER, loadClip } from "./src/clip.mjs";
import { WORKING, normalized, random, render } from "./src/effects.mjs";
import { promptDirection } from "./src/score.mjs";
import { entry, select, step } from "./src/search.mjs";

const POOL = 8, BATCH = 8;
const USAGE = `Usage: node evolve.mjs <photo> "<phrase>" [options]
  --proposals N  variants to try (800)
  --seed N       seed of the random programs (1)
  --out DIR      where the eight kept variants go (out)
  --size N       longest edge of the saved pictures, in pixels (2048)
  --keep FILE    breed from a kept variant's .json, made from the same photo`;
const usage = (code) => { (code ? console.error : console.log)(USAGE); process.exit(code); };
let parsed;
try {
  parsed = parseArgs({
    allowPositionals: true,
    options: {
      proposals: { type: "string", default: "800" }, seed: { type: "string", default: "1" },
      out: { type: "string", default: "out" }, size: { type: "string", default: "2048" }, keep: { type: "string" },
      help: { type: "boolean", short: "h" },
    },
  });
} catch (error) {
  console.error(error.message);
  usage(1);
}
const { values: opts, positionals: [photo, phrase] } = parsed;
if (opts.help) usage(0);
const [proposals, seed, size] = [opts.proposals, opts.seed, opts.size].map(Number);
if (!photo || !phrase || ![proposals, seed, size].every(Number.isInteger) || proposals < 1 || size < 1) usage(1);

// The photograph, oriented by its EXIF tag, fitted inside long x long pixels as float RGB.
const photograph = sharp(photo).rotate();
async function load(long) {
  const { data, info } = await photograph.clone().resize(long, long, { fit: "inside", withoutEnlargement: true })
    .removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const floats = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) floats[i] = data[i] / 255;
  return { data: floats, width: info.width, height: info.height };
}
function save({ data, width, height }, path) {
  const bytes = new Uint8ClampedArray(data.length);  // rounds and clamps to 0..255
  for (let i = 0; i < data.length; i++) bytes[i] = data[i] * 255;
  return sharp(bytes, { raw: { width, height, channels: 3 } }).jpeg({ quality: 92 }).toFile(path);
}

const scorer = await loadClip();
const source = await load(WORKING), [orig] = await scorer.embedImages([source]);
const [neutral, words] = await scorer.embedTexts([SCORER.neutral, phrase]);
const direction = promptDirection(words, neutral);
const options = { source, orig, direction, embed: scorer.embedImages, rnd: random(seed), batch: BATCH };

// A kept variant is pinned: the search breeds three proposals in four from it.
let pool = [];
if (opts.keep) {
  const recipe = normalized(JSON.parse(readFileSync(opts.keep, "utf8")).recipe);
  const [emb] = await scorer.embedImages([render(source, recipe)]);
  pool = [{ ...entry(recipe, emb, options), pinned: true }];
}

const started = performance.now();
for (let n = 0; n < proposals; n += BATCH) {
  pool = select([...pool, ...(await step(pool, options))], POOL);
  const best = Math.max(...pool.map((e) => e.score));
  process.stderr.write(`\r${n + BATCH} proposals, best score ${pool.length ? best.toFixed(3) : "-"}`);
}
console.error(` in ${((performance.now() - started) / 1000).toFixed(0)} s`);

mkdirSync(opts.out, { recursive: true });
const full = await load(size);
await Promise.all(pool.map(({ recipe, score }, k) => {
  writeFileSync(`${opts.out}/${k + 1}.json`, JSON.stringify({ phrase, score: Math.round(score * 1e4) / 1e4, recipe }, null, 1));
  return save(render(full, recipe), `${opts.out}/${k + 1}.jpg`);
}));
console.log(`${pool.length} variants in ${opts.out}/`);
