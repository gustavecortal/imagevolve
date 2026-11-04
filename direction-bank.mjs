// Writes src/bank.json: the mean of the unit directions from the neutral text to each phrase of src/bank.txt,
// which src/score.mjs removes from every prompt's direction. Run again whenever SCORER changes.
// Usage: node direction-bank.mjs
import { readFileSync, writeFileSync } from "node:fs";
import { SCORER, loadClip } from "./src/clip.mjs";
import { sub, unit } from "./src/score.mjs";

const phrases = readFileSync(new URL("src/bank.txt", import.meta.url), "utf8").trim().split("\n");
const scorer = await loadClip(), [neutral, ...bank] = await scorer.embedTexts([SCORER.neutral, ...phrases]);
const mean = new Float32Array(neutral.length);
for (const t of bank) { const u = unit(sub(t, neutral)); for (let i = 0; i < mean.length; i++) mean[i] += u[i] / bank.length; }
writeFileSync(new URL("src/bank.json", import.meta.url), JSON.stringify({
  scorer: SCORER.key, prompts: phrases.length, mean: Array.from(mean, (x) => Math.round(x * 1e6) / 1e6),
}));
console.log(`src/bank.json: ${phrases.length} phrases, mean direction length ${Math.hypot(...mean).toFixed(3)}`);
