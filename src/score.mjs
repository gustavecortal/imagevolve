// What a candidate is scored against: its change along a text direction, and the vector maths for it.
import bank from "./bank.json" with { type: "json" };
import { SCORER } from "./clip.mjs";

export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const unit = (a) => { const n = Math.hypot(...a); return a.map((v) => v / n); };

// Cosine similarity of two unit vectors.
export const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

// The unit direction a candidate's change should take: from the neutral text to the prompt, minus the mean of the
// same direction over a generic bank of 430 prompts (src/bank.txt), so the score rewards the change this prompt
// asks for rather than the change every prompt asks for (away from a plain photograph, toward a pale texture). A large
// generic bank, unlike a hand-made list, favours no prompt. Its mean is precomputed in bank.json by direction-bank.mjs.

if (bank.scorer !== SCORER.key) throw new Error("bank.json was made for another scorer: run node direction-bank.mjs");
const bankMean = Float32Array.from(bank.mean);

export const promptDirection = (prompt, neutral) => unit(sub(unit(sub(prompt, neutral)), bankMean));
