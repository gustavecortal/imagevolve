// The unit direction a candidate's change should take: from the neutral text to the prompt, minus the mean of the
// same direction over a generic bank of 430 prompts (direction-bank.mjs), so the score rewards the change this prompt
// asks for rather than the change every prompt asks for (away from a plain photograph, toward a pale texture). The
// bank replaced 24 chosen prompts: no hand-made list favours some prompts, and on held-out prompts its picks were as
// specific, by another model's judgment, and a blind reviewer preferred them 3 to 2. Its mean is
// precomputed for the scorer (direction-bank.json).
import bank from "./direction-bank.json" with { type: "json" };
import { SCORER, sub, unit } from "./objective.mjs";

if (bank.scorer !== SCORER.key) throw new Error("direction-bank.json was made for another scorer: run node direction-bank.mjs");
const bankMean = Float32Array.from(bank.mean);

export const promptDirection = (prompt, neutral) => unit(sub(unit(sub(prompt, neutral)), bankMean));
