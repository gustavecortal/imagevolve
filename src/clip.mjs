// CLIP ViT-B/32 text and image embeddings from ONNX Runtime: the model's own tokenizer, the processor's resizing and
// normalization (clip-input.mjs), the projection models' embeddings, L2-normalized. The model files are downloaded
// from the Hub at the revision objective.mjs pins, checked against their SHA-256, and kept in models/.
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-node";
import { SIZE, pixelValues } from "./clip-input.mjs";
import { SCORER, unit } from "./objective.mjs";

const CONTEXT = 77;  // tokens
const END = 49407;  // <|endoftext|>

// A file of the scorer's repository as bytes: the copy in models/ when its SHA-256 is the pinned one, else the Hub's.
async function read(path) {
  const local = fileURLToPath(new URL(`../models/${SCORER.repo}/${path}`, import.meta.url));
  const pinned = (bytes) => createHash("sha256").update(bytes).digest("hex") === SCORER.sha256[path];
  const cached = await readFile(local).catch(() => null);
  if (cached && pinned(cached)) return new Uint8Array(cached);
  console.error(`Downloading ${path}`);
  const response = await fetch(`https://huggingface.co/${SCORER.repo}/resolve/${SCORER.revision}/${path}`);
  if (!response.ok) throw new Error(`${path}: HTTP ${response.status}`);
  const bytes = Buffer.from(await response.arrayBuffer());
  if (!pinned(bytes)) throw new Error(`${path}: wrong SHA-256`);
  // Written whole, then renamed, so an interrupted run leaves no partial file behind.
  await mkdir(dirname(local), { recursive: true });
  await writeFile(`${local}.part`, bytes);
  await rename(`${local}.part`, local);
  return new Uint8Array(bytes);
}

// The scorer: embedTexts(texts) and embedImages(images), each giving unit vectors.
export async function loadClip() {
  const [tokenizerJson, tokenizerConfig, textBytes, visionBytes] = await Promise.all(
    ["tokenizer.json", "tokenizer_config.json", SCORER.files.text, SCORER.files.vision].map(read),
  );
  const json = (bytes) => JSON.parse(new TextDecoder().decode(bytes));
  const tokenizer = new Tokenizer(json(tokenizerJson), json(tokenizerConfig));
  const options = { executionProviders: ["cpu"], logSeverityLevel: 3 };
  const textModel = await ort.InferenceSession.create(textBytes, options);
  const visionModel = await ort.InferenceSession.create(visionBytes, options);

  return {
    // Texts padded with the end token, which changes nothing: the text model attends causally and pools at the
    // first end token (the largest id). A text longer than the context keeps its end token, as CLIP's tokenizer
    // truncates before adding it.
    async embedTexts(texts) {
      const ids = texts.map((t) => {
        const x = tokenizer.encode(t).ids;
        return x.length > CONTEXT ? [...x.slice(0, CONTEXT - 1), END] : x;
      });
      const n = Math.max(...ids.map((x) => x.length)), flat = new BigInt64Array(texts.length * n).fill(BigInt(END));
      ids.forEach((x, k) => x.forEach((id, j) => (flat[k * n + j] = BigInt(id))));
      const input = new ort.Tensor("int64", flat, [texts.length, n]);
      return rows(await textModel.run({ input_ids: input }), "text_embeds");
    },
    // images: [{ data, width, height }], RGB as floats in [0, 1].
    async embedImages(images) {
      const n = 3 * SIZE * SIZE, out = new Float32Array(images.length * n);
      images.forEach((img, k) => out.set(pixelValues(img), k * n));
      const input = new ort.Tensor("float32", out, [images.length, 3, SIZE, SIZE]);
      return rows(await visionModel.run({ pixel_values: input }), "image_embeds");
    },
  };
}

// One output's rows, L2-normalized.
function rows(outputs, name) {
  const e = outputs[name], [n, d] = e.dims, out = [];
  for (let i = 0; i < n; i++) out.push(unit(e.data.slice(i * d, (i + 1) * d)));
  return out;
}
