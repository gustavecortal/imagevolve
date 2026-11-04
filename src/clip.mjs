// CLIP ViT-B/32 text and image embeddings from ONNX Runtime: the model's own tokenizer, the processor's resizing and
// normalization (pixelValues, below), the projection models' embeddings, L2-normalized. The model files are downloaded
// from the Hub at the revision SCORER pins, checked against their SHA-256, and kept in models/.
import { createHash } from "node:crypto";
import { mkdir, readFile, rename, writeFile } from "node:fs/promises";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { Tokenizer } from "@huggingface/tokenizers";
import * as ort from "onnxruntime-node";

// The scorer: the Hugging Face repository and revision it pins, its model files with their SHA-256, and the neutral
// text directions start from. The text model is Xenova's q4f16 export of CLIP ViT-B/32 with its token table in 8 bits.
// The vision model reads 160 px rather than CLIP's 224 and holds 8-bit weights computed in int8: its direction scores
// follow float32's with a correlation of 0.998 at the median. export.py rebuilds both. 156 MB to download.
const repo = "gustavecortal/imagevolve-clip-vit-base-patch32", revision = "9c0448904d1e00771f11c9bccb195c4c6d19e946", neutral = "a photograph";
const files = { text: "onnx/text_model_q4f16_t8.onnx", vision: "onnx/vision_model_q8a4_160.onnx" }, size = 160;
const sha256 = {
  "tokenizer.json": "f7f3b7af117d467b58374797691a6438d3e6b9e9cef800dfd5dced7f697a90cd",
  "tokenizer_config.json": "60ba2912bc6344c94bc16bbdec27fa1209409167b6f2fdf3cfe9e65462ea3967",
  [files.text]: "7bc26c12ca135542a3d01040f7707bb617ba4629bd2bb44529d5ae7ed451f888",
  [files.vision]: "60a67699aef19887f4cf6c7b076407b2607cfac7c2b0f9c1239c9202121ece63",
};
export const SCORER = {
  repo, revision, files, sha256, size, neutral, key: `${repo}@${revision.slice(0, 7)} "${neutral}"`,
};


const CONTEXT = 77;  // tokens
const SIZE = SCORER.size, PLANE = SIZE * SIZE;
const MEAN = [0.48145466, 0.4578275, 0.40821073], STD = [0.26862954, 0.26130258, 0.27577711];
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
  for (let i = 0; i < n; i++) {
    const row = e.data.slice(i * d, (i + 1) * d), norm = Math.hypot(...row);  // float32, as the page computes it
    out.push(row.map((v) => v / norm));
  }
  return out;
}

// CLIP's image input: the whole picture squashed to SIZE x SIZE by an area average (no crop), then normalized as
// CLIP's processor does, as planar floats.
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
function pixelValues({ data, width, height }) {
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
