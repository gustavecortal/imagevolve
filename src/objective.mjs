// What a candidate is scored against, and the vector arithmetic for it.

// The scorer: the Hugging Face repository and revision it pins, its model files with their SHA-256, and the neutral
// text directions start from. The text model is Xenova's q4f16 export of CLIP ViT-B/32 with its token table in 8 bits.
// The vision model reads 160 px rather than CLIP's 224 and holds 8-bit weights computed in int8: its direction scores
// follow float32's at a correlation of 0.998. export.py rebuilds both. 154 MB to download.
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

export const sub = (a, b) => a.map((v, i) => v - b[i]);
export const unit = (a) => { const n = Math.hypot(...a); return a.map((v) => v / n); };

// Cosine similarity of two unit vectors.
export const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

