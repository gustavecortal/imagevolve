# /// script
# requires-python = ">=3.11"
# dependencies = ["huggingface-hub", "numpy", "onnx==1.23.2", "onnx-ir==1.1.0", "onnxruntime==1.31.1"]
# ///
"""Rebuild the scorer's model files from Xenova's ONNX export of CLIP ViT-B/32, and check that they hold the same
graphs and weights as the files on the Hub, which evolve.mjs downloads.

    uv run export.py [out_dir]          # default: export

- Vision: the float32 model reads 160 px instead of 224 (its 7x7 grid of position embeddings resized to 5x5,
  bilinear), then every weight MatMul is quantized to 8 bits (block 32, symmetric) computed in int8.
- Text: Xenova's q4f16 model with its fp16 token table (49408 x 512) stored as int8 rows, each with an fp16 scale.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from huggingface_hub import hf_hub_download
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

SOURCE, REVISION = "Xenova/clip-vit-base-patch32", "d15189d7028b43f1d3e65039190477f6af591c2a"
# The files evolve.mjs downloads (SCORER in src/clip.mjs).
TARGET, TARGET_REVISION = "gustavecortal/imagevolve-clip-vit-base-patch32", "9c0448904d1e00771f11c9bccb195c4c6d19e946"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "export")
FILES = ["tokenizer.json", "tokenizer_config.json", "onnx/vision_model_q8a4_160.onnx", "onnx/text_model_q4f16_t8.onnx"]
FROM, TO, WIDTH = 7, 5, 768  # position grid before and after, embedding width


def fetch(path, repo=SOURCE, revision=REVISION):
    return hf_hub_download(repo, path, revision=revision)


def small_input_vision(model):
    """The vision model for 160 px input: 1 + 5x5 position embeddings resampled from 1 + 7x7, and the shapes after."""
    g = model.graph
    init = {t.name: t for t in g.initializer}
    table = numpy_helper.to_array(init["vision_model.embeddings.position_embedding.weight"]).astype(np.float64)
    grid = table[1:].reshape(FROM, FROM, WIDTH)
    out = np.empty((1 + TO * TO, WIDTH), np.float64)
    out[0] = table[0]  # the class token's
    for y in range(TO):
        for x in range(TO):
            # Bilinear, sampling the old grid at the new cells' centres.
            fy, fx = (y + 0.5) * FROM / TO - 0.5, (x + 0.5) * FROM / TO - 0.5
            y0, x0 = max(0, int(np.floor(fy))), max(0, int(np.floor(fx)))
            y1, x1 = min(FROM - 1, y0 + 1), min(FROM - 1, x0 + 1)
            ay, ax = min(1, max(0, fy - y0)), min(1, max(0, fx - x0))
            out[1 + y * TO + x] = (1 - ay) * ((1 - ax) * grid[y0, x0] + ax * grid[y0, x1]) + ay * ((1 - ax) * grid[y1, x0] + ax * grid[y1, x1])
    replace(g, "vision_model.embeddings.position_embedding.weight", out.astype(np.float32))
    replace(g, "vision_model.embeddings.position_ids", np.arange(1 + TO * TO, dtype=np.int64)[None])

    # Reshape targets that name the token count, as initializers or Constant nodes.
    shapes = {n.input[1] for n in g.node if n.op_type == "Reshape"}
    tensors = [t for t in g.initializer if t.name in shapes]
    tensors += [a.t for n in g.node if n.op_type == "Constant" and n.output[0] in shapes for a in n.attribute if a.name == "value"]
    for t in tensors:
        v = numpy_helper.to_array(t)
        if t.data_type == TensorProto.INT64 and (v == 1 + FROM * FROM).any():
            t.CopyFrom(numpy_helper.from_array(np.where(v == 1 + FROM * FROM, 1 + TO * TO, v), t.name))
    del g.value_info[:]
    return model


def replace(graph, name, array):
    t = next(t for t in graph.initializer if t.name == name)
    t.CopyFrom(numpy_helper.from_array(array, name))


def eight_bit(model):
    quantizer = MatMulNBitsQuantizer(model, bits=8, block_size=32, is_symmetric=True, accuracy_level=4, op_types_to_quantize=("MatMul",))
    quantizer.process()
    return quantizer.model.model


def int8_token_table(model):
    """The one Gather reading the fp16 token table becomes Gather(int8) -> Cast(fp16) -> Mul(Gather(row scale))."""
    g, name = model.graph, "text_model.embeddings.token_embedding.weight"
    table = next(t for t in g.initializer if t.name == name)
    [old] = [n for n in g.node if name in n.input]
    assert old.op_type == "Gather" and table.data_type == TensorProto.FLOAT16
    rows = numpy_helper.to_array(table).astype(np.float64)
    scales = (np.abs(rows).max(axis=1) / 127).astype(np.float16)
    scales[~(scales > 0)] = np.float16(1e-4)  # a row of near zeros, whose scale would round to 0 in fp16
    q = np.clip(np.floor(rows / scales[:, None].astype(np.float64) + 0.5), -127, 127).astype(np.int8)
    g.initializer.remove(table)
    g.initializer.extend([numpy_helper.from_array(q, name + "_int8"), numpy_helper.from_array(scales[:, None], name + "_scale")])
    ids, at = old.input[1], list(g.node).index(old)
    g.node.remove(old)
    for k, node in enumerate([
        helper.make_node("Gather", [name + "_int8", ids], ["token_int8_rows"], "token_int8_gather", axis=0),
        helper.make_node("Cast", ["token_int8_rows"], ["token_rows_f16"], "token_int8_cast", to=TensorProto.FLOAT16),
        helper.make_node("Gather", [name + "_scale", ids], ["token_row_scales"], "token_scale_gather", axis=0),
        helper.make_node("Mul", ["token_rows_f16", "token_row_scales"], [old.output[0]], "token_dequantize"),
    ]):
        g.node.insert(at + k, node)
    return model


def content(path):
    """A file's bytes, or for a model the bytes ONNX's own serializer gives it: protobuf libraries may order a model's
    bytes differently while holding the same graph and weights."""
    return onnx.load(path).SerializeToString() if path.suffix == ".onnx" else path.read_bytes()


if __name__ == "__main__":
    (OUT / "onnx").mkdir(parents=True, exist_ok=True)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copyfile(fetch(name), OUT / name)
    onnx.save(eight_bit(small_input_vision(onnx.load(fetch("onnx/vision_model.onnx")))), OUT / "onnx/vision_model_q8a4_160.onnx")
    onnx.save(int8_token_table(onnx.load(fetch("onnx/text_model_q4f16.onnx"))), OUT / "onnx/text_model_q4f16_t8.onnx")
    for path in FILES:
        same = content(OUT / path) == content(Path(fetch(path, TARGET, TARGET_REVISION)))
        print(f"{'same as the Hub' if same else 'DIFFERENT from the Hub'}  {path}")
