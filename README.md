# imagevolve

[imagevolve](https://gustavecortal.com/imagevolve/) is a CLIP-guided evolutionary search over deterministic image transformations that bring your picture closer to your words. This repository runs the same search from the command line.

<p align="center">
  <a href="https://gustavecortal.com/imagevolve/"><img src="docs/demo.webp" width="720" alt="imagevolve on &quot;a storm at sea&quot;: a photograph of a carved wooden frame turns into swirling teal variants, one of which is picked and evolved further"></a>
</p>

## Quick start

With [Node.js](https://nodejs.org/) 24 or later:

```bash
git clone https://github.com/gustavecortal/imagevolve.git
cd imagevolve
npm install
node evolve.mjs docs/sample.jpg "a storm at sea"
```

The first run downloads the CLIP model (156 MB). The search then tries 800 variants on the CPU in under a minute, and saves the eight it keeps to `out/`. To evolve one of them further, pass its program:

```bash
node evolve.mjs docs/sample.jpg "a storm at sea" --keep out/3.json --out out2
```

## How it works

A variant is a program of a few image operations, such as tile shuffles, swirls or recolourings. No model draws anything: [CLIP](https://github.com/openai/CLIP) only scores each variant by how far it moves the picture toward your words. The search keeps the eight best variants that do not look alike. After you pick one, most new proposals mutate it: a parameter changes, or an operation is added, removed or borrowed from another variant.

## Rebuild the model

With [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv run export.py
```

This rebuilds the [CLIP model files](https://huggingface.co/gustavecortal/imagevolve-clip-vit-base-patch32), cut down for speed, from [Xenova's ONNX export](https://huggingface.co/Xenova/clip-vit-base-patch32), and checks them against the ones on Hugging Face.

## License

The code is under the [MIT license](LICENSE). The sample photograph is mine, under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
