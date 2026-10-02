# glm-5.3-dummy

Small GLM-5.3 checkpoints for testing serving infrastructure (HiCache, DP attention, EAGLE, KV
transfer) without loading the full model. Their output is not useful text.

`cut.py` keeps the first N decoder layers of a GLM-5.x checkpoint (`GlmMoeDsaForCausalLM`) plus
its MTP layer, renumbered to N. Kept tensors are copied byte for byte, so every quantization
works and everything except the depth is the full model's: the attention and DSA indexer shapes,
the KV cache layout, the MoE routing, the quantized kernels, the speculative-decoding layer.
`config.json` gets the new `num_hidden_layers`, the per-layer lists cut to N, and layer numbers in
`quantization_config` remapped.

Cutting GLM-5.3 to 8 layers keeps the 3 dense layers and 5 MoE layers, so both kinds of DSA
indexer layer (`full`, `shared`) are present.

## Models

| Repo | Source | Layers | Size |
|---|---|---|---|
| [nvcnvn/GLM-5.3-W4AFP8-8L](https://huggingface.co/nvcnvn/GLM-5.3-W4AFP8-8L) | [PhalaCloud/GLM-5.3-W4AFP8](https://huggingface.co/PhalaCloud/GLM-5.3-W4AFP8) | 8 + MTP | 36 GB |

Each model card names the source revision and the commit of `cut.py` that built it.

## Building one

The workflow **cut** (Actions → cut → Run workflow) builds one cut and uploads it. It reads only
the tensors it keeps, from the source repo with ranged reads, and uploads one shard at a time, so
the runner needs no more than one shard (5 GB) of disk. Rerunning it skips shards already uploaded.
It needs the repository secret `HF_TOKEN`: a Hugging Face fine-grained token with write access
to the target repo's namespace.

Locally (writes to `out/`, uploads only with `--repo`):

```sh
uv run cut.py --source PhalaCloud/GLM-5.3-W4AFP8 --layers 8 --dry-run
uv run cut.py --source PhalaCloud/GLM-5.3-W4AFP8 --layers 8 --license-from zai-org/GLM-5.3
```

`--source` also takes a local checkpoint directory. A Hugging Face cache snapshot
(`.../models--<org>--<name>/snapshots/<sha>`) keeps its repo id and revision for the model card.

## License

The scripts are under Apache-2.0 (`LICENSE`). Each model carries the license of the model it was
cut from.
