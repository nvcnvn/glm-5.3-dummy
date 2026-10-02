#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["huggingface_hub>=0.34"]
# ///
"""Cut a GLM-5.x checkpoint (GlmMoeDsaForCausalLM) to its first N decoder layers plus the MTP layer.

Tensors are copied byte for byte, so any quantization works. Only the tensors that are kept are
read: from a Hugging Face repo with ranged reads, or from a local directory.
"""

import argparse
import json
import os
import re
import shlex
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, HfFileSystem

LAYER = re.compile(r"model\.layers\.(\d+)")
HF_CACHE_SNAPSHOT = re.compile(r"models--(.+?)--(.+)/snapshots/([0-9a-f]{40})/?$")
NOT_COPIED = {"README.md", ".gitattributes", "config.json", "model.safetensors.index.json"}
PIECE = 128 << 20


@dataclass
class Tensor:
    name: str
    source_name: str
    file: str
    dtype: str
    shape: list
    start: int
    size: int


class Source:
    """A checkpoint on Hugging Face, or in a local directory (a Hugging Face cache snapshot keeps its repo id)."""

    def __init__(self, ref, revision):
        self.local = Path(ref) if Path(ref).is_dir() else None
        if self.local:
            m = HF_CACHE_SNAPSHOT.search(str(self.local.resolve()))
            self.id, self.revision = (f"{m[1]}/{m[2]}", m[3]) if m else (None, revision)
            self.names = sorted(p.name for p in self.local.iterdir() if p.is_file())
        else:
            info = HfApi().model_info(ref, revision=revision)
            self.id, self.revision = ref, info.sha
            self.names = sorted(s.rfilename for s in info.siblings if "/" not in s.rfilename)
            self.fs = HfFileSystem()

    def read(self, name, start=None, end=None):
        if self.local:
            with open(self.local / name, "rb") as f:
                f.seek(start or 0)
                return f.read() if end is None else f.read(end - (start or 0))
        return self.fs.cat_file(f"{self.id}@{self.revision}/{name}", start=start, end=end)

    def header(self, name):
        n = struct.unpack("<Q", self.read(name, 0, 8))[0]
        return 8 + n, json.loads(self.read(name, 8, 8 + n))


def layer_map(keep, total, nextn):
    def new_index(i):
        if i < keep:
            return i
        if total <= i < total + nextn:
            return keep + i - total
        return None
    return new_index


def relabel(s, new_index):
    """s with every layer number mapped, or None if s names a layer that is cut."""
    dropped = False

    def sub(m):
        nonlocal dropped
        j = new_index(int(m[1]))
        dropped |= j is None
        return f"model.layers.{j}"

    out = LAYER.sub(sub, s)
    return None if dropped else out


def relabel_all(obj, new_index):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = relabel(k, new_index)
            if nk is not None:
                out[nk] = relabel_all(v, new_index)
        return out
    if isinstance(obj, list):
        items = [relabel_all(v, new_index) for v in obj]
        return [v for v in items if v is not None]
    if isinstance(obj, str):
        return relabel(obj, new_index)
    return obj


def cut_config(config, keep):
    total = config["num_hidden_layers"]
    new_index = layer_map(keep, total, config.get("num_nextn_predict_layers", 0))
    out = {}
    for k, v in config.items():
        if isinstance(v, list) and len(v) == total:
            v = v[:keep]
        out[k] = v
    out["num_hidden_layers"] = keep
    out["first_k_dense_replace"] = min(config.get("first_k_dense_replace", 0), keep)
    if "quantization_config" in config:
        out["quantization_config"] = relabel_all(config["quantization_config"], new_index)
    return out


def plan_tensors(source, weight_map, new_index):
    by_file = {}
    for name, file in weight_map.items():
        new = relabel(name, new_index)
        if new is not None:
            by_file.setdefault(file, []).append((name, new))
    with ThreadPoolExecutor(16) as pool:
        headers = dict(zip(by_file, pool.map(source.header, by_file)))
    tensors = []
    for file, names in by_file.items():
        base, header = headers[file]
        for name, new in names:
            h = header[name]
            a, b = h["data_offsets"]
            tensors.append(Tensor(new, name, file, h["dtype"], h["shape"], base + a, b - a))
    tensors.sort(key=lambda t: (t.file, t.start))
    return tensors


def check(tensors, weight_map, keep, total, nextn):
    def per_layer(names):
        counts = {}
        for n in names:
            m = LAYER.match(n)
            if m:
                counts[int(m[1])] = counts.get(int(m[1]), 0) + 1
        return counts

    src, out = per_layer(weight_map), per_layer(t.name for t in tensors)
    expected = {i: src[i] for i in range(keep)}
    expected |= {keep + i: src[total + i] for i in range(nextn)}
    if out != expected:
        sys.exit(f"tensor counts per layer differ from the source: {out} != {expected}")
    if len({t.name for t in tensors}) != len(tensors):
        sys.exit("duplicate tensor names after renaming")


def split_shards(tensors, shard_bytes):
    shards, current, size = [], [], 0
    for t in tensors:
        if current and size + t.size > shard_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(t)
        size += t.size
    if current:
        shards.append(current)
    return shards


def shard_header(tensors):
    header, offset = {"__metadata__": {"format": "pt"}}, 0
    for t in tensors:
        header[t.name] = {"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.size]}
        offset += t.size
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    return struct.pack("<Q", len(blob)) + blob


def write_shard(source, path, tensors, threads):
    header = shard_header(tensors)
    runs, at = [], len(header)
    for t in tensors:
        last = runs[-1] if runs else None
        if last and last[0] == t.file and last[1] + last[3] == t.start:
            runs[-1] = (last[0], last[1], last[2], last[3] + t.size)
        else:
            runs.append((t.file, t.start, at, t.size))
        at += t.size
    pieces = [(file, start + o, dst + o, min(PIECE, size - o))
              for file, start, dst, size in runs for o in range(0, size, PIECE)]

    with open(path, "wb") as f:
        f.write(header)
        f.truncate(at)
        fd = f.fileno()

        def copy(piece):
            file, start, dst, size = piece
            data = source.read(file, start, start + size)
            if len(data) != size:
                raise IOError(f"{file}: read {len(data)} of {size} bytes at {start}")
            os.pwrite(fd, data, dst)

        with ThreadPoolExecutor(threads) as pool:
            list(pool.map(copy, pieces))


def provenance():
    if os.environ.get("GITHUB_SHA"):
        repo = f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
        return repo, os.environ["GITHUB_SHA"]
    here = Path(__file__).resolve().parent
    try:
        url = subprocess.run(["git", "remote", "get-url", "origin"], cwd=here, capture_output=True, text=True).stdout.strip()
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True).stdout.strip()
    except OSError:
        return "", ""
    if url.startswith("git@github.com:"):
        url = "https://github.com/" + url.removeprefix("git@github.com:")
    return url.removesuffix(".git"), sha


def model_card(args, source, config, cut, total_bytes):
    repo_url, sha = provenance()
    script = f"[cut.py]({repo_url}/blob/{sha}/cut.py)" if repo_url and sha else "cut.py"
    src = f"[{source.id}](https://huggingface.co/{source.id}/tree/{source.revision})"
    card = HfApi().model_info(args.license_from or source.id).card_data or {}
    license = "\n".join(f"{k}: {card[k]}" for k in ("license", "license_name", "license_link") if card.get(k))
    keep, total = cut["num_hidden_layers"], config["num_hidden_layers"]
    nextn = config.get("num_nextn_predict_layers", 0)
    mtp = f" and the MTP layer {total} (renumbered {keep})" if nextn else ""
    return f"""---
{license}
base_model: {source.id}
tags:
- glm
- testing
- pruned
---

# {args.repo.split("/")[-1]}

**For testing serving infrastructure only. Its output is not useful text.**

{src} (revision `{source.revision}`) cut to its first {keep} of {total} decoder layers{mtp}.
Every kept tensor is copied byte for byte, so the quantization, the attention and DSA indexer
shapes, the KV cache layout, the MoE routing and the speculative-decoding layer are those of
the full model; only the depth differs. {total_bytes / 1e9:.1f} GB.

Built by {script} with:

```
{shlex.join(["cut.py", *sys.argv[1:]])}
```

The original license is in `LICENSE`.
"""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="Hugging Face repo id or local checkpoint directory")
    p.add_argument("--revision", help="source revision (default: the current main)")
    p.add_argument("--layers", type=int, required=True, help="decoder layers to keep, from the first")
    p.add_argument("--out", type=Path, default=Path("out"), help="directory the cut is written to")
    p.add_argument("--repo", help="Hugging Face repo to upload to (shards are deleted locally once uploaded)")
    p.add_argument("--private", action="store_true", help="create --repo as private")
    p.add_argument("--license-from", help="Hugging Face repo to take LICENSE from when the source has none")
    p.add_argument("--shard-gb", type=float, default=5)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--dry-run", action="store_true", help="read only the headers and print the plan")
    args = p.parse_args()

    source = Source(args.source, args.revision)
    if args.repo and not source.id:
        sys.exit(f"{args.source} is not a Hugging Face cache snapshot: the model card needs its repo id")
    config = json.loads(source.read("config.json"))
    total, nextn = config["num_hidden_layers"], config.get("num_nextn_predict_layers", 0)
    if not 0 < args.layers < total:
        sys.exit(f"--layers must be between 1 and {total - 1}")
    new_index = layer_map(args.layers, total, nextn)
    if "model.safetensors.index.json" in source.names:
        weight_map = json.loads(source.read("model.safetensors.index.json"))["weight_map"]
    else:
        weight_map = {k: "model.safetensors" for k in source.header("model.safetensors")[1] if k != "__metadata__"}

    tensors = plan_tensors(source, weight_map, new_index)
    check(tensors, weight_map, args.layers, total, nextn)
    shards = split_shards(tensors, int(args.shard_gb * 1e9))
    names = [f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors" for i in range(len(shards))]
    total_bytes = sum(t.size for t in tensors)
    cut = cut_config(config, args.layers)
    print(f"{source.id or args.source}@{source.revision}: {len(tensors)} of {len(weight_map)} tensors, "
          f"{total_bytes / 1e9:.1f} GB in {len(shards)} shards", flush=True)
    if args.dry_run:
        print(json.dumps({k: v for k, v in cut.items() if v != config.get(k)}, indent=1))
        return

    extra = [n for n in source.names if n not in NOT_COPIED and not n.endswith(".safetensors")]
    license_source = None
    if "LICENSE" not in extra:
        if not args.license_from:
            sys.exit("the source has no LICENSE: pass --license-from")
        license_source = Source(args.license_from, None)

    api = HfApi()
    uploaded = {}
    if args.repo:
        api.create_repo(args.repo, private=args.private, exist_ok=True)
        uploaded = {f.path: f.size for f in api.list_repo_tree(args.repo) if hasattr(f, "size")}

    args.out.mkdir(parents=True, exist_ok=True)
    for name, shard in zip(names, shards):
        size = sum(t.size for t in shard)
        path = args.out / name
        if uploaded.get(name) == len(shard_header(shard)) + size:
            print(f"{name}: already uploaded", flush=True)
            continue
        t0 = time.time()
        write_shard(source, path, shard, args.threads)
        t1 = time.time()
        print(f"{name}: {size / 1e9:.2f} GB read in {t1 - t0:.0f} s ({size / 1e6 / (t1 - t0):.0f} MB/s)", flush=True)
        if args.repo:
            api.upload_file(path_or_fileobj=path, path_in_repo=name, repo_id=args.repo, commit_message=f"Add {name}")
            path.unlink()
            print(f"{name}: uploaded in {time.time() - t1:.0f} s", flush=True)

    index = {"metadata": {"total_size": total_bytes},
             "weight_map": {t.name: n for n, shard in zip(names, shards) for t in shard}}
    (args.out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")
    (args.out / "config.json").write_text(json.dumps(cut, indent=2) + "\n")
    for n in extra:
        (args.out / n).write_bytes(source.read(n))
    if license_source:
        (args.out / "LICENSE").write_bytes(license_source.read("LICENSE"))
    if args.repo:
        (args.out / "README.md").write_text(model_card(args, source, config, cut, total_bytes))
        small = ["model.safetensors.index.json", "config.json", "README.md", *extra, *(["LICENSE"] if license_source else [])]
        ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=args.out / n) for n in small]
        ops += [CommitOperationDelete(path_in_repo=n) for n in uploaded if n.endswith(".safetensors") and n not in names]
        api.create_commit(args.repo, operations=ops, commit_message=f"Cut {source.id}@{source.revision} to {args.layers} layers")
        print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
