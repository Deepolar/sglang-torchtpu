# SPDX-License-Identifier: Apache-2.0
"""RunAI loader for DeepSeek-V4.1 Flash with EP-local reads.

What each rank reads is decided by ``ep_local_expert_reader``; this class only
wires it into ``RunaiModelStreamerLoader``. Forked from
sgl-project/sglang-torchtpu#1744 (e0a43ce363, sglang_tpu/dsv41/loader.py).
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from pathlib import Path
from time import perf_counter

from sglang.srt.model_loader.loader import RunaiModelStreamerLoader
from sglang_tpu.models.deepseek_v41.ep_local_expert_reader import (
    EpLocalExpertReader,
    plan_expert_reads,
)

logger = logging.getLogger(__name__)


class DSV41RunaiModelStreamerLoader(RunaiModelStreamerLoader):
    """Reuse construction/postprocessing; specialize only primary weight I/O."""

    def __init__(self, load_config, model_config):
        super().__init__(load_config)
        self._primary_model_config = model_config

    def _get_weights_iterator(self, source):
        config = getattr(source.model_config, "hf_config", None)
        if (
            getattr(source.model_config, "is_draft_model", False)
            and getattr(config, "model_type", None) == "deepseek_v41"
            and self.load_config.draft_model_idx is None
        ):
            yield from self._draft_weights(source)
            return
        if (
            source.model_config is not self._primary_model_config
            or getattr(config, "model_type", None) != "deepseek_v41"
            or getattr(source.model_config, "is_draft_model", False)
            or source.model_or_path
            != getattr(
                self._primary_model_config,
                "model_weights",
                self._primary_model_config.model_path,
            )
            or source.revision != self._primary_model_config.revision
            or source.prefix
            or self.load_config.draft_model_idx is not None
        ):
            yield from super()._get_weights_iterator(source)
            return

        from runai_model_streamer import SafetensorsStreamer

        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR
        from sglang.srt.runtime_context import get_parallel
        from sglang.srt.utils.runai_utils import is_runai_obj_uri

        if (
            "tensor_names"
            not in inspect.signature(SafetensorsStreamer.stream_files).parameters
        ):
            raise RuntimeError(
                "V4.1 selective loading requires RunAI tensor_names support (>= 0.17.0)"
            )
        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        if is_runai_obj_uri(folder):
            raise ValueError(
                "V4.1 EP-local reads need a local or mounted checkpoint path"
            )
        weight_map = json.loads(
            (Path(folder) / "model.safetensors.index.json").read_text()
        )["weight_map"]
        parallel = get_parallel()
        plan = plan_expert_reads(
            weight_map,
            num_layers=config.num_hidden_layers,
            num_experts=config.n_routed_experts,
            ep_size=parallel.moe_ep_size,
            ep_rank=parallel.moe_ep_rank,
        )
        reader = EpLocalExpertReader(
            weight_map,
            files,
            plan,
            tp_rank=parallel.tp_rank,
            tp_size=parallel.tp_size,
        )
        stream_files = reader.stream_files
        logger.info(
            "V4.1 EP rank %d/%d: reading %d/%d tensors (%d local expert tensors, "
            "%d Engram row ranges) from %d shards",
            parallel.moe_ep_rank,
            parallel.moe_ep_size,
            len(plan.tensor_names),
            plan.total_tensors,
            plan.local_expert_tensors,
            len(reader.row_tables),
            len(stream_files),
        )
        # EP sets differ per rank, and RunAI distributed streaming requires
        # identical tensor_names, so each rank streams independently. Keep
        # owned=True (the default): loaders may retain CPU expert tensors.
        start = perf_counter()
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                stream_files,
                device="cpu",
                is_distributed=False,
                tensor_names=reader.streamed,
            )
            for name, tensor in streamer.get_tensors():
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield name, tensor
        logger.info(
            "V4.1 EP rank %d: streamed tensors consumed in %.1f s",
            parallel.moe_ep_rank,
            perf_counter() - start,
        )
        start = perf_counter()
        yield from reader.iter_row_tables()
        logger.info(
            "V4.1 EP rank %d: Engram row ranges consumed in %.1f s",
            parallel.moe_ep_rank,
            perf_counter() - start,
        )

    def _draft_weights(self, source):
        """The bundled DSpark draft reads only ``mtp.*`` tensors (its loader drops
        the rest); streaming the whole checkpoint needs a RunAI buffer as large
        as the ~100 GB Engram tables."""
        from runai_model_streamer import SafetensorsStreamer

        from sglang.srt.model_loader.weight_utils import RUNAI_STREAMER_TENSOR_ATTR

        folder, files = self._prepare_weights(source.model_or_path, source.revision)
        weight_map = json.loads(
            (Path(folder) / "model.safetensors.index.json").read_text()
        )["weight_map"]
        names = {n for n in weight_map if n.startswith("mtp.")}
        if not names:
            raise ValueError(
                "DSpark draft: the checkpoint index lists no mtp.* tensors"
            )
        names, local_experts = _draft_ep_local_names(
            names, config=source.model_config.hf_config
        )
        by_name = {path.rsplit("/", 1)[-1]: path for path in files}
        shards = sorted({by_name[weight_map[n]] for n in names})
        logger.info(
            "V4.1 DSpark draft: reading %d mtp tensors (%d local expert tensors) "
            "from %d shards",
            len(names),
            local_experts,
            len(shards),
        )
        with SafetensorsStreamer() as streamer:
            streamer.stream_files(
                shards, device="cpu", is_distributed=False, tensor_names=names
            )
            for name, tensor in streamer.get_tensors():
                setattr(tensor, RUNAI_STREAMER_TENSOR_ATTR, True)
                yield source.prefix + name, tensor


_DRAFT_EXPERT = re.compile(
    r"^mtp\.(0|[1-9]\d*)\.ffn\.experts\.(0|[1-9]\d*)\.w[123]\.(weight|scale)$"
)


def _draft_ep_local_names(names: set[str], *, config) -> tuple[set[str], int]:
    """Drop the draft's routed experts outside this EP rank's range.

    The draft MoE shards over the target's EP group (TpuDsv41DsparkDraftWorker),
    so a rank keeps experts [rank * E / ep, (rank + 1) * E / ep), as the target's
    plan_expert_reads does; vllm-torchtpu filters its draft the same way.
    """
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    ep_size, ep_rank = parallel.moe_ep_size, parallel.moe_ep_rank
    num_experts = int(
        getattr(config, "dspark_n_routed_experts", 0) or config.n_routed_experts
    )
    if ep_size <= 1:
        experts = sum(1 for n in names if _DRAFT_EXPERT.fullmatch(n))
        return names, experts
    if num_experts % ep_size:
        raise ValueError(
            f"DSpark draft: {num_experts} routed experts do not divide EP {ep_size}"
        )
    local = num_experts // ep_size
    lo, hi = ep_rank * local, (ep_rank + 1) * local
    kept, experts = set(), 0
    for name in names:
        match = _DRAFT_EXPERT.fullmatch(name)
        if match is not None:
            expert = int(match[2])
            if expert >= num_experts:
                raise ValueError(f"DSpark draft expert outside the config: {name}")
            if not lo <= expert < hi:
                continue
            experts += 1
        kept.add(name)
    return kept, experts
