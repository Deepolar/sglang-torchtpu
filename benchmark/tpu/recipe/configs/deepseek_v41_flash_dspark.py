"""DeepSeek-V4.1-Flash + its bundled DSpark draft on one tpu7x-8 host.

    python3 -m benchmark.tpu.launch_server \\
        --config recipe/configs/deepseek_v41_flash_dspark.py

The deepseek_v41_flash.py deployment with DSpark on the experimental TPU spec
worker. The draft's MoE shards over the target's EP8 group (~1.6 GB/device).
Measured on one tpu7x-8: GSM8K 1319 at c128 0.970 (target-only 0.973), accept
length 3.8 on GSM8K but ~1.05 on random-ids prompts, where it runs at a third
of target-only output throughput at c64 and c128. Use it for predictable text,
not for random or high-concurrency throughput workloads.
"""

from copy import deepcopy

from benchmark.tpu.recipe.configs.deepseek_v41_flash import CONFIG as BASE_CONFIG

CONFIG = deepcopy(BASE_CONFIG)
CONFIG["server_args"].update(
    {
        # The draft weights are the mtp.* tensors of the target checkpoint.
        "speculative_algorithm": "DSPARK",
        # 5 draft tokens per step: the verify runs 6 rows per request.
        "speculative_dspark_block_size": 5,
        # At 0.85 the larger KV pool leaves the 2048-token extend program
        # ~280 MB short of HBM at load; 0.83 loads every graph.
        "mem_fraction_static": 0.83,
    }
)
CONFIG["runtime_env"].update({"SGLANG_TPU_ENABLE_EXPERIMENTAL_SPEC_WORKER": "1"})
