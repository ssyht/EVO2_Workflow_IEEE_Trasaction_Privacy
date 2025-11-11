# Evo-2 Runtime Set Up Logs

## Model fetch & cache

* ```httpx – HTTP Request: GET https://huggingface.co/...```

First run only. Downloads the model files (```evo2_7b.pt```, ``config.json``, etc.) to the container’s Hugging Face cache (we mounted ``/opt/hf-cache``). Subsequent runs hit the local cache.

## Architecture printout

* ``Initializing StripedHyena with config: {...}``

**The model’s hyper-parameters dump. Common fields:**

  * ``vocab_size``: size of the DNA token vocabulary.

  * ``hidden_size``, ``num_filters``, ``num_layers``: width/depth (capacity).

  * ``attn_layer_idxs``, ``hcm_layer_idxs``: which blocks use attention vs. hyena mixers.

  * ``use_flash_attn: True``: FlashAttention kernels enabled.

  * ``use_fp8_input_projections: True``: FP8 path active (your L40S supports this).

  * ``max_seqlen``: maximum tokens the runtime is configured to handle in one pass.

## Device placement

* ``Distributing across 1 GPUs, approximately 32 layers per GPU``

You have a single GPU (L40S). All 32 blocks go to cuda:0.

* ``Assigned layer_idx=X to device='cuda:0'`` and

``Parameter count for block N: ...``

A per-block placement + parameter count sanity check. Confirms everything is resident on the GPU and helps spot partial loads.

## Weight layout tweaks & workspaces

* ``Initialized model``
All tensors constructed and moved to the GPU.

* ``Adjusting Wqkv for column split (permuting rows)``
One-time layout transformation to make Q/K/V projections contiguous for the fused kernels.

* ``Fixup applied: Allocating cuBLAS workspace for device=0``
Reserves CUDA work buffers used by GEMMs. Normal.

## Inference parameters and tests

* ``Initializing inference params with max_seqlen=...``
Builds the runtime state (KV cache shapes, sampler buffers, etc.) sized to your request.

* ``Prompt: "GAA..." Output: "GCA..." Score: ...``
The test harness runs quick generation and scoring passes.

  * **Output** is the continuation produced by the sampler.

  * **Score** is a model likelihood-style value (more negative = less likely under the model).

* ``Test Results: % Matching Nucleotides: 89.5 | Test Passed: Score matches expected ...``
A deterministic check that your stack (CUDA/FlashAttention/Transformer-Engine) is wired correctly. Passing means you’re good.

## “Extra keys” and other benign messages

* ``Extra keys in state_dict: {...}``
**Safe to ignore;** checkpoints often contain auxiliary tensors the runtime doesn’t need.

* ``UserWarning: Casting complex values to real discards the imaginary part``
Harmless internal cast during kernel setup. Doesn’t affect outputs.

## (From earlier) shared-memory notice

* ``NOTE: The SHMEM allocation limit is set to the default of 64MB ...``
That’s the container warning. We solved it by running with ``--ipc=host`` (or ``--shm-size=2g``) and raising ulimits.
