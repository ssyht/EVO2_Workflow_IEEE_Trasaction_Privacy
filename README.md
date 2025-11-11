# EVO2_Workflow_IEEE_Trasaction_Privacy - Details from the main repo

## Usage

### Checkpoints

We provide the following model checkpoints, hosted on [HuggingFace](https://huggingface.co/arcinstitute):
| Checkpoint Name                        | Description |
|----------------------------------------|-------------|
| `evo2_7b`  | 7B parameter model with 1M context |
| `evo2_40b`  | 40B parameter model with 1M context (requires multiple GPUs) |
| `evo2_7b_base`  | 7B parameter model with 8K context |
| `evo2_40b_base`  | 40B parameter model with 8K context |
| `evo2_1b_base`  | Smaller 1B parameter model with 8K context |
| `evo2_7b_262k`  | 7B parameter model with 262K context |
| `evo2_7b_microviridae`  | 7B parameter base model fine-tuned on Microviridae genomes |

**Note:** The 40B model requires multiple GPUs. Vortex automatically handles device placement, splitting the model across available CUDA devices.

