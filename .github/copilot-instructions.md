# Refrag Copilot Instructions

## Critical: Virtual Environment
**ALWAYS activate the `refrag` venv before running ANY commands.** This is mandatory across all tooling and automation:
```bash
source refrag/bin/activate  # Linux/Mac/WSL
which python  # Must end with /refrag/bin/python
```
Never use system Python. Guard scripts should verify this before execution.

## Architecture Overview

### Hydra-Driven Execution Model
Refrag uses a **single entrypoint** (`run.py`) that composes three orthogonal config groups:
- **`app=`** (what to run): `train`, `eval`, `chatcli`, `hello_dist`, `alignment`
- **`env=`** (where to run): `local`, `docker`, `awsbatch`
- **`driver=`** (how to run): `inline`, `torchrun`

**Flow**: `run.py` → `env.run(cfg)` → `driver.run(cfg)` → `app.run(cfg)`

Example: `python run.py app=train env=local driver=torchrun driver.nproc_per_node=4`

### Never Decorate with @hydra.main
Only `run.py` uses `@hydra.main`. Apps, drivers, and envs receive a `DictConfig` object and implement `run(cfg)`.

### Key Components
- **Apps** (`src/refrag/framework/apps/*.py`): Pure Python classes implementing `run(cfg)` - define the task logic
  - `TrainerApp`: Standard SFT training via Hugging Face Trainer
  - `RefragTrainerApp`: Alignment training with encoder-decoder token mapping
  - `HelloDistApp`: Minimal distributed example (canonical reference for new distributed apps)
  
- **Drivers** (`src/refrag/framework/drivers/*.py`): Control process topology
  - `InlineDriver`: Run app in current process
  - `TorchrunDriver`: Launch distributed training via torchrun (auto-detects `RANK` env var)
  
- **Envs** (`src/refrag/framework/envs/*.py`): Control execution environment
  - `LocalLauncher`: Execute on current machine
  - `DockerLauncher`: Build and exec in container
  - `AWSBatchLauncher`: Submit to AWS Batch with S3 overlay

## Refrag Model: Two Core Problems

### 1. Token-Level Alignment (Encoder ↔ Decoder)
**Problem**: Different tokenizers split text differently. Token position `i` in encoder ≠ position `i` in decoder.

**Solution**: Dynamic programming computes alignment by matching decoded prefix strings:
```python
# In src/refrag/model/tokenizer/tokenizer_utils.py
dp = compute_dp_matrix(decoder_ids, encoder_ids, dec_decode_fn, enc_decode_fn)
pairs = alignment_pairs_from_dp(dp)  # Returns (dec_idx, enc_idx) where prefixes match
```

**Usage in training** (`refrag/alignment_trainer.py`):
- Compute embeddings: `DE = decoder.embed_tokens(dec_ids)`, `RE = projector(encoder(enc_ids))`
- Only compute losses (MSE, KLD, argmax) at aligned positions from DP output
- Never use naive position-wise loss when M ≠ N

### 2. Document Segmentation (RefragTokenizer)
**Problem**: Need to mix encoder-processed documents with decoder-processed prompts.

**Solution**: `RefragTokenizer` parses `<doc id="...">content</doc>` tags:
- **Outside tags**: Decoder tokenizer → regular token embeddings
- **Inside tags**: Encoder tokenizer → pack into rows of width `compression` → CLS embedding per row → project to decoder dim

Example:
```python
text = "Question: What is AI? <doc id='1'>Artificial intelligence is...</doc> Answer:"
segments = tokenizer.parse_and_tokenize(text)
# Returns: [decoder_seg, encoder_seg, decoder_seg]
embeddings = model.forward(segments)  # Mixed embedding sequence
```

## Common Workflows

### Local Training
```bash
source refrag/bin/activate
python run.py app=train env=local driver=inline
```

### Distributed Training (Single Node)
```bash
python run.py app=train env=local driver=torchrun driver.nproc_per_node=4
```

### AWS Batch Submission
```bash
python run.py app=train env=awsbatch env.gpus=8 env.num_hosts=2
```

### Running Tests
```bash
pytest tests/refrag/tokenizer/  # Test alignment and tokenization logic
pytest tests/  # Full test suite
```

## Configuration Patterns

### Overriding Config Values
```bash
# Override nested config
python run.py app=train app.training_args.num_train_epochs=5

# Add new config key
python run.py app=train +app.new_param=value
```

### Config File Structure
- `conf/config.yaml`: Top-level defaults and global settings (seed, device, HF_TOKEN)
- `conf/app/*.yaml`: App-specific configs with `_target_` pointing to Python class
- `conf/driver/*.yaml`: Driver configs (torchrun params, etc.)
- `conf/env/*.yaml`: Environment configs (Docker image, Batch queue, etc.)

## Code Patterns

### Creating New Apps
1. Create class in `src/refrag/framework/apps/` with `__init__(**cfg)` and `run(cfg)` methods
2. Add config file in `conf/app/your_app.yaml` with `_target_: refrag.framework.apps.your_module.YourApp`
3. Run with `python run.py app=your_app`

### Distributed Apps (Use HelloDistApp as Template)
```python
class YourDistApp:
    def run(self, cfg: DictConfig):
        if torch.distributed.is_available():
            torch.distributed.init_process_group(backend="gloo")
            rank = torch.distributed.get_rank()
            # Your distributed logic
        # Your single-process fallback
```

### Torchrun Driver Pattern
`TorchrunDriver` detects if already running under torchrun via `RANK` env var. If not, it spawns torchrun subprocess with transformed overrides (forces `env=local` to avoid nested launches).

## Debugging Tips

### Check Hydra Config Resolution
```python
from omegaconf import OmegaConf
print(OmegaConf.to_yaml(cfg, resolve=True))  # Already done in run.py
```

### Per-Rank Logging
Each process writes to `outputs/<app>/<run_name>/logs/run-{RANK}.log`. Check all rank logs for distributed issues.

### Verify Environment
```bash
which python  # Must be refrag/bin/python
pip list | grep torch  # Check dependencies
echo $RANK  # Present when under torchrun
```

## Key Files Reference

- **Entry**: `run.py` - Only `@hydra.main` entrypoint
- **Model**: `src/refrag/model/modeling_refrag.py` - RefragModel composing encoder + projector + decoder
- **Tokenizer**: `src/refrag/model/tokenization_refrag.py` - Document tag parsing and packing
- **Alignment**: `src/refrag/model/alignment_trainer.py` - DP-based token alignment training
- **Utils**: `src/refrag/model/tokenizer/tokenizer_utils.py` - DP matrix computation for prefix alignment
- **Drivers**: `src/refrag/framework/drivers/torchrun.py` - Most complex driver (handles nested invocation)
- **Apps**: `src/refrag/framework/apps/hello_dist.py` - Canonical distributed app template

## Dependencies
Core: PyTorch, Transformers, Hydra, TRL, datasets, accelerate
AWS: boto3, awscli, s3fs
Dev: pytest, black, isort, tensorboard

Install: `pip install -r requirements.txt` (inside venv!)

## Common Pitfalls

1. **Forgetting venv activation**: Leads to missing packages or wrong Python version
2. **Multiple @hydra.main**: Only `run.py` should have this decorator
3. **Hardcoded env in nested torchrun**: `TorchrunDriver` must transform `env=` to `local` to avoid infinite recursion
4. **Position-wise losses without alignment**: Always use DP alignment when encoder/decoder tokenizations differ
5. **Special token mismatches**: Strip or normalize BOS/EOS tokens before alignment to avoid spurious mismatches
6. **HF_TOKEN not set**: `run.py` requires `cfg.HF_TOKEN` for model downloads (set in `conf/config.yaml` or env var)
