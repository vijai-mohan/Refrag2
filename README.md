# Refrag2: Retrieval-Augmented Generation with Aligned Embeddings

The primary objective of this project is to advance the Refrag work from a research prototype into production-ready technology. To achieve this, several key improvements are required:

## 1. Upgrade Refrag recipes to use post-trained models
Transition recipe foundations from **pretrained** models to **post-trained** models.  
This change will:
- broaden compatibility across open-source model families
- potentially simplify the overall recipe pipeline

## 2. Revise the context window format to include document identifiers
Update the context window so that **document IDs** are embedded alongside raw text.  
This enables:
- interleaving embeddings and text
- maintaining citation and provenance fidelity throughout retrieval

## 3. Enable query-dependent encoding
Modify the encoder to **condition on the input query** to drive chunk selection.  
This upgrade:
- makes chunk scoring query-specific
- removes the need for reinforcement learning
- allows for simple chunk-level gating to determine which segments are passed forward
## Documentation

Use this table of contents to find what you need quickly:
- Project overview and model/tokenizer: (this README)
- Local development (venv = `refrag` is mandatory): see “Local Development” below
- Architecture: docs/ARCHITECTURE.md
- Torchrun (local/distributed): docs/TORCHRUN_GUIDE.md
- AWS Batch setup: docs/AWS_BATCH_SETUP.md
- CPU vs GPU queues on Batch: docs/AWS_BATCH_CPU_GPU_QUEUES.md
- Evaluation: docs/EVALUATION.md
- Docs index: docs/README.md
- Webapp: webapp/README.md

---

## Project Overview

Refrag is a machine learning project that trains and aligns encoder-decoder models for retrieval-augmented generation. The project uses Hydra for configuration management and supports multiple execution environments (local, Docker, AWS Batch).

### Project Organization

```
Refrag/
├── conf/                          # Hydra configuration files (app/env/driver groups)
├── docs/                          # Documentation
├── infra/                         # Terraform + AWS Batch tooling
├── run.py                         # Main Hydra entrypoint (adds src/ to sys.path)
├── requirements.txt               # Development dependencies
├── src/
│   └── refrag/
│       ├── framework/             # Hydra apps, envs, drivers (hello_dist lives here)
│       ├── model/                 # Publishable Refrag model + tokenizer package
│       ├── training/              # Training scripts/pipelines (main_alignment, etc.)
│       └── eval/                  # Evaluation utilities
├── tests/
│   └── refrag/                    # Pytest suites mirroring src/refrag
└── webapp/                        # Web application for serving/demo
```

### Key Technologies
- **PyTorch**: Core deep learning framework
- **Transformers (Hugging Face)**: Pre-trained encoder and decoder models
- **Hydra**: Configuration management and multi-run orchestration
- **AWS Batch**: Distributed training infrastructure (optional)
- **Docker**: Containerized execution environment

## Secrets & Authentication
- Export `HF_TOKEN` before running `python run.py` or sample scripts; configs default to `hf_fake_token_use_env` to avoid committed tokens (example: `export HF_TOKEN=hf_xxx`).
- Never hardcode AWS/HF tokens, passwords, or usernames in code or configs—use env vars, AWS profiles, or secret managers instead of committing credentials.
- Docker/Batch runs should pass through `HF_TOKEN` (see `conf/env/docker.yaml` `env_passthrough`) and rely on your local AWS credentials/profile rather than embedding keys.

---

## The Two Core Problems

Refrag addresses two fundamental challenges in retrieval-augmented generation:

### Problem 1: Alignment of Decoder and Encoder+Projector Embedding Spaces

**Challenge**: Different tokenizers (encoder tokenizer `ET` vs decoder tokenizer `DT`) split text differently due to different subword vocabularies and special tokens. This means token positions in `ET(P)` and `DT(P)` are not 1:1 aligned.

**Goal**: Align encoder embeddings and decoder embeddings so that a projector `A` can map encoder embeddings into the decoder embedding space, enabling:
1. **Embedding alignment** (MSE loss): Make decoder embeddings `DE` and projected encoder embeddings `RE` match where tokens correspond semantically
2. **Distribution alignment** (KLD/cross-entropy): Make decoder output distributions from `DE` and `RE` similar
3. **Token-level correspondence**: Compute losses only at positions where decoded text prefixes match between decoder and encoder tokenizations

**Solution**: Use dynamic programming (DP) to compute token-level alignment by checking decoded text prefix equality. This produces anchor pairs `(i, j)` where `ET.decode(enc_ids[:i]) == DT.decode(dec_ids[:j])`, allowing semantically equivalent spans to be compared despite different tokenization granularity.

### Problem 2: Context Segmentation with RefragTokenizer

**Challenge**: In retrieval-augmented generation, we need to efficiently encode long documents (context) while maintaining compatibility with a decoder model for generation.

**Goal**: Segment input text into encoder-processed regions (documents) and decoder-processed regions (queries/prompts), where:
- Documents are compressed using an encoder + projector
- Queries and generation targets use the decoder tokenizer
- The system maintains proper alignment between the two modalities

**Solution**: The `RefragTokenizer` uses document tags `<doc id="...">content</doc>` to segment text:
- **Decoder segments**: Text outside `<doc>` tags, tokenized with decoder tokenizer
- **Encoder segments**: Text inside `<doc>` tags, tokenized with encoder tokenizer, then packed into rows of `compression` width

Each encoder row produces a single embedding (CLS token) that is projected into decoder embedding space, effectively compressing `compression` encoder tokens into one decoder-sized embedding.

---

## RefragModel Architecture

The `RefragModel` is a hybrid architecture that combines:
1. **Encoder Model** (`E`): Processes document segments (e.g., RoBERTa, BERT)
2. **Projector** (`A`): Linear layer mapping encoder hidden dim → decoder hidden dim
3. **Decoder Model** (`D`): Causal language model for generation (e.g., GPT-2, Gemma)

### Key Components

#### RefragTokenizer (`src/refrag/model/tokenization_refrag.py`)
Parses and tokenizes mixed encoder/decoder input:

```python
tokenizer = RefragTokenizer(
    encoder_tokenizer=enc_tok,
    decoder_tokenizer=dec_tok,
    compression=4,  # Pack 4 encoder tokens per row
    encoder_pad_token_id=0
)

# Input with document tags
text = "Question: What is AI? <doc id='1'>Artificial intelligence is...</doc> Answer:"

# Parse and tokenize
segments = tokenizer.parse_and_tokenize(text)
# Returns: [
#   {'type': 'decoder', 'tokens': [...], 'raw_text': 'Question: What is AI? '},
#   {'type': 'encoder', 'tokens': [...], 'packed_tokens': [[...], [...]], 
#    'attention_mask': [[...], [...]], 'raw_text': 'Artificial intelligence is...'},
#   {'type': 'decoder', 'tokens': [...], 'raw_text': ' Answer:'}
# ]
```

**Packing mechanism**: Encoder tokens are packed into rows of width `compression`. Each row is independently encoded, producing one CLS embedding that represents `compression` tokens worth of context.

#### RefragModel (`src/refrag/model/modeling_refrag.py`)
Processes tokenized segments and produces embeddings:

```python
model = RefragModel(config)
# config specifies encoder_name_or_path, decoder_name_or_path, compression, training flags

# Forward pass
embeddings = model.forward(segments)
# Returns list of embedding dicts in original order:
# [
#   {'type': 'decoder', 'embeddings': tensor(L1, D), 'length': L1},
#   {'type': 'encoder', 'embeddings': tensor(num_rows, D), 'length': num_rows},
#   {'type': 'decoder', 'embeddings': tensor(L2, D), 'length': L2}
# ]
```

**Model configuration** (`RefragConfig`):
- `encoder_name_or_path`: HuggingFace model name for encoder (default: "roberta-base")
- `decoder_name_or_path`: HuggingFace model name for decoder (default: "sshleifer/tiny-gpt2")
- `compression`: Number of encoder tokens per packed row (default: 4)
- `training_model`: String of letters controlling which components to train:
  - `"p"`: train projector
  - `"e"`: train encoder
  - `"d"`: train decoder
  - `"ped"`: train all (default)

### Forward Pass Details

1. **Decoder segments**: 
   - Token IDs → Decoder embedding lookup → Embeddings `(L, D)`

2. **Encoder segments**:
   - Packed tokens `(num_rows, compression)` → Encoder forward pass (per row)
   - Extract CLS embedding (or pooler output) → `(num_rows, enc_dim)`
   - Projector: `(num_rows, enc_dim)` → `(num_rows, dec_dim)`

3. **Output**: Concatenated embedding sequence maintaining original token order

---

## Alignment Training (Problem 1 Solution)

### Key Notation
- `P`: input prompt string (text)
- `DT`: decoder tokenizer
- `ET`: encoder tokenizer
- `D`: decoder model (with embedding matrix `D.embed_tokens`)
- `E`: encoder model
- `A`: projector mapping encoder embedding space → decoder embedding space
- `DT(P)`: list of decoder token ids for prompt `P`, length `N`
- `ET(P)`: list of encoder token ids for prompt `P`, length `M`
- `DE = D.embed_tokens(DT(P))` → shape `(B, N, K)` (decoder token embeddings)
- `RE = A(E(ET(P)))` → shape `(B, M, K)` (projected encoder embeddings)

### Alignment Strategy

1. Compute token id sequences: `dec_ids = DT.encode(P)` and `enc_ids = ET.encode(P)`
2. Compute token-level alignment using **dynamic programming (DP)** that finds matches by checking decoded-text prefixes
3. Use the DP-derived mapping to aggregate/align embeddings and compute losses only at matched alignment points

### Dynamic Programming Alignment Algorithm

We construct a DP table where:
- **Rows** represent encoder token indices `i = 0..M`
- **Columns** represent decoder token indices `j = 0..N`
- **Cell `(i, j)`** stores `True` if prefix equality holds:
  ```
  ET.decode(enc_ids[:i]) == DT.decode(dec_ids[:j])
  ```

**Example**: Suppose we have:
- Encoder tokenization: `["Re", "frag", "ation"]` → tokens `e1, e2, e3` (M=3)
- Decoder tokenization: `["Refr", "ag", "ation"]` → tokens `d1, d2, d3` (N=3)


**DP Table (encoder rows × decoder columns)**:

| enc \ dec | j=0 ("") | j=1 ("Refr") | j=2 ("Refrag") | j=3 ("Refragation") |
|-----------|----------|--------------|----------------|---------------------|
| i=0 ("")  | **True** | False        | False          | False               |
| i=1 ("Re")| False    | **False**     | False          | False               |
| i=2 ("Refrag")| False | False       | **True**       | False               |
| i=3 ("Refragation")| False | False  | False          | **True**            |

The `True` cells represent alignment anchors where prefix text matches exactly.

### DP Algorithm Steps

1. Create `(M+1) × (N+1)` boolean table `match[i][j]`
2. Initialize `match[0][0] = True` (empty prefixes match)
3. For each cell `(i, j)`, compute:
   ```python
   match[i][j] = (ET.decode(enc_ids[:i]) == DT.decode(dec_ids[:j]))
   ```
4. Extract monotonic anchor pairs `(i, j)` where `match[i][j] == True`
5. Define aligned chunks between successive anchors `(i0, j0) → (i1, j1)`:
   - Encoder span: `enc_ids[i0:i1]`
   - Decoder span: `dec_ids[j0:j1]`

### Complexity and Optimizations

**Complexity**: O(M × N) decode checks per example

**Optimizations**:
- Cache prefix-decoded strings for both tokenizers
- Incrementally build decoded prefixes (decode only new token and append)
- Precompute alignments offline for the dataset if training multiple epochs
- Group examples by similar `(M, N)` to vectorize operations

**Cost-aware alternative**: Use edit-distance style DP with costs if strict prefix equality is too rigid (handles special tokens, spacing differences better).

### Loss Computation

Given an aligned chunk `(i0, j0) → (i1, j1)`:

1. **Embedding loss** (MSE):
   ```python
   span_RE = mean(RE[:, i0:i1, :], dim=1)  # shape (B, K)
   span_DE = mean(DE[:, j0:j1, :], dim=1)  # shape (B, K)
   L_embed = MSE(span_RE, span_DE)
   ```

2. **Distribution loss** (KLD/cross-entropy):
   - Compute decoder logits from both `DE` and `RE` embeddings
   - Compare softmax distributions using KLD or cross-entropy

3. **Argmax loss** (optional):
   - Encourage argmax token from `DE` and `RE` logits to match
   - Compute cross-entropy on argmax distributions

**Where to compute losses**: Only at aligned chunks where decoded prefixes match exactly.

### Practical Training Flow

1. Load dataset (Wikipedia-like texts) per configuration in `conf/`
2. For each prompt `P` in a batch:
   - Compute `enc_ids = ET.encode(P)` and `dec_ids = DT.encode(P)`
   - Precompute or retrieve alignment anchors using DP
3. Compute embeddings:
   - `RE = A(E(enc_ids))` (projected encoder embeddings)
   - `DE = D.embed_tokens(dec_ids)` (decoder embeddings)
4. For each aligned chunk:
   - Pool embeddings and compute configured losses (MSE, KLD, argmax)
5. Sum or weight losses and backpropagate
6. Use gradient clipping and scheduler as configured

---

## Tokenizer Locations and Implementation

### Main Tokenizer Utils (`src/refrag/model/tokenizer/tokenizer_utils.py`)
Primary utilities used across the project for:
- Encoding/decoding with different tokenizers
- Prefix equality checks
- DP alignment computation and caching

### Refrag Tokenizer (`src/refrag/model/tokenization_refrag.py`)
Model-specific tokenization with document segmentation:
- Parses `<doc id="...">content</doc>` tags
- Segments text into encoder/decoder regions
- Packs encoder tokens into rows

### Alignment Tokenizer
Specialized tokenizer used to initialize the Refrag model and perform token-level alignment precomputation. Configuration in `conf/` and `src/refrag/model/tokenization_refrag.py`.

---

## Running the Alignment Trainer

This repository uses Hydra for configuration. The `run.py` entrypoint composes configuration from `conf/`.

### Basic Usage

```bash
# Activate virtual environment
source refrag/bin/activate  # or: . refrag/bin/activate

# Run with default config
python run.py app=alignment

# Override training parameters
python run.py app=alignment \
  app.dataset=wikipedia \
  app.training.epochs=3 \
  app.training.steps_per_epoch=100

# Specify model and dataset path
python run.py app=alignment \
  +app.model_name="roberta-base" \
  +app.dataset.path="data/wikipedia_sample.jsonl"

# Use main_alignment.py directly
python main_alignment.py \
  trainer.epochs=1 \
  trainer.steps_per_epoch=5 \
  trainer.batch_size=2
```

### Configuration Structure

- **App configs**: `conf/app/*.yaml` (alignment, train, eval, chatcli)
- **Driver configs**: `conf/driver/*.yaml` (inline, torchrun)
- **Trainer configs**: `conf/trainer/*.yaml`
- **Model configs**: Models specified in app configs

---

## Testing

Unit tests validate tokenization and alignment logic:

```bash
# Run tokenizer tests
pytest tests/refrag/tokenizer/

# Test specific functionality
pytest tests/refrag/tokenizer/test_alignment.py -v
```

**Test coverage includes**:
- Roundtrip encode→decode for both ET and DT
- Prefix equality checks used by DP algorithm
- Small DP table examples reproducing documentation examples
- RefragTokenizer document parsing and packing

---

## Common Pitfalls

1. **Special tokens**: Different tokenizers insert BOS/EOS tokens. Normalize sequences by removing or consistently handling special tokens before alignment.

2. **Position-wise losses**: Never perform naive position-wise losses when `M ≠ N`. Always use DP-derived mapping.

3. **Memory**: Precomputing alignments for very large datasets can be memory-intensive. Consider on-the-fly computation with caching.

4. **Tokenizer differences**: Test alignment with your specific tokenizer pair, as behavior varies across models.

---

## Project Execution Modes

### Local Execution
```bash
python run.py app=train env=local
```

### Docker Execution
```bash
python run.py app=train env=docker
```

### Distributed Training (torchrun)
```bash
python run.py app=train env=torchrun env.nproc_per_node=4
```

### AWS Batch
Submit jobs using the batch submitter:
```bash
python infra/scripts/batch_submitter.py \
  --image <ECR_IMAGE> \
  --queue <BATCH_QUEUE> \
  --overrides "app=train model=gemma_1b train.steps=1000"
```

---

## Additional Documentation

- **Architecture**: `docs/ARCHITECTURE.md` - High-level system design
- **Alignment Algorithm**: `docs/ALIGNMENT_ALGORITHM.md` - Detailed alignment theory
- **Evaluation**: `docs/EVALUATION.md` - Model evaluation procedures
- **Torchrun Guide**: `docs/TORCHRUN_GUIDE.md` - Distributed training setup
- **Chat CLI**: `src/refrag/framework/apps/README_CHATCLI.md` - Interactive chat interface
- **Webapp**: `webapp/README.md` - Web serving deployment

---

## Development Setup

1. **Create virtual environment**:
   ```bash
   python -m venv refrag
   source refrag/bin/activate  # Linux/Mac
   # or: refrag\Scripts\activate  # Windows
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run tests**:
   ```bash
   pytest tests/
   ```

4. **Start development server** (webapp):
   ```bash
   cd webapp
   python server.py
   ```

---

## Local Development

All local work (running scripts, tests, linters, benchmarks, Hydra drivers) MUST occur inside a Python virtual environment named `refrag` located at the project root. Tooling (including GitHub Copilot / automated agents) should always verify that `which python` resolves to the `./refrag` environment before executing commands.

### Why a fixed venv name?
Using a deterministic environment name (`refrag/`) lets automation and docs assume a stable interpreter path, avoids accidental system Python usage, and keeps dependency isolation simple across shells, WSL, Docker bind mounts, and CI.

### Create / Recreate the environment
If the `refrag/` directory does not exist or you need a clean environment:

```bash
python3 -m venv refrag
# OR explicitly with pyenv / custom python:
/path/to/python -m venv refrag
```

### Activate the environment
Linux / macOS / WSL:
```bash
source refrag/bin/activate
```

Windows (PowerShell):
```powershell
refrag\Scripts\Activate.ps1
```

Windows (CMD):
```cmd
refrag\Scripts\activate.bat
```

Confirm:
```bash
which python        # should end with /refrag/bin/python
python -V           # expected version
pip list            # project dependencies
```

### Install dependencies
```bash
pip install -r requirements.txt
```
If you add new packages: update `requirements.txt`, then re-run the install.

### Running code (always inside venv)
```bash
python run.py app=hello_dist
python run.py app=train env=torchrun env.nproc_per_node=2
pytest -q
```

### Enforcing venv usage in scripts / AI tooling
Automation and assistants should perform this guard before executing project code:
```bash
if [ "$(basename $(dirname $(which python)))" != "refrag" ]; then
  echo "ERROR: virtual environment 'refrag' not active" >&2
  exit 1
fi
```
For multi-step sessions, keep the environment active; do NOT invoke system python.

### Upgrading dependencies safely
```bash
pip install --upgrade somepackage
pip freeze > /tmp/new.freeze
# Manually reconcile changes; then edit requirements.txt
pip install -r requirements.txt --upgrade
```

### Common pitfalls
- Forgetting to activate venv: leads to missing packages or wrong Python version.
- Creating multiple venvs (e.g., `venv/`, `.venv/`): remove them to avoid confusion.
- Mixing global pip installs: always inspect `which pip` (should be under `refrag/bin`).

### Quick diagnostic script
```bash
python - <<'PY'
import sys, pathlib
p = pathlib.Path(sys.executable)
assert 'refrag' in p.parts, f"Not in refrag venv: {p}"
print("✅ refrag venv active:", p)
PY
```



