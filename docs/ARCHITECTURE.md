High-level architecture and organization

Goal
- Train and evaluate an instruction-finetuned LLM (e.g., Gemma 1B) using a single Hydra driver that can run locally, in a container, or be submitted to AWS Batch.
- Provide an aligned encoder–decoder "Refrag" path for retrieval-augmented generation alongside standard SFT training.

Components
- Code repository (this GitHub repo):
  - Core Refrag model + tokenizer under `src/refrag/model/`
  - Training / eval / demo apps under `src/refrag/framework/apps/` (Hydra `app=` group)
  - Launch environments under `src/refrag/framework/envs/` (Hydra `env=` group)
  - Drivers under `src/refrag/framework/drivers/` (Hydra `driver=` group)
  - Hydra configs under `conf/`
  - Webapp under `webapp/`
- Docker image: custom image built from `Dockerfile`, used for local development, Jupyter notebooks, and AWS Batch jobs.
- AWS Batch: runs containerized training jobs. A single job definition points at the Hydra driver (`python run.py`) so the same image can run any task by providing Hydra overrides.
- Hydra: central configuration system. `run.py` is the only entrypoint; everything else is configured via `conf/`.

Repository layout (high level)

```
Refrag/
├── run.py                # Single Hydra entrypoint (adds src/ to sys.path)
├── src/
│   └── refrag/
│       ├── framework/    # Apps, envs, drivers (Hydra runtime)
│       ├── model/        # Model + tokenizer + alignment utilities
│       ├── training/     # Scripts/pipelines for SFT/alignment
│       └── eval/         # Evaluation helpers
├── conf/
│   ├── config.yaml       # Hydra defaults: env, driver, app
│   ├── app/              # app=train, app=hello_dist, app=alignment, ...
│   ├── env/              # env=local, env=docker, env=awsbatch
│   └── driver/           # driver=inline, driver=torchrun
├── docs/                 # Documentation hub
├── infra/                # Terraform + infra docs
├── tests/                # Mirrors src/refrag/*
└── webapp/               # Streaming and HTTP serving
```

Hydra driver pattern
- `run.py` is the **only** `@hydra.main` entrypoint.
- It composes config from `conf/config.yaml` and the selected groups (`app`, `env`, `driver`).
- It instantiates `cfg.env` (an object in `src/refrag/framework/envs/`), which in turn instantiates a driver from `src/refrag/framework/drivers/`, which finally instantiates and runs the app class from `src/refrag/framework/apps/`.
- Apps and drivers **never** decorate anything with `@hydra.main`; they just accept a `DictConfig`.

Execution chain (conceptual)

```
python run.py app=<app_name> env=<env_name> driver=<driver_name>

run.py (@hydra.main)
  ├─ instantiate cfg.env  (e.g. refrag.framework.envs.local.LocalLauncher,
  │                        refrag.framework.envs.docker.DockerLauncher,
  │                        refrag.framework.envs.awsbatch.AWSBatchLauncher)
  └─ env.run(cfg)
       ├─ decides which driver to use (inline vs torchrun vs batch)
       ├─ driver builds argv / process topology
       └─ driver instantiates cfg.app and calls app.run(cfg)
```

Two worlds: **apps** and **environments/drivers**

- **Apps (`src/refrag/framework/apps/`, Hydra group `app=`)**
  - Pure Python classes with a `run(cfg: DictConfig)` method.
  - Encapsulate *what* to do: training, evaluation, chat, hello world, alignment.
  - Examples:
    - `refrag.framework.apps.train.TrainerApp` – SFT training of an LLM with Hugging Face `Trainer`.
    - `refrag.framework.apps.eval.EvalApp` – evaluation runs.
    - `refrag.framework.apps.chatcli.ChatCLIApp` – interactive CLI around a HF model.
    - `refrag.framework.apps.hello_dist.HelloDistApp` – minimal distributed demo.
    - `refrag.framework.apps.train.RefragTrainerApp` – Refrag alignment training.

- **Env (`src/refrag/framework/envs/`, Hydra group `env=`)**
  - Decide *where* and *as what* the code runs: locally, in Docker, or on AWS Batch.
  - Examples:
    - `refrag.framework.envs.local.LocalLauncher` – run the chosen driver on the same machine.
    - `refrag.framework.envs.docker.DockerLauncher` – build/exec a container and run inside it.
    - `refrag.framework.envs.awsbatch.AWSBatchLauncher` – submit a job to AWS Batch, including overlaying the local checkout to S3.

- **Drivers (`src/refrag/framework/drivers/`, Hydra group `driver=`)**
  - Decide *how* the app is executed as a process tree: inline, via `torchrun`, etc.
  - Examples:
    - `refrag.framework.drivers.inline.InlineDriver` – call `app.run(cfg)` in the current Python process.
    - `refrag.framework.drivers.torchrun.TorchrunDriver` – spawn a `torchrun` process with the correct hydra overrides and environment for distributed training.

Hello World path through Hydra

We use `app=hello_dist` as the canonical, minimal example.

1. Configs involved

   - `conf/config.yaml` (top‑level):

     ```yaml
     defaults:
       - _self_
       - env: local
       - driver: inline
       - app: chatcli
     ```

     When you run `python run.py app=hello_dist env=torchrun`, Hydra composes:

     - `app` group → `conf/app/hello_dist.yaml`
     - `env` group → `conf/env/torchrun.yaml` (if used) or `env/local.yaml`
     - `driver` group → `conf/driver/torchrun.yaml` (if selected)

   - `conf/app/hello_dist.yaml`:

     ```yaml
    _target_: refrag.framework.apps.hello_dist.HelloDistApp
     message: "Hello from distributed PyTorch!"
     ```

   - `conf/env/local.yaml`:

     ```yaml
    _target_: refrag.framework.envs.local.LocalLauncher
     ```

   - `conf/driver/inline.yaml`:

     ```yaml
    _target_: refrag.framework.drivers.inline.InlineDriver
     ```

   - `conf/driver/torchrun.yaml`:

     ```yaml
    _target_: refrag.framework.drivers.torchrun.TorchrunDriver
     nproc_per_node: 2
     nnodes: 1
     node_rank: 0
     master_addr: 127.0.0.1
     master_port: 29500
     standalone: true
     ```

2. Local, single‑process Hello World

   ```bash
   # Inline driver, local env
   python run.py app=hello_dist env=local driver=inline
   ```

   Flow:
  - `run.py` → `refrag.framework.envs.local.LocalLauncher.run(cfg)` → `refrag.framework.drivers.inline.InlineDriver.run(cfg)` → instantiate `HelloDistApp` → `HelloDistApp.run(cfg)` in‑process.

3. Distributed Hello World with torchrun

   ```bash
   # Still local env, but use torchrun driver
   python run.py app=hello_dist env=local driver=torchrun

   # Or using env=torchrun pattern if you configure such an env wrapper
   python run.py app=hello_dist driver=torchrun env=local driver.nproc_per_node=4
   ```

   Flow:
  - `run.py` → `refrag.framework.envs.local.LocalLauncher.run(cfg)` → `refrag.framework.drivers.torchrun.TorchrunDriver.run(cfg)`.
   - `TorchrunDriver` builds the `torchrun` command (nproc, nnodes, master settings) and re‑invokes `python run.py` under `torchrun` with overrides transformed so that the inner run uses `env=local` and `driver=inline`.
   - Inside each `torchrun` worker process, `run.py` is called again, Hydra reconstructs the config, and since `RANK` is set, `TorchrunDriver.run` short‑circuits and just instantiates `HelloDistApp` and calls `run(cfg)`.

  The `HelloDistApp` itself (in `src/refrag/framework/apps/hello_dist.py`) handles `torch.distributed` initialization and a simple `all_reduce` example; it is a good template for writing your own distributed apps.

Refrag path: model + alignment

While the Hello World path shows the Hydra plumbing, the "Refrag" pieces live in `refrag/` and are used by alignment‑ and retrieval‑focused apps.

- `src/refrag/model/modeling_refrag.py` – defines `RefragModel`, which wires an encoder model, a projector, and a decoder model into a single forward that can accept encoder/decoder segments.
- `src/refrag/model/tokenization_refrag.py` – defines `RefragTokenizer`, which parses `<doc ...>...</doc>` blocks into encoder segments and treats everything else as decoder segments; it also handles packing encoder tokens into `compression`‑wide rows.
- `src/refrag/model/alignment_trainer.py` – training logic that uses a DP token-alignment algorithm (implemented in `src/refrag/model/tokenizer/tokenizer_utils.py`) to compute MSE / KLD losses between decoder embeddings and projected encoder embeddings at aligned positions.

In the "new" Refrag usage pattern, you typically:
- Train or finetune a pure decoder model (`app=train`) using standard SFT.
- Use alignment training (`app=alignment`) to learn the projector and optionally adjust encoder/decoder to live in a shared space.
- Use `RefragTokenizer` + `RefragModel` in downstream RAG pipelines or the `webapp/` serving path.

See the root `README.md` for a full narrative of the two core problems (alignment and segmentation) and how the code solves them.
