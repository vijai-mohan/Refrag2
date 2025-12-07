# Refrag TODOs

## Runtime & Launching
| Task | Status | Links/Notes |
| --- | --- | --- |
| Docker image with PyTorch stack | ✅ | `Dockerfile` builds CUDA 12.4 runtime with project requirements. |
| WSL environment + venv | ✅ | Use the `refrag` venv (`refrag/bin/activate`), documented in `README.md`. |
| PyCharm using venv | ✅ | Point PyCharm to the `refrag` interpreter; no repo changes needed. |
| Hydra config for local runs | ✅ | `run.py`, `conf/config.yaml`, `conf/driver/local.yaml`. |
| Hydra config for torchrun | ✅ | `conf/driver/torchrun.yaml`, `src/refrag/framework/drivers/torchrun.py`. |
| Job-running architecture entrypoint | ✅ | `run.py`, diagrams/flow in `docs/ARCHITECTURE.md`. |

## Applications & Trainers
| Task | Status | Links/Notes |
| --- | --- | --- |
| Train: AlignmentTrainer | ✅ | `src/refrag/model/alignment_trainer.py` (loss adapters, custom Trainer). |
| Train: SFT Trainer | ✅ | `src/refrag/framework/apps/train.py` (`SFTTrainer` path). |
| Train: RL Trainer | ❌ | No RL trainer implemented yet. **Action:** define RL objective/datasets and implement trainer. |
| Train: Refrag Trainer | ⏳ | `RefragTrainerApp` in `src/refrag/framework/apps/train.py` uses placeholder data; needs real dataset/objectives. **Action:** wire real data and loss/metrics. |
| Evaluation: RAGAS | ✅ | `src/refrag/framework/apps/eval.py` (`_run_ragas`), usage in `docs/EVALUATION.md`. |
| Evaluation: CRAG | ❌ | Not implemented. **Action:** add CRAG eval pipeline/config. |
| Evaluation: lm-eval | 🚧 | `src/refrag/framework/apps/eval.py` (`_run_lm_eval`); mostly done, add option to disable context injection. **Action:** add no-context toggle. |
| Chat CLI | ✅ | `src/refrag/framework/apps/chatcli.py`, default app in `conf/config.yaml`. |
| Serving (webapp) | 🚧 | Backend ready in `webapp/server.py`; deploy wiring/UI lives in external repo `RefragUI`. **Action:** integrate deploy path here. |

## Launchers
| Task | Status | Links/Notes |
| --- | --- | --- |
| LocalLauncher | ✅ | `src/refrag/framework/envs/local.py`, `src/refrag/framework/drivers/inline.py`. |
| TorchrunLauncher | ✅ | `src/refrag/framework/drivers/torchrun.py`, `conf/driver/torchrun.yaml`. |
| BatchLauncher | ✅ | `src/refrag/framework/envs/awsbatch.py`, `conf/env/aws_batch.yaml`. |
| DockerLauncher | ✅ | `src/refrag/framework/envs/docker.py`. |

## AWS Infra
| Task | Status | Links/Notes |
| --- | --- | --- |
| Terraform AWS settings (Batch, ECR, S3, IAM) | ✅ | `infra/terraform/*.tf`, quickstart in `infra/README.md`. |
| Spot instances + job queue | ✅ | `infra/terraform/batch.tf`. |
| Docker images in ECR | ✅ | GH Actions workflow `.github/workflows/docker_ecr.yml`. |
| Auto-build from GitHub | ✅ | `.github/workflows/docker_ecr.yml` pushes on `main`. |
| ECS/Lambda to webapp | ❌ | Not provisioned yet. |
| AWS account permissions caveat | ✅ | See `docs/AWS_BATCH_SETUP.md` and `infra/README.md`. |
| GitHub ↔ AWS integration | ✅ | Secrets-based ECR push workflow configured. |

## Remaining Backlog
| Task | Status | Links/Notes |
| --- | --- | --- |
| Baseline eval metrics for Refrag models on RAG use cases | ⭕️ | Run `python run.py task=eval ...` per `docs/EVALUATION.md`; publish results under `outputs/eval/`. **Action:** execute runs and commit summaries. |
| Run evaluation in AWS | ⭕️ | Use `env=aws_batch` with eval app; see `infra/README.md` for submit steps. **Action:** submit Batch eval job and capture outputs. |
| Train Refrag model end-to-end | ⏳ | Extend `RefragTrainerApp` to real data/objectives and run via `app=train`. **Action:** specify dataset/config and run training. |
