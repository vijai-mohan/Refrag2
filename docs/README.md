# Refrag Documentation Hub

This is the central documentation index for the project. Use the table of contents below to navigate. All local development MUST run inside a Python virtual environment named `refrag` at the repo root.

Table of contents
- Getting Started
  - Local Development (venv = `refrag`)
  - Project Overview
- Architecture
- Training & Drivers
  - Torchrun (local/distributed)
  - AWS Batch (setup, CPU/GPU queues, launcher usage)
  - Terraform (provisioning)
- Evaluation
- Webapp
- Reference & Tips

---

## Getting Started

### Local Development (required venv: `refrag`)
- Create venv: `python3 -m venv refrag`
- Activate (Linux/WSL/macOS): `source refrag/bin/activate`
- Install deps: `pip install -r requirements.txt`
- Verify: `which python` should end with `/refrag/bin/python`

See the root README “Local Development” section for guard snippets and troubleshooting.

### Project Overview
High-level goals and concepts are in the root README (model, tokenizer, alignment problems, training flow). Start there if you’re new to the codebase.

- Root README: ../README.md

---

## Architecture
- docs/ARCHITECTURE.md – components, Hydra entrypoints, drivers.

---

## Training & Drivers

### Torchrun (local/distributed)
- Guide: ./TORCHRUN_GUIDE.md
- Quick runs:
  - CPU: `python run.py app=hello_dist driver=torchrun`
  - N GPUs (single node): `python run.py app=train driver=torchrun driver.nproc_per_node=N`

### AWS Batch
- Setup guide: ./AWS_BATCH_SETUP.md
- CPU vs GPU queues: ./AWS_BATCH_CPU_GPU_QUEUES.md
- Hydra env config: `conf/env/awsbatch.yaml`
- Typical runs:
  - CPU: `python run.py app=hello_dist env=awsbatch`
  - Single GPU: `python run.py app=hello_dist env=awsbatch env.gpus=1`
  - Multi-host: `python run.py app=train env=awsbatch env.num_hosts=4 env.gpus_per_host=2`

### Terraform (provisioning)
Use Terraform to create ECR, S3, IAM, and Batch resources:

```bash
cd infra/terraform
terraform init
terraform apply
```

After apply, push the Docker image to ECR and run jobs via the AWS Batch env + torchrun driver. For detailed steps, see ./AWS_BATCH_SETUP.md.

---

## Evaluation
- docs/EVALUATION.md – evaluation procedures, metrics, and examples.

---

## Webapp
- webapp/README.md – server, endpoints, and development notes.

---

## Reference & Tips
- Use the `refrag` venv consistently in local dev and inside Docker (mount at /workspace and create the venv there).
- Prefer Hydra overrides for configuration instead of hardcoding.
- For CI/CD, GitHub Actions can build and push to ECR; see workflow in `.github/workflows/` (if present) and AWS guides above.

Links
- Root README: ../README.md
- Architecture: ./ARCHITECTURE.md
- Torchrun Guide: ./TORCHRUN_GUIDE.md
- AWS Batch Setup: ./AWS_BATCH_SETUP.md
- CPU/GPU Queues: ./AWS_BATCH_CPU_GPU_QUEUES.md
- Evaluation: ./EVALUATION.md
- Webapp: ../webapp/README.md
