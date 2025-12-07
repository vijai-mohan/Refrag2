# Repository Guidelines

## Project Structure & Modules
- Core model + tokenizer live under `src/refrag/model/` (`modeling_refrag.py`, `tokenization_refrag.py`, `alignment_trainer.py`, `config_refrag.py`).
- Hydra entrypoint is `run.py` with configs under `conf/` (`app/`, `env/`, `driver/` groups); default app is `chatcli`.
- Hydra apps/drivers/envs live under `src/refrag/framework/` (e.g., `src/refrag/framework/apps/train.py`, `envs/local.py`, `drivers/torchrun.py`).
- Docs sit in `docs/` (start with `README.md`, `ARCHITECTURE.md`, `RefragPaper.pdf` for context). Infra/Terraform in `infra/`; web demo in `webapp/`; tests in `tests/`.

## Build, Test, and Development Commands
- Always work inside the root venv named `refrag`: `source refrag/bin/activate` then `pip install -r requirements.txt`.
- Quick sanity runs: `python run.py app=hello_dist` (inline demo), `python run.py app=train env=torchrun driver=torchrun driver.nproc_per_node=2` (distributed training), `python run.py app=alignment` (alignment path, see conf for overrides).
- Web server: `cd webapp && python server.py` (Flask SSE endpoints).
- Tests: `pytest -q` or scope down, e.g., `pytest tests/refrag/tokenizer -q`.
- Docker/AWS Batch: follow `docs/AWS_BATCH_SETUP.md`; Terraform from `infra/terraform` (`terraform init && terraform apply`).

## Coding Style & Naming Conventions
- Python: prefer type hints, snake_case functions/variables, CapWords classes. Keep functions small and composable.
- Formatting: run `black` and `isort` (defaults) before committing; avoid manual style drift.
- Config: use Hydra overrides instead of hardcoding; keep new options in the proper group under `conf/`.

## Testing Guidelines
- Framework: `pytest`. Place new tests in `tests/` mirroring module paths (e.g., `tests/refrag/tokenizer/test_<topic>.py`).
- Name tests descriptively (`test_parses_multiple_docs`, etc.) and keep fixtures small. Use tiny tokenizer/model names to keep runtime low.
- For new Hydra apps or drivers, add at least a smoke test that instantiates them with a minimal config.

## Commit & Pull Request Guidelines
- Commits are short, present-tense summaries (e.g., `Fix alignment packing`, `Update failing tests`). Squash only if asked.
- PRs: include purpose, key commands run (`pytest`, `python run.py ...`), and any config or infra changes. Link issues/tasks; add screenshots for webapp/UI tweaks. Note if Refrag paper assumptions or tokenizer alignment logic changed.

## Security & Configuration Tips
- Do not commit secrets or AWS creds; prefer environment variables and `.env` files ignored by git.
- When running Batch/Terraform, confirm the correct AWS profile/region; avoid writing outside the repo unless intentional.
- Background reading: skim `RefragPaper.pdf` and `docs/ARCHITECTURE.md` before changing alignment/tokenizer logic to keep terminology consistent.
