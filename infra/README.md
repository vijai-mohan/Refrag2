# Infrastructure (Terraform & AWS Batch)

Single place for provisioning and operating Refrag infrastructure with Terraform and AWS Batch.

## Quick setup
1. Install Terraform (>=1.6), AWS CLI, and Docker; activate the `refrag` venv if you use helper scripts.
2. Configure AWS credentials (`aws configure`) and verify with `aws sts get-caller-identity`.
3. Pick a unique S3 bucket for this deployment and pass it explicitly (do **not** leave the placeholder `CONFIGURE_S3_FOLDER`): `-var="s3_bucket=<your-bucket-name>"` or via `terraform.tfvars`.
4. Deploy: `cd infra/terraform && terraform init && terraform apply -var="s3_bucket=<your-bucket-name>"` (add other `-var` overrides as needed).
5. Capture outputs: `terraform output -raw ecr_repository_url`, `terraform output -raw s3_bucket_name`, `terraform output -raw batch_job_queue_name`.
6. Build and push an image to ECR from repo root:
   ```bash
   ECR_REPO=$(cd infra/terraform && terraform output -raw ecr_repository_url)
   docker build -t refrag:latest .
   docker tag refrag:latest $ECR_REPO:latest
   aws ecr get-login-password --region ${AWS_REGION:-us-east-1} | docker login --username AWS --password-stdin ${ECR_REPO%%/*}
   docker push $ECR_REPO:latest
   ```
7. Submit a smoke job (Python launcher): `python run.py app=hello_dist env=aws_batch message="Hello from Batch"` or use `infra/scripts/batch_submitter.py` with the outputs above.

---

## Prerequisites
- Terraform installed (macOS: `brew install terraform`; Linux/WSL: download zip from HashiCorp; Windows: Chocolatey or installer).
- AWS CLI installed (`pip install awscli`) and configured (`aws configure`).
- Docker available locally for building/pushing images.
- Permissions to create ECR, S3, Batch, and IAM resources in your target AWS account/region.

## Terraform workflow
- Plan/apply:
  ```bash
  cd infra/terraform
  terraform init
  terraform plan -var="s3_bucket=<your-bucket-name>"
  terraform apply -var="s3_bucket=<your-bucket-name>"
  ```
- Destroy (dangerous): `terraform destroy -var="s3_bucket=<your-bucket-name>"`.
- Targeted updates: `terraform apply -target=aws_batch_job_definition.jobdef -var="s3_bucket=<your-bucket-name>"`.

## Configuration highlights
- Required override: `s3_bucket` must be set to a bucket you own; the default `CONFIGURE_S3_FOLDER` is a placeholder and should be replaced (or set to empty to let Terraform create a random-suffix bucket).
- Common overrides: `region` (default `us-east-1`), `project` prefix, `batch_max_vcpus`, `batch_instance_types` (CPU/GPU mix).
- Use `terraform.tfvars` to keep overrides together (example):
  ```hcl
  region          = "us-west-2"
  project         = "refrag"
  s3_bucket       = "my-refrag-bucket"
  batch_max_vcpus = 64
  ```

## Outputs you need
- `ecr_repository_url` for tagging/pushing images.
- `s3_bucket_name` for data and job artifacts.
- `batch_job_queue_name` and `batch_job_definition_name` for submissions.
View with `terraform output` or `terraform output -raw <name>`.

## Run jobs on AWS Batch
- Python launcher (recommended): `python run.py app=hello_dist env=aws_batch message="hello"`.
- Scripted submitter: set env vars from Terraform outputs, then `python infra/scripts/batch_submitter.py`.
- AWS CLI example:
  ```bash
  aws batch submit-job \
    --job-name refrag-test \
    --job-queue "$(cd infra/terraform && terraform output -raw batch_job_queue_name)" \
    --job-definition "$(cd infra/terraform && terraform output -raw batch_job_definition_name)" \
    --container-overrides '{"command":["python","run.py","app=hello_dist"]}'
  ```

## Troubleshooting
- Auth issues: `aws sts get-caller-identity` and rerun `aws configure` if needed.
- Stuck jobs/capacity: add more `batch_instance_types` or adjust `batch_max_vcpus`, then `terraform apply`.
- Terraform cache problems: remove `.terraform` and `.terraform.lock.hcl`, then `terraform init`.

For architecture background see `docs/ARCHITECTURE.md`; AWS Batch specifics remain in `docs/AWS_BATCH_SETUP.md` and `docs/AWS_BATCH_CPU_GPU_QUEUES.md` if you need deeper detail.
