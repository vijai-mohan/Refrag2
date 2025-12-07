# AWS Batch Setup and Usage Guide

Complete guide to setting up and using AWS Batch with Refrag for cost-effective LLM training.

## Overview

This setup allows you to:
1. Run training jobs on AWS Batch with Spot instances (~70-90% cost savings)
2. Use GPU instances (g5.xlarge, g5.2xlarge, etc.) on-demand
3. Automatically sync your local code to S3 before job execution
4. Submit jobs directly from command line using Hydra

---

## One-Time Setup

### Step 1: Install Prerequisites

```bash
# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/

# Install AWS CLI
pip install awscli boto3

# Configure AWS credentials
aws configure
```

### Step 2: Create IAM User for GitHub Actions

```bash
# Create user
aws iam create-user --user-name github-ecr-push-user

# Attach ECR push policy
aws iam put-user-policy --user-name github-ecr-push-user --policy-name ECRPushPolicy --policy-document '{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "*"
    }
  ]
}'

# Create access key (save the output!)
aws iam create-access-key --user-name github-ecr-push-user
```

### Step 3: Add GitHub Secrets

Go to your repository → Settings → Secrets and variables → Actions:

1. Add `AWS_ACCESS_KEY_ID` - From previous step
2. Add `AWS_SECRET_ACCESS_KEY` - From previous step

### Step 4: Deploy Infrastructure with Terraform

```bash
cd infra/terraform

# Initialize
terraform init

# Preview changes
terraform plan

# Deploy (type 'yes' when prompted)
terraform apply

# Save outputs for later use
terraform output -json > outputs.json
```

**What gets created:**
- ECR repository for Docker images
- S3 bucket for data and code overlays
- AWS Batch compute environment (Spot instances)
- AWS Batch job queue
- AWS Batch job definition template
- IAM roles and policies

### Step 5: Build and Push Docker Image

```bash
# Navigate to project root
cd ../..

# Get ECR repository URL
ECR_REPO=$(cd infra/terraform && terraform output -raw ecr_repository_url)

# Login to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ${ECR_REPO%%/*}

# Build image with BuildKit cache
export DOCKER_BUILDKIT=1
docker build -t refrag:latest .

# Tag and push
docker tag refrag:latest $ECR_REPO:latest
docker push $ECR_REPO:latest
```

---

## Usage

### Run a Job on AWS Batch

**Basic usage:**

```bash
python run.py app=hello_dist env=awsbatch message="Hello from AWS Batch"
```

**With GPU:**

```bash
python run.py app=hello_dist env=awsbatch env.gpus=1 env.vcpus=8 env.memory=16384
```

**Training example:**

```bash
python run.py app=train env=awsbatch \
  env.gpus=1 \
  env.vcpus=8 \
  env.memory=32768 \
  model=gemma_1b \
  train.epochs=3
```

**Disable code overlay (use image code only):**

```bash
python run.py app=hello_dist env=awsbatch env.overlay.enabled=false
```

---

## Configuration

### Environment Configuration

Edit `conf/env/awsbatch.yaml`:

```yaml
# Compute resources
vcpus: 4
memory: 8192
gpus: 0

# AWS settings
region: us-east-1
job_queue: refrag-job-queue
job_definition_name: refrag-job-def

# Overlay settings
overlay:
  enabled: true
  s3_bucket: leykainc
  s3_prefix: jobs/batch
```

### Terraform Configuration

Edit `infra/terraform/terraform.tfvars`:

```hcl
region               = "us-west-2"
batch_max_vcpus      = 128
batch_instance_types = ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"]
```

Apply changes:

```bash
cd infra/terraform
terraform apply
```

---

## Monitoring

### View Job Status

```bash
# List running jobs
aws batch list-jobs --job-queue refrag-job-queue --job-status RUNNING

# Describe specific job
aws batch describe-jobs --jobs <JOB_ID>
```

### View Logs

```bash
# Get log stream from job description
aws batch describe-jobs --jobs <JOB_ID> \
  --query 'jobs[0].container.logStreamName' --output text

# Tail logs
aws logs tail /aws/batch/job --follow --log-stream-names <LOG_STREAM>
```

### AWS Console

- **Batch Dashboard**: https://console.aws.amazon.com/batch/
- **CloudWatch Logs**: https://console.aws.amazon.com/cloudwatch/

---

## Cost Estimation

### Typical Costs (Spot Instances)

| Instance Type | vCPUs | RAM | GPU | Spot Price/hr | On-Demand Price/hr |
|---------------|-------|-----|-----|---------------|-------------------|
| c6i.large | 2 | 4 GB | 0 | ~$0.03 | ~$0.085 |
| m6i.large | 2 | 8 GB | 0 | ~$0.04 | ~$0.096 |
| g5.xlarge | 4 | 16 GB | 1 (A10G) | ~$0.31 | ~$1.01 |
| g5.2xlarge | 8 | 32 GB | 1 (A10G) | ~$0.36 | ~$1.21 |
| g5.4xlarge | 16 | 64 GB | 1 (A10G) | ~$0.54 | ~$1.64 |

### Example Training Costs

**Small model (1B params, 8 hours on g5.xlarge):**
- Spot: $0.31 × 8 = **$2.48**
- On-demand: $1.01 × 8 = $8.08

**Medium model (7B params, 24 hours on g5.2xlarge):**
- Spot: $0.36 × 24 = **$8.64**
- On-demand: $1.21 × 24 = $29.04

**Additional costs:**
- S3 storage: ~$0.023/GB/month
- Data transfer: First 100GB/month free
- CloudWatch Logs: ~$0.50/GB ingested

---

## Troubleshooting

### Job Submission Fails

**Check ECR image exists:**
```bash
aws ecr describe-images --repository-name refrag-repo --region us-east-1
```

**Check compute environment is enabled:**
```bash
aws batch describe-compute-environments \
  --compute-environments refrag-compute-env
```

### Job Stuck in RUNNABLE

This means AWS Batch can't find available spot capacity.

**Solutions:**
1. Add more instance types to `batch_instance_types`
2. Increase `batch_max_vcpus`
3. Try different region
4. Use on-demand instances (modify Terraform)

**Quick fix:**
```bash
cd infra/terraform
terraform apply -var='batch_instance_types=["g5.xlarge","g5.2xlarge","g5.4xlarge","p3.2xlarge"]'
```

### Overlay Upload Fails

**Check S3 bucket permissions:**
```bash
aws s3 ls s3://leykainc/
```

**Check IAM role:**
```bash
aws iam get-role-policy --role-name refrag-batch-job-role --policy-name refrag-batch-job-policy
```

### Out of Memory

Increase memory in the env config:

```bash
python run.py app=train env=awsbatch env.memory=32768
```

Or update default in `conf/env/awsbatch.yaml`

---

## CI/CD Integration

### GitHub Actions Auto-Deploy

The workflow in `.github/workflows/deploy-ecr.yml` automatically:
1. Builds Docker image on push to `main`
2. Pushes to ECR with commit SHA and `latest` tags
3. Ready for Batch jobs immediately

**Manual trigger:**
Go to Actions → Build and Push to ECR → Run workflow

---

## Advanced Usage

### Custom Instance Types

```bash
python run.py app=train env=awsbatch \
  env.gpu_instance_types='[g5.2xlarge,g5.4xlarge]'
```

### Environment Variables

```bash
export WANDB_API_KEY=your_key
python run.py app=train env=awsbatch
```

Environment variables in `env.env_passthrough` are automatically passed to jobs.

### Multiple Regions

Deploy Terraform in multiple regions for redundancy:

```bash
terraform apply -var="region=us-west-2"
```

Update the env config to match.

---

## Best Practices

1. **Use Spot Instances** - Default configuration saves ~70-90%
2. **Enable Overlay** - Quickly test code changes without rebuilding image
3. **Monitor Costs** - Use AWS Cost Explorer to track spending
4. **Set Budgets** - Create AWS Budget alerts for cost overruns
5. **Clean Up** - Run `terraform destroy` when not actively training
6. **Version Images** - Tag images with commit SHA for reproducibility
7. **Checkpoint Frequently** - Save checkpoints to S3 in case of spot interruption

---

## Cleanup

### Temporary Cleanup (Keep Infrastructure)

```bash
# Stop running jobs
aws batch terminate-job --job-id <JOB_ID> --reason "Manual termination"

# Disable compute environment (stops scaling)
aws batch update-compute-environment \
  --compute-environment refrag-compute-env \
  --state DISABLED
```

### Full Cleanup (Delete Everything)

```bash
cd infra/terraform
terraform destroy
```

Type `yes` to confirm.

**⚠️ Warning:** This deletes:
- All Docker images in ECR
- All data in S3 bucket
- All Batch resources
- All IAM roles

---

## Next Steps

1. **Test the setup**: Run `app=hello_dist` to verify everything works
2. **Train a model**: Try `app=train` with a small model
3. **Monitor costs**: Set up AWS Budgets and Cost Alerts
4. **Optimize**: Adjust instance types based on workload

For detailed Terraform documentation, see [infra/terraform/README.md](infra/terraform/README.md)

