# AWS Batch Separate CPU and GPU Queues

## Overview

The AWS Batch infrastructure has been updated to use separate compute environments and job queues for CPU and GPU workloads. This provides better resource management, cost optimization, and prevents CPU jobs from being blocked by GPU instance availability.

---

## Changes Made

### 1. Terraform Infrastructure (`infra/terraform/`)

#### **batch.tf**
- Created separate compute environments:
  - `refrag-cpu-compute-env` - CPU-only instances
  - `refrag-gpu-compute-env` - GPU instances (g5.xlarge, g5.2xlarge, etc.)
- Created separate job queues:
  - `refrag-cpu-job-queue` - For CPU workloads
  - `refrag-gpu-job-queue` - For GPU workloads
- Created separate job definitions:
  - `refrag-cpu-job-def` - 4 vCPUs, 8192 MB memory
  - `refrag-gpu-job-def` - 8 vCPUs, 32768 MB memory, 1 GPU

#### **variables.tf**
Separated configuration for CPU and GPU:
```hcl
# CPU Configuration
batch_cpu_max_vcpus = 32  # Default
batch_cpu_instance_types = ["c6i.large", "m6i.large", ...]

# GPU Configuration
batch_gpu_max_vcpus = 64  # Default
batch_gpu_instance_types = ["g5.xlarge", "g5.2xlarge", "g5.4xlarge"]
```

#### **outputs.tf**
Added separate outputs for CPU and GPU resources:
- `batch_cpu_job_queue_name`
- `batch_cpu_job_definition_name`
- `batch_gpu_job_queue_name`
- `batch_gpu_job_definition_name`

### 2. AWS Batch Environment (`src/refrag/framework/envs/awsbatch.py`)

#### **Auto-Selection Logic**
The launcher now automatically selects the appropriate queue based on GPU requirements:

```python
def _get_queue_and_jobdef(self, gpus: int) -> tuple[str, str]:
    if gpus > 0:
        return self.gpu_job_queue, self.gpu_job_definition_name
    else:
        return self.cpu_job_queue, self.cpu_job_definition_name
```

#### **Configuration**
Added fields for separate queues:
- `cpu_job_queue` - Name of CPU queue
- `gpu_job_queue` - Name of GPU queue
- `cpu_job_definition_name` - CPU job definition
- `gpu_job_definition_name` - GPU job definition

### 3. Env Config (`conf/env/awsbatch.yaml`)

Updated to include both queue configurations:
```yaml
# Queue names (auto-selected based on GPU count)
cpu_job_queue: refrag-cpu-job-queue
gpu_job_queue: refrag-gpu-job-queue
cpu_job_definition_name: refrag-cpu-job-def
gpu_job_definition_name: refrag-gpu-job-def

# Manual override (leave empty for auto-selection)
job_queue: ""  # Auto-selects based on gpus parameter
job_definition_name: ""  # Auto-selects based on gpus parameter
```

### 4. Batch Submitter Script (`infra/scripts/batch_submitter.py`)

Updated to support GPU configuration:
- Added `USE_GPU` environment variable (0 or 1)
- Added `GPU_COUNT` environment variable
- Auto-selects queue based on `USE_GPU`
- Passes GPU resource requirements when registering job definitions

---

## Usage

### Automatic Queue Selection (Recommended)

The environment automatically selects the appropriate queue based on the `gpus` parameter:

**CPU Job (uses CPU queue):**
```bash
python run.py app=hello_dist env=awsbatch message="Hello from CPU"
# or explicitly:
python run.py app=hello_dist env=awsbatch env.gpus=0
```

**GPU Job (uses GPU queue):**
```bash
python run.py app=train env=awsbatch env.gpus=1
```

**Multiple GPUs:**
```bash
python run.py app=train env=awsbatch env.gpus=2 env.vcpus=16 env.memory=65536
```

### Manual Queue Override

If you need to explicitly specify a queue:

```bash
python run.py app=train env=awsbatch \
  env.job_queue=refrag-gpu-job-queue \
  env.job_definition_name=refrag-gpu-job-def \
  env.gpus=1
```

### Using Batch Submitter Script

**CPU Job:**
```bash
export USE_GPU=0
export ECR_IMAGE=<your-ecr-url>:latest
python infra/scripts/batch_submitter.py
```

**GPU Job:**
```bash
export USE_GPU=1
export GPU_COUNT=1
export ECR_IMAGE=<your-ecr-url>:latest
python infra/scripts/batch_submitter.py
```

---

## Benefits

### 1. **Resource Isolation**
- CPU jobs don't compete with GPU jobs for resources
- Separate scaling for CPU and GPU compute environments

### 2. **Cost Optimization**
- CPU jobs use cheaper instance types (c6i, m6i)
- GPU jobs only use expensive GPU instances when needed
- Independent max_vcpus limits for cost control

### 3. **Better Availability**
- CPU jobs aren't blocked by GPU instance unavailability
- GPU spot interruptions don't affect CPU workloads

### 4. **Simplified Configuration**
- Clear separation of concerns
- Easier to tune instance types per workload

---

## Deployment

### Deploy New Infrastructure

```bash
cd infra/terraform

# Preview changes
terraform plan

# Apply (will create new resources)
terraform apply
```

**Note:** This creates new compute environments and queues. Existing jobs will continue using old queue until completed.

### Migrate Existing Setup

If you already have Terraform deployed:

```bash
cd infra/terraform

# Remove old single queue from state
terraform state rm aws_batch_compute_environment.ce
terraform state rm aws_batch_job_queue.queue
terraform state rm aws_batch_job_definition.jobdef

# Apply new configuration
terraform apply
```

### Verify Deployment

```bash
# Check outputs
terraform output

# Should show:
# - batch_cpu_job_queue_name
# - batch_cpu_job_definition_name
# - batch_gpu_job_queue_name
# - batch_gpu_job_definition_name
```

---

## Configuration Examples

### Custom CPU Instance Types

```bash
terraform apply -var='batch_cpu_instance_types=["c6i.xlarge","c6i.2xlarge","m6i.xlarge"]'
```

### Custom GPU Instance Types

```bash
terraform apply -var='batch_gpu_instance_types=["g5.2xlarge","g5.4xlarge","p3.2xlarge"]'
```

### Increase GPU Capacity

```bash
terraform apply -var='batch_gpu_max_vcpus=128'
```

### terraform.tfvars Example

```hcl
region = "us-east-1"

# CPU Configuration
batch_cpu_max_vcpus = 64
batch_cpu_instance_types = [
  "c6i.large",
  "c6i.xlarge",
  "m6i.large",
  "m6i.xlarge"
]

# GPU Configuration  
batch_gpu_max_vcpus = 128
batch_gpu_instance_types = [
  "g5.xlarge",
  "g5.2xlarge",
  "g5.4xlarge",
  "g5.8xlarge"
]
```

---

## Monitoring

### Check Queue Status

```bash
# CPU Queue
aws batch describe-job-queues --job-queues refrag-cpu-job-queue

# GPU Queue
aws batch describe-job-queues --job-queues refrag-gpu-job-queue
```

### List Running Jobs by Queue

```bash
# CPU Jobs
aws batch list-jobs --job-queue refrag-cpu-job-queue --job-status RUNNING

# GPU Jobs
aws batch list-jobs --job-queue refrag-gpu-job-queue --job-status RUNNING
```

### Check Compute Environment Health

```bash
# CPU Compute Environment
aws batch describe-compute-environments \
  --compute-environments refrag-cpu-compute-env

# GPU Compute Environment
aws batch describe-compute-environments \
  --compute-environments refrag-gpu-compute-env
```

---

## Cost Comparison

### Example: 8-hour Training Job

**Using Shared Queue (Old):**
- Mixed instance pool including GPU instances
- Higher minimum cost even for CPU jobs

**Using Separate Queues (New):**

| Workload Type | Instance | Spot Price | 8hr Cost |
|--------------|----------|------------|----------|
| CPU (small) | c6i.large | ~$0.03/hr | ~$0.24 |
| CPU (medium) | m6i.xlarge | ~$0.08/hr | ~$0.64 |
| GPU (1x A10G) | g5.xlarge | ~$0.31/hr | ~$2.48 |
| GPU (2x A10G) | g5.4xlarge | ~$0.54/hr | ~$4.32 |

**Savings:** CPU-only jobs can save 90%+ by using appropriate instance types.

---

## Troubleshooting

### Job Stuck in RUNNABLE

**Check compute environment status:**
```bash
aws batch describe-compute-environments \
  --compute-environments refrag-cpu-compute-env refrag-gpu-compute-env
```

**Common causes:**
1. Max vCPUs reached - increase `batch_cpu_max_vcpus` or `batch_gpu_max_vcpus`
2. No spot capacity - add more instance types
3. Compute environment disabled - check state is "ENABLED"

### Wrong Queue Selected

The launcher selects queue based on `launcher.gpus`:
- `gpus=0` → CPU queue
- `gpus>0` → GPU queue

Override manually if needed:
```bash
python run.py app=train \
  env=aws_batch \
  env.job_queue=refrag-gpu-job-queue \
  env.gpus=1
```

### Queue Doesn't Exist After Terraform Apply

Terraform may need to destroy old resources first:
```bash
terraform state list  # Check what exists
terraform destroy -target=aws_batch_job_queue.queue  # Remove old queue
terraform apply  # Create new queues
```

---

## Rollback

If you need to rollback to a single queue:

```bash
cd infra/terraform

# Remove separate queues
terraform destroy \
  -target=aws_batch_job_queue.cpu_queue \
  -target=aws_batch_job_queue.gpu_queue \
  -target=aws_batch_compute_environment.cpu \
  -target=aws_batch_compute_environment.gpu

# Revert batch.tf to previous version
git checkout HEAD~1 infra/terraform/batch.tf

# Apply old configuration
terraform apply
```

---

## Future Enhancements

1. **Multi-region queues** - Deploy to multiple regions for better spot availability
2. **Priority queues** - Add high/low priority queues per type
3. **Auto-scaling policies** - Custom scaling based on job queue depth
4. **Cost tracking** - Tag jobs with cost allocation tags
5. **On-demand fallback** - Automatically use on-demand if spot unavailable

---

## Summary

The separate CPU and GPU queues provide:
- ✅ Better resource isolation
- ✅ Improved cost efficiency  
- ✅ Higher availability
- ✅ Automatic queue selection
- ✅ Easier configuration management

All existing code continues to work - the launcher automatically selects the appropriate queue based on GPU requirements.

