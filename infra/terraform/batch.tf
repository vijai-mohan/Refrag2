# CPU Compute Environment
resource "aws_batch_compute_environment" "cpu" {
  compute_environment_name = "${var.project}-cpu-compute-env"
  service_role             = aws_iam_role.batch_service_role.arn
  type                     = "MANAGED"

  compute_resources {
    type                = "SPOT"
    min_vcpus           = 0
    max_vcpus           = var.batch_cpu_max_vcpus
    desired_vcpus       = 0

    # Terraform provider v5 does not accept instance_types list here; choose first as primary
    instance_type       = var.batch_cpu_instance_types
    subnets             = data.aws_subnets.default.ids
    security_group_ids  = [data.aws_security_group.default.id]
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    instance_role       = aws_iam_instance_profile.batch_instance_profile.arn
  }
}

# GPU Compute Environment
resource "aws_batch_compute_environment" "gpu" {
  compute_environment_name = "${var.project}-gpu-compute-env"
  service_role             = aws_iam_role.batch_service_role.arn
  type                     = "MANAGED"

  compute_resources {
    type                = "SPOT"
    min_vcpus           = 0
    max_vcpus           = var.batch_gpu_max_vcpus
    desired_vcpus       = 0
    instance_type       = var.batch_gpu_instance_types
    subnets             = data.aws_subnets.default.ids
    security_group_ids  = [data.aws_security_group.default.id]
    allocation_strategy = "SPOT_CAPACITY_OPTIMIZED"
    instance_role       = aws_iam_instance_profile.batch_instance_profile.arn
  }
}

# CPU Job Queue
resource "aws_batch_job_queue" "cpu_queue" {
  name     = "${var.project}-cpu-job-queue"
  priority = 1
  state    = "ENABLED"

  compute_environment_order {
    order                = 1
    compute_environment   = aws_batch_compute_environment.cpu.arn
  }
}

# GPU Job Queue
resource "aws_batch_job_queue" "gpu_queue" {
  name     = "${var.project}-gpu-job-queue"
  priority = 1
  state    = "ENABLED"

  compute_environment_order {
    order                = 1
    compute_environment   = aws_batch_compute_environment.gpu.arn
  }
}

locals {
  base_env = [
    { name = "S3_BUCKET", value = local.bucket_name },
    { name = "AWS_DEFAULT_REGION", value = var.region }
  ]

  efs_volumes = var.enable_efs ? [
    {
      name = "efs-jobs"
      efsVolumeConfiguration = {
        fileSystemId      = aws_efs_file_system.main.id
        transitEncryption = "ENABLED"
        authorizationConfig = {
          accessPointId = aws_efs_access_point.jobs.id
          iam           = "ENABLED"
        }
      }
    },
    {
      name = "efs-cache"
      efsVolumeConfiguration = {
        fileSystemId      = aws_efs_file_system.main.id
        transitEncryption = "ENABLED"
        authorizationConfig = {
          accessPointId = aws_efs_access_point.cache.id
          iam           = "ENABLED"
        }
      }
    }
  ] : []

  efs_mount_points = var.enable_efs ? [
    { sourceVolume = "efs-jobs", containerPath = "/workspace/outputs", readOnly = false },
    { sourceVolume = "efs-cache", containerPath = "/root/.cache", readOnly = false }
  ] : []

  cpu_container_props = merge(
    {
      image   = local.ecr_image
      vcpus   = 4
      memory  = 8192
      command = ["python", "run.py"]
      environment = local.base_env
      jobRoleArn       = aws_iam_role.batch_job_role.arn
      executionRoleArn = aws_iam_role.batch_execution_role.arn
    },
    length(local.efs_volumes) > 0 ? { volumes = local.efs_volumes } : {},
    length(local.efs_mount_points) > 0 ? { mountPoints = local.efs_mount_points } : {}
  )

  gpu_container_props = merge(
    {
      image   = local.ecr_image
      vcpus   = 8
      memory  = 32768
      command = ["python", "run.py"]
      environment = local.base_env
      jobRoleArn       = aws_iam_role.batch_job_role.arn
      executionRoleArn = aws_iam_role.batch_execution_role.arn
      resourceRequirements = [
        { type = "GPU", value = "1" }
      ]
    },
    length(local.efs_volumes) > 0 ? { volumes = local.efs_volumes } : {},
    length(local.efs_mount_points) > 0 ? { mountPoints = local.efs_mount_points } : {}
  )
}

# CPU Job Definition
resource "aws_batch_job_definition" "cpu_jobdef" {
  name = "${var.project}-cpu-job-def"
  type = "container"

  container_properties = jsonencode(local.cpu_container_props)
}

# GPU Job Definition
resource "aws_batch_job_definition" "gpu_jobdef" {
  name = "${var.project}-gpu-job-def"
  type = "container"

  container_properties = jsonencode(local.gpu_container_props)
}

# Execution role for Batch to pull from ECR and write logs
resource "aws_iam_role" "batch_execution_role" {
  name = "${var.project}-batch-execution-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "batch_execution_ecr" {
  role       = aws_iam_role.batch_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_role_policy_attachment" "batch_execution_logs" {
  role       = aws_iam_role.batch_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess"
}
