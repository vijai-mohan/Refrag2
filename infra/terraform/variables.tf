variable "region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "refrag"
}

variable "ecr_repo_name" {
  type    = string
  default = "refrag"
}

# Provide your own bucket name. Replace the placeholder or set to empty to let Terraform create one with a random suffix.
variable "s3_bucket" {
  type    = string
  default = "CONFIGURE_S3_FOLDER"
}

# CPU Batch Configuration
variable "batch_cpu_max_vcpus" {
  description = "Max vCPUs for CPU Batch compute environment"
  type        = number
  default     = 32
}

variable "batch_cpu_instance_types" {
  description = "Instance types for CPU-only Batch jobs"
  type        = list(string)
  default     = ["c6i.2xlarge", "m6i.2xlarge"]
}

# GPU Batch Configuration
variable "batch_gpu_max_vcpus" {
  description = "Max vCPUs for GPU Batch compute environment"
  type        = number
  default     = 192

}

variable "batch_gpu_instance_types" {
  description = "Instance types for GPU Batch jobs"
  type        = list(string)
  default     = ["g4ad.xlarge","g4ad.2xlarge","g4ad.4xlarge", "g4ad.8xlarge", "g4dn.2xlarge", "g4dn.12xlarge", "g4dn.metal", "g5.xlarge", "g5.2xlarge", "g5.4xlarge", "g5.8xlarge", "g5.12xlarge", "g5.24xlarge", "g5.48xlarge"]
}

variable "ecr_repo_use_existing" {
  description = "If true, use an existing ECR repository with name var.ecr_repo_name instead of creating a new one"
  type        = bool
  default     = true
}

variable "enable_efs" {
  type        = bool
  default     = true
  description = "Whether to create and mount EFS for Batch jobs"
}

variable "efs_root_path_outputs" {
  type        = string
  default     = "/jobs"
  description = "Root path in EFS for job outputs (access point root)"
}

variable "efs_root_path_cache" {
  type        = string
  default     = "/data/cache" # EFS access point cannot use segment starting with '.'; container still mounts to /root/.cache
  description = "Root path in EFS for huggingface cache (access point root)"
}
