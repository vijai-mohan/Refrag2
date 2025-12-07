output "ecr_repository_url" {
  value       = local.ecr_repository_url
  description = "ECR repository URL for Docker images"
}

output "s3_bucket_name" {
  value       = local.bucket_name
  description = "S3 bucket name for data and artifacts"
}

# CPU Queue Outputs
output "batch_cpu_job_queue_name" {
  value       = aws_batch_job_queue.cpu_queue.name
  description = "AWS Batch CPU job queue name"
}

output "batch_cpu_job_definition_name" {
  value       = aws_batch_job_definition.cpu_jobdef.name
  description = "AWS Batch CPU job definition name"
}

# GPU Queue Outputs
output "batch_gpu_job_queue_name" {
  value       = aws_batch_job_queue.gpu_queue.name
  description = "AWS Batch GPU job queue name"
}

output "batch_gpu_job_definition_name" {
  value       = aws_batch_job_definition.gpu_jobdef.name
  description = "AWS Batch GPU job definition name"
}

# Backward compatibility (defaults to GPU queue)
output "batch_job_queue_name" {
  value       = aws_batch_job_queue.gpu_queue.name
  description = "Default AWS Batch job queue name (GPU)"
}

output "batch_job_definition_name" {
  value       = aws_batch_job_definition.gpu_jobdef.name
  description = "Default AWS Batch job definition name (GPU)"
}

output "efs_id" {
  value       = var.enable_efs ? aws_efs_file_system.main.id : null
  description = "EFS filesystem ID"
}

output "efs_access_point_jobs" {
  value       = var.enable_efs ? aws_efs_access_point.jobs.arn : null
  description = "EFS access point for /jobs"
}

output "efs_access_point_cache" {
  value       = var.enable_efs ? aws_efs_access_point.cache.arn : null
  description = "EFS access point for /data/.cache"
}
