# Conditional ECR repository: use existing if requested, else create

locals {
  use_existing_ecr = var.ecr_repo_use_existing
}

data "aws_ecr_repository" "existing" {
  count = local.use_existing_ecr ? 1 : 0
  name  = var.ecr_repo_name
}

resource "aws_ecr_repository" "repo" {
  count = local.use_existing_ecr ? 0 : 1
  name  = var.ecr_repo_name
  image_scanning_configuration {
    scan_on_push = true
  }
  tags = {
    Project = var.project
  }
}

locals {
  ecr_repository_url = local.use_existing_ecr ? data.aws_ecr_repository.existing[0].repository_url : aws_ecr_repository.repo[0].repository_url
  # Always use full repository URL for Batch job definitions; previous logic used short name when existing which breaks pull auth.
  ecr_image          = "${local.ecr_repository_url}:latest"
}
