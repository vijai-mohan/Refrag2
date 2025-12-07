data "aws_iam_policy_document" "ec2_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      identifiers = ["ec2.amazonaws.com"]
      type        = "Service"
    }
  }
}

data "aws_iam_policy_document" "ecs_tasks_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      identifiers = ["ecs-tasks.amazonaws.com"]
      type        = "Service"
    }
  }
}

data "aws_iam_policy_document" "batch_service_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      identifiers = ["batch.amazonaws.com"]
      type        = "Service"
    }
  }
}

resource "aws_iam_role" "batch_instance_role" {
  name               = "${var.project}-batch-instance-role"
  assume_role_policy = data.aws_iam_policy_document.ec2_assume_role.json
}

resource "aws_iam_instance_profile" "batch_instance_profile" {
  name = "${var.project}-batch-instance-profile"
  role = aws_iam_role.batch_instance_role.name
}

resource "aws_iam_role_policy_attachment" "batch_instance_role_attach" {
  role       = aws_iam_role.batch_instance_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonEC2ContainerServiceforEC2Role"
}

resource "aws_iam_role" "batch_service_role" {
  name               = "${var.project}-batch-service-role"
  assume_role_policy = data.aws_iam_policy_document.batch_service_assume_role.json
}

resource "aws_iam_role_policy_attachment" "batch_service_role_attach" {
  role       = aws_iam_role.batch_service_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSBatchServiceRole"
}

resource "aws_iam_role" "batch_job_role" {
  name               = "${var.project}-batch-job-role"
  assume_role_policy = data.aws_iam_policy_document.ecs_tasks_assume_role.json
}

data "aws_iam_policy_document" "batch_job_role_inline" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "ecr:GetAuthorizationToken",
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
      "sts:GetCallerIdentity"
    ]
    resources = [
      local.bucket_arn,
      "${local.bucket_arn}/*",
      "*" # ECR + STS require wildcard unless scoping to specific repos
    ]
  }
}

resource "aws_iam_role_policy" "batch_job_policy" {
  name = "${var.project}-batch-job-policy"
  role = aws_iam_role.batch_job_role.id
  policy = data.aws_iam_policy_document.batch_job_role_inline.json
}

resource "aws_iam_role_policy_attachment" "batch_execution_efs" {
  role       = aws_iam_role.batch_execution_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonElasticFileSystemClientReadWriteAccess"
}

resource "aws_iam_role_policy_attachment" "batch_job_efs" {
  role       = aws_iam_role.batch_job_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonElasticFileSystemClientReadWriteAccess"
}
