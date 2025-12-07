# S3 bucket: use existing if var.s3_bucket is provided, else create a new one with random suffix

locals {
  have_existing_bucket = length(var.s3_bucket) > 0
  bucket_name          = local.have_existing_bucket ? var.s3_bucket : "${var.project}-data-${random_id.suffix.hex}"
}

data "aws_s3_bucket" "existing" {
  count  = local.have_existing_bucket ? 1 : 0
  bucket = local.bucket_name
}

resource "aws_s3_bucket" "data" {
  count  = local.have_existing_bucket ? 0 : 1
  bucket = local.bucket_name
  tags = {
    Project = var.project
  }
}

# Enforce bucket owner controls only when we create the bucket
resource "aws_s3_bucket_ownership_controls" "data" {
  count  = local.have_existing_bucket ? 0 : 1
  bucket = aws_s3_bucket.data[0].id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

# Unified local for downstream references (ARN)
locals {
  bucket_arn = local.have_existing_bucket ? data.aws_s3_bucket.existing[0].arn : aws_s3_bucket.data[0].arn
}
