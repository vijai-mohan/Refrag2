data "aws_vpc" "default" {
  default = true
}

# Collect all public subnets in the default VPC (filter by mapPublicIpOnLaunch)
data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}

# Default security group in the default VPC
data "aws_security_group" "default" {
  filter {
    name   = "group-name"
    values = ["default"]
  }
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
}
