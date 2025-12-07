resource "aws_efs_file_system" "main" {
  creation_token = "${var.project}-efs"
  encrypted      = true
  tags = {
    Name    = "${var.project}-efs"
    Project = var.project
  }
}

resource "aws_security_group" "efs" {
  name_prefix = "${var.project}-efs-sg-"
  description = "Allow NFS from Batch instances"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description     = "NFS from default SG (Batch instances)"
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [data.aws_security_group.default.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name    = "${var.project}-efs-sg"
    Project = var.project
  }
}

resource "aws_efs_mount_target" "main" {
  for_each        = toset(data.aws_subnets.default.ids)
  file_system_id  = aws_efs_file_system.main.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

# Access points for structured paths
resource "aws_efs_access_point" "jobs" {
  file_system_id = aws_efs_file_system.main.id
  root_directory {
    path = var.efs_root_path_outputs
    creation_info {
      owner_uid   = 0
      owner_gid   = 0
      permissions = "0777"
    }
  }
  posix_user {
    uid = 0
    gid = 0
  }
  tags = {
    Name = "${var.project}-ap-jobs"
  }
}

resource "aws_efs_access_point" "cache" {
  file_system_id = aws_efs_file_system.main.id
  root_directory {
    path = var.efs_root_path_cache
    creation_info {
      owner_uid   = 0
      owner_gid   = 0
      permissions = "0777"
    }
  }
  posix_user {
    uid = 0
    gid = 0
  }
  tags = {
    Name = "${var.project}-ap-cache"
  }
}
