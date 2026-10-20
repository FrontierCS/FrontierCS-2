data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}
data "aws_availability_zones" "available" {
  state = "available"
}

check "exact_aws_account" {
  assert {
    condition     = data.aws_caller_identity.current.account_id == var.expected_account_id
    error_message = "AWS caller account does not match expected_account_id."
  }
}

check "private_vpc" {
  assert {
    condition     = startswith(var.vpc_cidr, "10.")
    error_message = "vpc_cidr must use a /16 inside 10.0.0.0/8."
  }
}

check "service_network_is_separate" {
  assert {
    condition     = cidrhost(var.vpc_cidr, 1) != cidrhost(var.service_ipv4_cidr, 1)
    error_message = "service_ipv4_cidr must not equal vpc_cidr."
  }
}

locals {
  name = "frontiercs2-${var.site_name}"
  azs  = slice(data.aws_availability_zones.available.names, 0, var.availability_zone_count)

  private_subnets = [
    for index, _ in local.azs : cidrsubnet(var.vpc_cidr, 8, index)
  ]
  public_subnets = [
    for index, _ in local.azs : cidrsubnet(var.vpc_cidr, 8, index + 100)
  ]

  artifact_bucket_name = "frontiercs2-${var.site_name}-${data.aws_caller_identity.current.account_id}-${var.aws_region}-artifacts"
  common_tags = merge(var.tags, {
    "frontier:site"      = var.site_name
    "frontier:lifecycle" = "persistent"
  })
}

module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "6.6.1"

  name = local.name
  cidr = var.vpc_cidr

  azs             = local.azs
  private_subnets = local.private_subnets
  public_subnets  = local.public_subnets

  enable_dns_hostnames = true
  enable_dns_support   = true

  enable_nat_gateway = true
  single_nat_gateway = var.single_nat_gateway

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = "1"
  }
  public_subnet_tags = {
    "kubernetes.io/role/elb" = "1"
  }

  tags = local.common_tags
}

resource "aws_kms_key" "platform" {
  description             = "FrontierCS2 ${var.site_name} platform data"
  deletion_window_in_days = 30
  enable_key_rotation     = true

  lifecycle {
    prevent_destroy = true
  }

  tags = local.common_tags
}

resource "aws_kms_alias" "platform" {
  name          = "alias/${local.name}"
  target_key_id = aws_kms_key.platform.key_id
}

data "aws_iam_policy_document" "ebs_csi_assume" {
  statement {
    sid     = "EksPodIdentity"
    effect  = "Allow"
    actions = ["sts:AssumeRole", "sts:TagSession"]

    principals {
      type        = "Service"
      identifiers = ["pods.eks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "ebs_csi" {
  name               = "${local.name}-ebs-csi"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "ebs_csi" {
  role       = aws_iam_role.ebs_csi.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEBSCSIDriverPolicy"
}

data "aws_iam_policy_document" "ebs_csi_kms" {
  statement {
    sid = "UseSiteKmsKey"
    actions = [
      "kms:CreateGrant",
      "kms:Decrypt",
      "kms:DescribeKey",
      "kms:Encrypt",
      "kms:GenerateDataKeyWithoutPlaintext",
      "kms:ReEncrypt*"
    ]
    resources = [aws_kms_key.platform.arn]
  }
}

resource "aws_iam_role_policy" "ebs_csi_kms" {
  name   = "site-kms-key"
  role   = aws_iam_role.ebs_csi.id
  policy = data.aws_iam_policy_document.ebs_csi_kms.json
}

resource "aws_iam_role" "efs_csi" {
  name               = "${local.name}-efs-csi"
  assume_role_policy = data.aws_iam_policy_document.ebs_csi_assume.json
  tags               = local.common_tags
}

resource "aws_iam_role_policy_attachment" "efs_csi" {
  role       = aws_iam_role.efs_csi.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/service-role/AmazonEFSCSIDriverPolicy"
}

module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "21.24.0"

  name               = local.name
  kubernetes_version = var.kubernetes_version
  service_ipv4_cidr  = var.service_ipv4_cidr

  authentication_mode                      = "API"
  enable_cluster_creator_admin_permissions = true
  enable_irsa                              = true
  deletion_protection                      = true
  endpoint_private_access                  = true
  endpoint_public_access                   = true
  endpoint_public_access_cidrs             = var.operator_cidrs
  enabled_log_types                        = ["api", "audit", "authenticator", "controllerManager", "scheduler"]
  cloudwatch_log_group_retention_in_days   = 30
  compute_config                           = { enabled = false }
  create_auto_mode_iam_resources           = false

  access_entries = {
    for principal in var.operator_principals : principal => {
      principal_arn = principal
      policy_associations = {
        cluster_admin = {
          policy_arn = "arn:${data.aws_partition.current.partition}:eks::aws:cluster-access-policy/AmazonEKSClusterAdminPolicy"
          access_scope = {
            type = "cluster"
          }
        }
      }
    }
  }

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  encryption_config = {
    resources        = ["secrets"]
    provider_key_arn = aws_kms_key.platform.arn
  }

  addons = {
    coredns    = {}
    kube-proxy = {}
    eks-pod-identity-agent = {
      before_compute = true
    }
    vpc-cni = {
      before_compute = true
      configuration_values = jsonencode({
        enableNetworkPolicy = "true"
        env = {
          # A fresh cluster cannot enter strict mode in the same apply that
          # creates its first nodes: CoreDNS/EBS/VPC-CNI bootstrap policies
          # must exist first. The Management-only profile stays standard and
          # the future production installer owns that explicit transition.
          NETWORK_POLICY_ENFORCING_MODE = "standard"
          # Reserve addresses, not an entire spare ENI, in the /24 subnets.
          WARM_IP_TARGET    = "2"
          MINIMUM_IP_TARGET = "8"
        }
      })
    }
    aws-ebs-csi-driver = {
      pod_identity_association = [{
        role_arn        = aws_iam_role.ebs_csi.arn
        service_account = "ebs-csi-controller-sa"
      }]
    }
    aws-efs-csi-driver = {
      pod_identity_association = [{
        role_arn        = aws_iam_role.efs_csi.arn
        service_account = "efs-csi-controller-sa"
      }]
    }
  }

  eks_managed_node_groups = {
    system = {
      name                     = "${local.name}-system"
      iam_role_use_name_prefix = false
      iam_role_name            = "${local.name}-system-node"
      # Management and LiteLLM Postgres sit on zonal gp3 volumes. A multi-AZ
      # system pool strands those volumes whenever a node rollout lands in
      # another zone, so the pool is pinned to the first private subnet.
      subnet_ids     = [module.vpc.private_subnets[0]]
      ami_type       = "AL2023_x86_64_STANDARD"
      instance_types = var.system_node_group.instance_types
      capacity_type  = "ON_DEMAND"
      min_size       = var.system_node_group.min_size
      desired_size   = var.system_node_group.desired_size
      max_size       = var.system_node_group.max_size

      metadata_options = {
        http_endpoint               = "enabled"
        http_put_response_hop_limit = 1
        http_tokens                 = "required"
      }

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = var.system_node_group.root_volume_gib
            volume_type           = "gp3"
            encrypted             = true
            delete_on_termination = true
          }
        }
      }

      labels = {
        "frontier.dev/site" = var.site_name
        "frontier.dev/pool" = "system"
      }

      cloudinit_pre_nodeadm = [{
        content_type = "text/x-shellscript; charset=\"us-ascii\""
        content      = <<-EOT
          #!/bin/bash
          set -euo pipefail
          endpoint=${var.local_registry_endpoint}
          install -d -m 0755 "/etc/containerd/certs.d/$endpoint"
          printf '%s\n' \
            "server = \"http://$endpoint\"" \
            "[host.\"http://$endpoint\"]" \
            '  capabilities = ["pull", "resolve"]' \
            > "/etc/containerd/certs.d/$endpoint/hosts.toml"
          chmod 0644 "/etc/containerd/certs.d/$endpoint/hosts.toml"
        EOT
      }]
    }

    cpu = {
      name                     = "${local.name}-cpu"
      iam_role_use_name_prefix = false
      iam_role_name            = "${local.name}-cpu-node"
      ami_type                 = "AL2023_x86_64_STANDARD"
      instance_types           = var.cpu_node_group.instance_types
      capacity_type            = "ON_DEMAND"
      min_size                 = var.cpu_node_group.min_size
      desired_size             = var.cpu_node_group.desired_size
      max_size                 = var.cpu_node_group.max_size

      metadata_options = {
        http_endpoint               = "enabled"
        http_put_response_hop_limit = 1
        http_tokens                 = "required"
      }

      block_device_mappings = {
        xvda = {
          device_name = "/dev/xvda"
          ebs = {
            volume_size           = var.cpu_node_group.root_volume_gib
            volume_type           = "gp3"
            encrypted             = true
            delete_on_termination = true
          }
        }
      }

      labels = {
        "frontier.dev/site" = var.site_name
        "frontier.dev/pool" = "cpu"
      }


      cloudinit_pre_nodeadm = [{
        content_type = "text/x-shellscript; charset=\"us-ascii\""
        content      = <<-EOT
          #!/bin/bash
          set -euo pipefail
          endpoint=${var.local_registry_endpoint}
          install -d -m 0755 "/etc/containerd/certs.d/$endpoint"
          printf '%s\n' \
            "server = \"http://$endpoint\"" \
            "[host.\"http://$endpoint\"]" \
            '  capabilities = ["pull", "resolve"]' \
            > "/etc/containerd/certs.d/$endpoint/hosts.toml"
          chmod 0644 "/etc/containerd/certs.d/$endpoint/hosts.toml"
        EOT
      }]
    }
  }

  tags = local.common_tags
}

data "aws_iam_policy_document" "management_artifact_assume" {
  statement {
    sid     = "ManagementWebIdentity"
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${replace(module.eks.cluster_oidc_issuer_url, "https://", "")}:sub"
      values   = ["system:serviceaccount:frontier-system:frontier-management"]
    }
  }
}

resource "aws_iam_role" "management_artifact_reader" {
  name               = "${local.name}-management-artifact-reader"
  assume_role_policy = data.aws_iam_policy_document.management_artifact_assume.json
  tags = merge(local.common_tags, {
    "frontier:component" = "management-artifact-reader"
  })
}

data "aws_iam_policy_document" "management_artifact_reader" {
  dynamic "statement" {
    for_each = length(var.manual_capacity_nodegroup_arns) == 2 ? [1] : []
    content {
      sid       = "ObserveManualWorkerCapacity"
      effect    = "Allow"
      actions   = ["eks:DescribeNodegroup", "eks:ListUpdates", "eks:DescribeUpdate"]
      resources = concat(var.manual_capacity_nodegroup_arns, [module.eks.cluster_arn])
    }
  }
  dynamic "statement" {
    for_each = length(var.manual_capacity_nodegroup_arns) == 2 ? [1] : []
    content {
      sid       = "ApplyManualWorkerCapacity"
      effect    = "Allow"
      actions   = ["eks:UpdateNodegroupConfig"]
      resources = var.manual_capacity_nodegroup_arns
    }
  }
  statement {
    sid       = "ReadRunEvidence"
    effect    = "Allow"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.artifacts.arn}/runs/*"]
  }

  statement {
    sid       = "DecryptRunEvidence"
    effect    = "Allow"
    actions   = ["kms:Decrypt"]
    resources = [aws_kms_key.platform.arn]

    condition {
      test     = "StringEquals"
      variable = "kms:ViaService"
      values   = ["s3.${var.aws_region}.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "management_artifact_reader" {
  name   = "run-evidence-read"
  role   = aws_iam_role.management_artifact_reader.id
  policy = data.aws_iam_policy_document.management_artifact_reader.json
}

resource "aws_security_group" "efs" {
  name_prefix = "${local.name}-efs-"
  description = "NFS from FrontierCS2 EKS nodes only"
  vpc_id      = module.vpc.vpc_id

  tags = local.common_tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_vpc_security_group_ingress_rule" "efs_from_nodes" {
  security_group_id            = aws_security_group.efs.id
  referenced_security_group_id = module.eks.node_security_group_id
  from_port                    = 2049
  to_port                      = 2049
  ip_protocol                  = "tcp"
  description                  = "EFS CSI mounts from EKS nodes"
}

resource "aws_vpc_security_group_egress_rule" "efs_all" {
  security_group_id = aws_security_group.efs.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1"
  description       = "Return traffic for NFS sessions"
}

resource "aws_efs_file_system" "preview" {
  encrypted        = true
  kms_key_id       = aws_kms_key.platform.arn
  performance_mode = "generalPurpose"
  throughput_mode  = "bursting"

  lifecycle {
    prevent_destroy = true
  }

  tags = merge(local.common_tags, {
    Name = "${local.name}-preview-shared"
  })
}

resource "aws_efs_mount_target" "preview" {
  for_each = {
    for index, subnet_id in module.vpc.private_subnets : local.azs[index] => subnet_id
  }

  file_system_id  = aws_efs_file_system.preview.id
  subnet_id       = each.value
  security_groups = [aws_security_group.efs.id]
}

resource "aws_efs_access_point" "object_store" {
  file_system_id = aws_efs_file_system.preview.id

  posix_user {
    uid = 65532
    gid = 65532
  }

  root_directory {
    path = "/frontier-object-store"
    creation_info {
      owner_uid   = 65532
      owner_gid   = 65532
      permissions = "0770"
    }
  }

  tags = merge(local.common_tags, {
    "frontier:component" = "preview-object-store"
  })
}

resource "aws_efs_access_point" "registry" {
  file_system_id = aws_efs_file_system.preview.id

  posix_user {
    uid = 65532
    gid = 65532
  }

  root_directory {
    path = "/frontier-registry"
    creation_info {
      owner_uid   = 65532
      owner_gid   = 65532
      permissions = "0770"
    }
  }

  tags = merge(local.common_tags, {
    "frontier:component" = "preview-registry"
  })
}

resource "aws_s3_bucket" "artifacts" {
  bucket        = local.artifact_bucket_name
  force_destroy = false

  lifecycle {
    prevent_destroy = true
  }

  tags = local.common_tags
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    bucket_key_enabled = true
    apply_server_side_encryption_by_default {
      kms_master_key_id = aws_kms_key.platform.arn
      sse_algorithm     = "aws:kms"
    }
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts_tls" {
  bucket = aws_s3_bucket.artifacts.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Sid       = "DenyInsecureTransport"
      Effect    = "Deny"
      Principal = "*"
      Action    = "s3:*"
      Resource = [
        aws_s3_bucket.artifacts.arn,
        "${aws_s3_bucket.artifacts.arn}/*"
      ]
      Condition = {
        Bool = { "aws:SecureTransport" = "false" }
      }
    }]
  })
}

resource "aws_ecr_repository" "platform" {
  for_each = toset(["management", "driver", "composed"])

  name                 = "${local.name}/${each.key}"
  image_tag_mutability = "IMMUTABLE"
  force_delete         = false

  encryption_configuration {
    encryption_type = "KMS"
    kms_key         = aws_kms_key.platform.arn
  }

  image_scanning_configuration {
    scan_on_push = true
  }

  tags = merge(local.common_tags, {
    "frontier:component" = each.key
  })
}

resource "aws_ecr_lifecycle_policy" "platform" {
  for_each   = aws_ecr_repository.platform
  repository = each.value.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Expire untagged images after 14 days"
      selection = {
        tagStatus   = "untagged"
        countType   = "sinceImagePushed"
        countUnit   = "days"
        countNumber = 14
      }
      action = { type = "expire" }
    }]
  })
}
