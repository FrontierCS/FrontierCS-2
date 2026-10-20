output "account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "aws_region" {
  value = var.aws_region
}

output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "vpc_id" {
  value = module.vpc.vpc_id
}

output "vpc_cidr" {
  value = module.vpc.vpc_cidr_block
}

output "service_ipv4_cidr" {
  value = var.service_ipv4_cidr
}

output "private_subnet_ids" {
  value = module.vpc.private_subnets
}

output "nat_gateway_public_ips" {
  value = module.vpc.nat_public_ips
}

output "platform_kms_key_arn" {
  value = aws_kms_key.platform.arn
}

output "artifact_bucket" {
  value = aws_s3_bucket.artifacts.id
}

output "management_artifact_reader_role_arn" {
  value = aws_iam_role.management_artifact_reader.arn
}

output "management_artifact_reader_role_id" {
  value = aws_iam_role.management_artifact_reader.unique_id
}

output "ecr_repositories" {
  value = { for name, repository in aws_ecr_repository.platform : name => repository.repository_url }
}

output "system_node_group_name" {
  value = split(":", module.eks.eks_managed_node_groups["system"].node_group_id)[1]
}

output "cpu_node_group_name" {
  value = split(":", module.eks.eks_managed_node_groups["cpu"].node_group_id)[1]
}

output "efs_file_system_id" {
  value = aws_efs_file_system.preview.id
}

output "efs_object_store_access_point_id" {
  value = aws_efs_access_point.object_store.id
}

output "efs_registry_access_point_id" {
  value = aws_efs_access_point.registry.id
}
