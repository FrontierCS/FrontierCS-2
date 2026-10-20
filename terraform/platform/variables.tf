variable "aws_region" {
  type        = string
  description = "AWS region for the FrontierCS2 site."

  validation {
    condition     = can(regex("^[a-z]{2}(-[a-z]+)+-[0-9]$", var.aws_region))
    error_message = "aws_region must be a canonical AWS region name."
  }
}

variable "expected_account_id" {
  type        = string
  description = "Exact AWS account allowed to receive this site."

  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_account_id))
    error_message = "expected_account_id must contain exactly 12 digits."
  }
}

variable "site_name" {
  type        = string
  description = "Stable lowercase site identifier."

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,22}[a-z0-9]$", var.site_name))
    error_message = "site_name must be a 3-24 character lowercase slug."
  }
}

variable "kubernetes_version" {
  type        = string
  description = "Pinned EKS Kubernetes version."

  validation {
    condition     = var.kubernetes_version == "1.34"
    error_message = "This release supports only Kubernetes 1.34."
  }
}

variable "vpc_cidr" {
  type        = string
  description = "Private /16 CIDR for the dedicated VPC."

  validation {
    condition     = can(cidrhost(var.vpc_cidr, 0)) && endswith(var.vpc_cidr, "/16")
    error_message = "vpc_cidr must be a valid private IPv4 /16."
  }
}

variable "service_ipv4_cidr" {
  type        = string
  description = "Dedicated Kubernetes Service IPv4 /16, disjoint from the VPC."

  validation {
    condition     = can(cidrhost(var.service_ipv4_cidr, 0)) && endswith(var.service_ipv4_cidr, "/16")
    error_message = "service_ipv4_cidr must be a valid IPv4 /16."
  }
}

variable "local_registry_endpoint" {
  type        = string
  description = "Fixed ClusterIP:5000 used by the explicit executable-preview registry."

  validation {
    condition     = can(regex("^[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+:5000$", var.local_registry_endpoint))
    error_message = "local_registry_endpoint must be an IPv4 address on port 5000."
  }
}

variable "availability_zone_count" {
  type        = number
  description = "Number of Availability Zones used by the site."

  validation {
    condition     = contains([2, 3], var.availability_zone_count)
    error_message = "availability_zone_count must be 2 or 3."
  }
}

variable "operator_cidrs" {
  type        = list(string)
  description = "Exact operator IPv4 /32 CIDRs allowed to reach the public EKS endpoint."

  validation {
    condition = (
      length(var.operator_cidrs) > 0 &&
      alltrue([for cidr in var.operator_cidrs : can(cidrhost(cidr, 0)) && endswith(cidr, "/32")]) &&
      !contains(var.operator_cidrs, "0.0.0.0/0")
    )
    error_message = "operator_cidrs must contain only exact IPv4 /32 values."
  }
}

variable "operator_principals" {
  type        = list(string)
  description = "Additional IAM user or role ARNs granted cluster administrator access."
  default     = []
  nullable    = false

  validation {
    condition = (
      length(distinct(var.operator_principals)) == length(var.operator_principals) &&
      alltrue([for arn in var.operator_principals : can(regex("^arn:aws(-cn|-us-gov)?:iam::[0-9]{12}:(user|role)/[A-Za-z0-9+=,.@_-]+(/[A-Za-z0-9+=,.@_-]+)*$", arn))])
    )
    error_message = "operator_principals must contain unique IAM user or role ARNs."
  }
}

variable "single_nat_gateway" {
  type        = bool
  description = "Use one NAT gateway for the MVP site."

  validation {
    condition     = var.single_nat_gateway
    error_message = "v0.1 requires one NAT gateway for private-node egress."
  }
}

variable "system_node_group" {
  type = object({
    instance_types  = list(string)
    min_size        = number
    desired_size    = number
    max_size        = number
    root_volume_gib = number
  })
}

variable "cpu_node_group" {
  type = object({
    instance_types  = list(string)
    min_size        = number
    desired_size    = number
    max_size        = number
    root_volume_gib = number
  })
}

variable "tags" {
  type        = map(string)
  description = "Non-secret ownership tags applied to every supported resource."
}
variable "manual_capacity_nodegroup_arns" {
  description = "Opt-in exact two worker group ARNs for trusted Management manual capacity controls."
  type        = list(string)
  default     = []
  validation {
    condition = length(var.manual_capacity_nodegroup_arns) == 0 || (
      length(distinct(var.manual_capacity_nodegroup_arns)) == 2 &&
      alltrue([for arn in var.manual_capacity_nodegroup_arns : can(regex("^arn:aws:eks:[a-z0-9-]+:[0-9]{12}:nodegroup/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+/[A-Za-z0-9_-]+$", arn))])
    )
    error_message = "Manual capacity requires exactly two distinct, non-wildcard nodegroup ARNs."
  }
}
