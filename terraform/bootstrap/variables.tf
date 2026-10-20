variable "aws_region" {
  description = "AWS region that stores this site's Terraform state."
  type        = string

  validation {
    condition     = can(regex("^[a-z]{2}(-[a-z]+)+-[0-9]$", var.aws_region))
    error_message = "aws_region must be a canonical AWS region name."
  }
}

variable "expected_account_id" {
  description = "Exact AWS account allowed to receive this site."
  type        = string

  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_account_id))
    error_message = "expected_account_id must contain exactly 12 digits."
  }
}

variable "site_name" {
  description = "Stable lowercase site identifier."
  type        = string

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,22}[a-z0-9]$", var.site_name))
    error_message = "site_name must be a 3-24 character lowercase slug."
  }
}

variable "state_bucket_name" {
  description = "Globally unique bucket used only for Terraform state."
  type        = string

  validation {
    condition     = can(regex("^[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]$", var.state_bucket_name))
    error_message = "state_bucket_name must be a valid lowercase S3 bucket name."
  }
}

variable "tags" {
  description = "Non-secret ownership tags applied to bootstrap resources."
  type        = map(string)
}

