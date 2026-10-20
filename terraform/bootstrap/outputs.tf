output "account_id" {
  value = data.aws_caller_identity.current.account_id
}

output "state_bucket" {
  value = aws_s3_bucket.terraform_state.id
}

output "state_bucket_arn" {
  value = aws_s3_bucket.terraform_state.arn
}

