output "s3_root" {
  description = "Value for SOCCER_S3_ROOT"
  value       = "s3://${aws_s3_bucket.data.bucket}"
}

output "kms_key_arn" {
  value = aws_kms_key.data.arn
}

output "gpu_server_user" {
  description = "IAM user for the GPU server; create its access key by hand"
  value       = aws_iam_user.gpu_server.name
}

output "transfer_lambda" {
  value = var.enable_transfer_lambda ? aws_lambda_function.transfer[0].function_name : null
}
