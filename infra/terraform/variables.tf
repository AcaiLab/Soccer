variable "region" {
  description = "AWS region for the bucket and any Lambda"
  type        = string
}

variable "aws_profile" {
  description = "Local AWS CLI profile used to apply this configuration"
  type        = string
}

variable "project" {
  description = "Name prefix for every resource"
  type        = string
  default     = "soccer-vision"
}

variable "bucket_name" {
  description = "Globally unique S3 bucket name for videos, annotations, frames, features and results"
  type        = string
}

variable "ssm_prefix" {
  description = "SSM Parameter Store path holding the Dropbox settings (created by hand, never in Terraform state)"
  type        = string
  default     = "/soccer"
}

variable "server_can_transfer" {
  description = "Let the GPU server identity write videos/ (needed if the Dropbox -> S3 copy runs on the server)"
  type        = bool
  default     = true
}

variable "enable_transfer_lambda" {
  description = "Create the Lambda that copies one game from Dropbox to S3 per invocation"
  type        = bool
  default     = false
}

variable "lambda_package" {
  description = "Path of the Lambda zip built by infra/build_lambda.sh"
  type        = string
  default     = "../build/transfer_lambda.zip"
}

variable "noncurrent_version_days" {
  description = "Days to keep overwritten or deleted object versions"
  type        = number
  default     = 30
}
