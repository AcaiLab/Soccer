# Storage and access for the soccer pipeline.
#
# Layout in the bucket:
#   videos/<league>/<match>/*.mkv   Dropbox copy (written by the transfer)
#   json/<split>/...                annotation JSONs
#   frames/, features/, results/    pipeline outputs (written by the GPU server)
#
# Secrets are never in Terraform: the server's access key and the Dropbox SSM
# parameters are created by hand after `apply` (see docs/remote_gpu_server.md),
# because Terraform state stores values in plain text.

data "aws_caller_identity" "current" {}

locals {
  bucket_arn   = aws_s3_bucket.data.arn
  ssm_arn      = "arn:aws:ssm:${var.region}:${data.aws_caller_identity.current.account_id}:parameter${var.ssm_prefix}/dropbox/*"
  read_prefix  = ["videos/*", "json/*", "frames/*", "features/*", "results/*"]
  write_prefix = concat(["json/*", "frames/*", "features/*", "results/*"], var.server_can_transfer ? ["videos/*"] : [])
}

# --- Encryption key ----------------------------------------------------------

resource "aws_kms_key" "data" {
  description             = "${var.project} data bucket"
  enable_key_rotation     = true
  deletion_window_in_days = 30
}

resource "aws_kms_alias" "data" {
  name          = "alias/${var.project}-data"
  target_key_id = aws_kms_key.data.key_id
}

# --- Bucket --------------------------------------------------------------------

resource "aws_s3_bucket" "data" {
  bucket = var.bucket_name
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.data.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_lifecycle_configuration" "data" {
  bucket = aws_s3_bucket.data.id

  rule {
    id     = "abort-incomplete-uploads"
    status = "Enabled"
    filter {}
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}
    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_days
    }
  }
}

data "aws_iam_policy_document" "bucket" {
  statement {
    sid       = "DenyInsecureTransport"
    effect    = "Deny"
    actions   = ["s3:*"]
    resources = [local.bucket_arn, "${local.bucket_arn}/*"]
    principals {
      type        = "*"
      identifiers = ["*"]
    }
    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "data" {
  bucket     = aws_s3_bucket.data.id
  policy     = data.aws_iam_policy_document.bucket.json
  depends_on = [aws_s3_bucket_public_access_block.data]
}

# --- GPU server identity (least privilege) ------------------------------------
# An IAM user because the server is outside AWS. Create its access key by hand
# (aws iam create-access-key) and store it only on the server, mode 600.

resource "aws_iam_user" "gpu_server" {
  name = "${var.project}-gpu-server"
}

data "aws_iam_policy_document" "gpu_server" {
  # Unconditional: without bucket-level ListBucket, S3 answers 403 instead of
  # 404 for missing objects, which breaks the "already copied?" check. The
  # bucket holds only this project's data.
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]
  }
  statement {
    sid       = "ReadInputsAndOutputs"
    actions   = ["s3:GetObject"]
    resources = [for p in local.read_prefix : "${local.bucket_arn}/${p}"]
  }
  statement {
    sid       = "WriteOutputs"
    actions   = ["s3:PutObject", "s3:AbortMultipartUpload", "s3:ListMultipartUploadParts"]
    resources = [for p in local.write_prefix : "${local.bucket_arn}/${p}"]
  }
  dynamic "statement" {
    for_each = var.server_can_transfer ? [1] : []
    content {
      sid       = "ReplaceFailedCopies"
      actions   = ["s3:DeleteObject"]
      resources = ["${local.bucket_arn}/videos/*"]
    }
  }
  statement {
    sid       = "UseDataKey"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
  dynamic "statement" {
    for_each = var.server_can_transfer ? [1] : []
    content {
      sid       = "ReadDropboxSettings"
      actions   = ["ssm:GetParameter"]
      resources = [local.ssm_arn]
    }
  }
}

resource "aws_iam_user_policy" "gpu_server" {
  name   = "${var.project}-gpu-server"
  user   = aws_iam_user.gpu_server.name
  policy = data.aws_iam_policy_document.gpu_server.json
}

# --- Optional: Dropbox -> S3 transfer Lambda ----------------------------------
# One invocation copies one game (event: {"league", "match_id", "dest"}).
# Streams to S3, so it needs no /tmp space; a 600 MB half takes well under the
# 15-minute limit at Dropbox speeds.

data "aws_iam_policy_document" "lambda_assume" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "transfer" {
  count              = var.enable_transfer_lambda ? 1 : 0
  name               = "${var.project}-transfer"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume.json
}

data "aws_iam_policy_document" "transfer" {
  statement {
    sid       = "Logs"
    actions   = ["logs:CreateLogStream", "logs:PutLogEvents"]
    resources = ["arn:aws:logs:${var.region}:${data.aws_caller_identity.current.account_id}:log-group:/aws/lambda/${var.project}-transfer:*"]
  }
  statement {
    sid       = "ListBucket"
    actions   = ["s3:ListBucket"]
    resources = [local.bucket_arn]
  }
  statement {
    sid       = "WriteVideos"
    actions   = ["s3:PutObject", "s3:GetObject", "s3:DeleteObject", "s3:AbortMultipartUpload"]
    resources = ["${local.bucket_arn}/videos/*"]
  }
  statement {
    sid       = "UseDataKey"
    actions   = ["kms:Decrypt", "kms:GenerateDataKey"]
    resources = [aws_kms_key.data.arn]
  }
  statement {
    sid       = "ReadDropboxSettings"
    actions   = ["ssm:GetParameter"]
    resources = [local.ssm_arn]
  }
}

resource "aws_iam_role_policy" "transfer" {
  count  = var.enable_transfer_lambda ? 1 : 0
  name   = "${var.project}-transfer"
  role   = aws_iam_role.transfer[0].id
  policy = data.aws_iam_policy_document.transfer.json
}

resource "aws_cloudwatch_log_group" "transfer" {
  count             = var.enable_transfer_lambda ? 1 : 0
  name              = "/aws/lambda/${var.project}-transfer"
  retention_in_days = 30
}

resource "aws_lambda_function" "transfer" {
  count            = var.enable_transfer_lambda ? 1 : 0
  function_name    = "${var.project}-transfer"
  role             = aws_iam_role.transfer[0].arn
  runtime          = "python3.11"
  handler          = "soccer.transfer.handler"
  filename         = var.lambda_package
  source_code_hash = filebase64sha256(var.lambda_package)
  timeout          = 900
  memory_size      = 1024
  environment {
    variables = { SOCCER_SSM_PREFIX = var.ssm_prefix }
  }
  depends_on = [aws_cloudwatch_log_group.transfer]
}
