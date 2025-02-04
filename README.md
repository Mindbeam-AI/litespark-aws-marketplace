# Mindbeam Lepton LLM Pre-training
Mindbeam's Lepton is a quantum-inspired algorithm that offers accelerated pre-training of large anguage models. This repository shows how to use the algorithm from AWS marketplace to train a model using data from Hugging Face.

## Prerequisites
To use Lepton LLM, make sure that
- Your IAM role has AmazonSageMakerFullAccess
- Either your AWS account has a subscription to `Lepton - LLM Pre-Training v1.1`.
- Or your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used:
  a. aws-marketplace:ViewSubscriptions
  b. aws-marketplace:Unsubscribe
  c. aws-marketplace:Subscribe
