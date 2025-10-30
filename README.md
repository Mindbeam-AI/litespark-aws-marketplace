# Mindbeam Litespark LLM Pre-training
Mindbeam's Litespark is an advanced algorithm that offers accelerated pre-training of large language models. This repository shows how to use the algorithm from AWS marketplace to train a model using custom datasets hosted in user's S3 bucket or local EFS/FSx filesystem.

## Prerequisites
To use Litespark LLM Pre-training, make sure that
- Your IAM role has *AmazonSageMakerFullAccess*, *AmazonS3FullAccess*, *CloudWatchEventsFullAccess*, *AmazonElasticContainerRegistryPublicFullAccess*, *AmazonEC2ContainerRegistryFullAccess*. These are combined into the role [*SageMaker-SageMakerOps*](arn:aws:iam::975050170529:role/service-role/SageMaker-SageMakerOps).
- Either your AWS account has a subscription to [Litespark - LLM Pre-Training](https://aws.amazon.com/marketplace/pp/prodview-q76fxekgwhezc?sr=0-1&ref_=beagle&applicationId=AWSMPContessa).
- Or your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used:
  - *aws-marketplace:ViewSubscriptions*
  - *aws-marketplace:Unsubscribe*
  - *aws-marketplace:Subscribe*

## Create a training job
- Log in to your AWS console.
- Go to Amazon SageMaker AI > Training > Algorithms.
- Go to AWS Marketplace subscriptions.
- Select Litespark - LLM Pre-Training, click on Actions > Create training job.
- In Job settings, give a Job name, select an appropriate IAM role that has the *AmazonSageMakerFullAccess* IAM policy attached (e.g. *SageMaker-SageMakerOps*).
- In Algorithm source, select "An algorithm subscription from AWS Marketplace".
- Under "Choose an algorithm subscription", select Mindbeam: Pre-Training LLM Optimization Consultation.
- Under "Resource configuration", select either `ml.p4de.24xlarge` or `ml.p5.48xlarge` as instance type, instance count 1, and 1 GB additional storage volume per instance.
- Optionally, select a "Keep alive period" in between 1 minutes and 60 minutes.
- Under "Encryption key", select "No Custom Encryption".
- Select an appropriate "Stopping condition" that does not prematurely stop the run.
- Under "Network", check "Enable network isolation".

### Hyperparameters
- GLOBAL_BATCH_SIZE: the total number of samples processed before a model update, default is 256.
- MICRO_BATCH_SIZE: the number of samples processed in a single forward/backward pass on each GPU, default is 4.
- NUM_NODES: number of nodes, default is 1.
- HIDDEN_SIZE: parameter for llama model, default is 768.
- INTERMEDIATE_SIZE: parameter for llama model, default is 2072.
- N_HEADS: parameter for llama model, default is 12.
- N_KV_HEADS: parameter for llama model, default is 12.
- N_LAYERS: parameter for llama model, default is 12.
- SEQ_LEN: training sequence length, default is 4096.
- VOCAB_SIZE: vocabulary size, must be consistent with tokenizer and data preparation.
- TOKENIZER: default is mindbeam-ai/Litespark. To use custom tokenizers, host the tokenizer in your S3 bucket path `s3://<my-bucket>/<org-name>/<model-name>`. When creating the training job, under `Input data configuration` click on `Add channel`, then put channel name `tokenizer`, Data source select `S3`, S3 data type select `S3prefix`, S3 data distribution type select `FullyReplicated`, and then for S3 location put `s3://<my-bucket>/<org-name>/<model-name>`. 
- LEARNING_RATE: maximum learning rate with a cosine scheduler, default is 1.6e-3.
- MAX_STEPS: maximum number of steps for pre-training, default is 10000.
- WARMUP_STEPS: number of steps required to reach the maximum learning rate, default is 500.
- WEIGHT_DECAY: adds a penalty term to the loss function that reduces large weights, default is 0.01.
- BETA1: first moment for the AdamW optimizer, default is 0.9.
- BETA2: second moment for the AdamW optimizer, default is 0.95.
- GRAD_CLIP: clip gradients above the threshold, default is 1.0.
- MIN_LR: minimum learning rate at the end of training, default is 4e-4.
- CHECKPOINT_INTERVAL: number of steps after which checkpoint is saved periodically, default is 1000.

### Data preparation
Any LLM pre-training dataset from Hugging Face can be prepared for training with Litespark using the accompanying script [prepare_data.py](https://github.com/Mindbeam-AI/Litespark-aws-marketplace/blob/main/prepare_dataset/prepare_data.py). The scripts downloads the raw dataset from Hugging Face to a local directory and prepares the dataset with appropriate tokenization. To upload the prepared data to your specified S3 bucket, use the script [upload_to_s3.py](https://github.com/Mindbeam-AI/Litespark-aws-marketplace/blob/main/prepare_dataset/upload_to_s3.py).

### Data configuration
- Under Input data configuration, the default is only one channel named "train".
- Choose "Data source" as S3, "S3 data type" as S3Prefix, "S3 data distribution type" as FullyReplicated, and provide "S3 location" to the prepared data in your S3 bucket.
- Optionally, provide S3 location or a local directory for saving intermediate checkpoints. The algorithm generates checkpoints at every 4000 steps.
- Provide a "S3 output path" to save the final checkpoint.

After the above steps are completed, click on "Create training job". 

## Monitor a training job
- After a training job is created, you can view the progress at Amazon SageMaker AI > Training jobs.
- Click on the job name to view the details.
- View metrics in the Monitor section. The following metrics are shown: CPU Utilization, GPU Utilization, Memory Utilization, Disk Utilization, GPU Memory Utilization, batches_per_sec (per GPU), flops_per_sec (per GPU), (training) loss, samples_per_sec (per GPU), mfu (model flops utilization per GPU), tokens_per_sec (per GPU).
- To view logs in Cloud Watch, go to Monitor > View logs. In the new page, go to Log streams, and select the appropriate log stream.
- Checkpoints and outputs will be saved in your specified S3 bucket as safetensors.
