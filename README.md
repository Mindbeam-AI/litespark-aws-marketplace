# Mindbeam Lepton LLM Pre-training
Mindbeam's Lepton is a quantum-inspired algorithm that offers accelerated pre-training of large anguage models. This repository shows how to use the algorithm from AWS marketplace to train a model using data from Hugging Face.

## Prerequisites
To use Lepton LLM Pre-training, make sure that
- Your IAM role has *AmazonSageMakerFullAccess*.
- Either your AWS account has a subscription to [Lepton - LLM Pre-Training v1.2](https://aws.amazon.com/marketplace/pp/prodview-pmofsct5z6s6s).
- Or your IAM role has these three permissions and you have authority to make AWS Marketplace subscriptions in the AWS account used:
  - *aws-marketplace:ViewSubscriptions*
  - *aws-marketplace:Unsubscribe*
  - *aws-marketplace:Subscribe*

## Create a training job
- Log in to your AWS console.
- Go to Amazon SageMaker AI > Training > Algorithms.
- Go to AWS Marketplace subscriptions.
- Select Lepton - LLM Pre-Training v1.2, click on Actions > Create training job.
- In Job settings, give a Job name, select an appropriate IAM role that has the *AmazonSageMakerFullAccess* IAM policy attached (e.g. SageMaker-SageMakerOps).
- In Algorithm source, select "An algorithm subscription from AWS Marketplace".
- Under "Choose an algorithm subscription", select Lepton - LLM Pre-Training v1.1.
- Under "Resource configuration", select either `ml.p4de.24xlarge` or `ml.p5.48xlarge` as instance type, instance count 1, and 1 GB additional storage volume per instance.
- Optionally, select a "Keep alive period" in between 1 minutes and 60 minutes.
- Under "Encryption key", select "No Custom Encryption".
- Select an appropriate "Stopping condition" that does not prematurely stop the run.
- Under "Network", check "Enable network isolation".

### Hyperparameters
- GLOBAL_BATCH_SIZE: the total number of samples processed before a model update, default is 512.
- MICRO_BATCH_SIZE: the number of samples processed in a single forward/backward pass on each GPU, default is 8.
- LEARNING_RATE: maximum learning rate with a cosine scheduler, default is 1.6e-3.
- MAX_STEP: maximum number of steps for pre-training, default is 96180 or 2 epochs.
- WARMUP_STEPS: number of steps required to reach the maximum learning rate, default is 500.
- WEIGHT_DECAY: adds a penalty term to the loss function that reduces large weights, default is 0.01.
- BETA1: first moment for the AdamW optimizer, default is 0.85.
- BETA2: second moment for the AdamW optimizer, default is 0.95.
- GRAD_CLIP: clip gradients above the threshold, default is 1.0.
- MIN_LR: minimum learning rate at the end of training, default is 4e-4.
- MODEL_NAME: default model is "mindbeam-2k". Keep this model for now, additional models will be available in future releases.
- SAVE_STEP_INTERVAL: number of steps after which checkpoint is saved periodically.
- EVAL_STEP_INTERVAL: number of steps after which model is validated.

### Data preparation
Any LLM pre-training dataset from Hugging Face can be prepared for training with Lepton using the accompanying notebook [dataprep.ipynb](https://github.com/Mindbeam-AI/lepton-aws-marketplace/blob/main/dataprep.ipynb). We recommend the [Expository-Prose-V1](https://huggingface.co/datasets/pints-ai/Expository-Prose-V1) dataset. The notebook downloads the raw dataset from Hugging Face to a local directory, prepares the dataset with appropriate tokenization, and uploads the prepared data to your specified S3 bucket.

### Data configuration
- Under Input data configuration, the default is only one channel named "train".
- Choose "Data source" as S3, "S3 data type" as S3Prefix, "S3 data distribution type" as FullyReplicated, and provide "S3 location" to the prepared data in your S3 bucket.
- Optionally, provide S3 location or a local directory for saving intermediate checkpoints. The algorithm generates checkpoints at every 4000 steps.
- Provide a "S3 output path" to save the final checkpoint.

After the above steps are completed, click on "Create training job". 

## Monitor a training job
- After a training job is created, you can view the progress at Amazon SageMaker AI > Training jobs.
- Click on the job name to view the details.
- View metrics in the Monitor section.
- To view logs in Cloud Watch, go to Monitor > View logs. In the new page, go to Log streams, and select the appropriate log stream.
- Checkpoints and outputs will be saved in your specified S3 bucket.
