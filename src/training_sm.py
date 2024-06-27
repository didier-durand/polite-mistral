import json

import boto3
from botocore.response import StreamingBody
from sagemaker import hyperparameters
from sagemaker.huggingface import get_huggingface_llm_image_uri, HuggingFaceModel
from sagemaker.jumpstart.estimator import JumpStartEstimator

from bedrock import GenAiModel
from iam import SM_ROLE
from util import to_json, check_response

AWS_LLM_TRAINING = "aws-llm-training"

INSTANCE_TYPE_G5_8X = "ml.g5.8xlarge"  # 1 GPU  - 128 GB
INSTANCE_TYPE_G5_16X = "ml.g5.16xlarge"  # 1 GPU  - 256 GB
INSTANCE_TYPE_G5_12X = "ml.g5.12xlarge"  # 4 GPUs - 192 GB
INSTANCE_TYPE_G5_24X = "ml.g5.24xlarge"  # 4 GPUs - 384 GB


def train_model(training_s3loc: str = "", output_s3loc: str = ""):
    model = GenAiModel.HF_MISTRAL_7B
    # model = GenAiModel.HF_GEMMA_7B
    #
    model_id = model.value
    model_version = "*"
    model_hyperparameters = hyperparameters.retrieve_default(
        model_id=model_id,
        model_version=model_version
    )
    match model:
        case GenAiModel.HF_MISTRAL_7B:
            model_hyperparameters["epoch"] = "1"
            model_hyperparameters["per_device_train_batch_size"] = "2"
            model_hyperparameters["gradient_accumulation_steps"] = "2"
            model_hyperparameters["instruction_tuned"] = "True"
            instance_type = INSTANCE_TYPE_G5_24X
            model_environment = None
        case GenAiModel.HF_GEMMA_7B:
            model_hyperparameters["epoch"] = "1"
            model_environment = {
                "accept_eula": "true"
            }
            instance_type = INSTANCE_TYPE_G5_12X
        case _:
            raise ValueError("unknown model: " + model_id)
    print("--- hyperparameters:", to_json(model_hyperparameters))
    hyperparameters.validate(
        model_id=model_id,
        model_version=model_version,
        hyperparameters=model_hyperparameters
    )
    estimator = JumpStartEstimator(
        model_id=model_id,
        hyperparameters=model_hyperparameters,
        instance_type=instance_type,
        output_path=output_s3loc,
        environment=model_environment,
        role=SM_ROLE,

    )
    print("--- estimator image uri:", estimator.image_uri)
    estimator.fit(inputs=training_s3loc, logs="All", wait=True)


def deploy_model():
    llm_image = get_huggingface_llm_image_uri("huggingface")
    print(llm_image)
    instance_type = INSTANCE_TYPE_G5_8X
    number_of_gpu = 1
    health_check_timeout = 300
    config = {
        'HF_MODEL_ID': "/opt/ml/model",  # path to where sagemaker stores the model
        'SM_NUM_GPUS': json.dumps(number_of_gpu),  # Number of GPU used per replica
        'MAX_INPUT_LENGTH': json.dumps(1024),  # Max length of input text
        'MAX_TOTAL_TOKENS': json.dumps(2048),  # Max length of the generation (including input text)
    }

    llm_model = HuggingFaceModel(
        role=SM_ROLE,
        image_uri=llm_image,
        model_data=to_json({'S3DataSource': {
            'S3Uri': "s3://aws-llm-training/job_output/hf-llm-mistral-7b-2024-06-19-12-27-18-715/output/model/",
            'S3DataType': 'S3Prefix', 'CompressionType': 'None'}}),
        env=config
    )
    response = llm_model.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        container_startup_health_check_timeout=health_check_timeout,
    )
    print(to_json(response))


def access_model(endpoint_name: str, max_tokens: int = 32_768):
    sm_client = boto3.client("sagemaker")
    response = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print(to_json(response))
    check_response(response)
    assert "InService" == response["EndpointStatus"]
    body = to_json({
        "inputs": "What is a Large Language Model ?",
        "max_tokens": max_tokens
    })
    response = (boto3.client("sagemaker-runtime")
                .invoke_endpoint(EndpointName=endpoint_name,
                                 ContentType="application/json",
                                 Body=body))
    print(to_json(response))
    response_body: StreamingBody = response["Body"]
    print(to_json(response_body.read()))
