
import boto3
import sagemaker
from pathlib import Path


BUCKET = "vmate-mlschool3"
S3_LOCATION = f"s3://{BUCKET}/penguins"
DATA_FILEPATH = Path().resolve() / "data.csv"


sagemaker_client = boto3.client("sagemaker")
iam_client = boto3.client("iam")
role = sagemaker.get_execution_role()
region = boto3.Session().region_name
sagemaker_session = sagemaker.session.Session()
