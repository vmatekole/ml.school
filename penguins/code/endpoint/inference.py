
from model import PenguinModel

import os
import json
import boto3
from pathlib import Path
import numpy as np
import pandas as pd

from pickle import load

s3 = boto3.resource("s3")

import torch
import torch.nn as nn
import torch.optim as optim

BUCKET = "vmate-mlschool4"
def _get_pipeline(directory=None):
    """
    Returns the Scikit-Learn pipeline used to transform the dataset.
    """
    try:
        if(directory is None):
            directory = os.environ.get("PREPROCESSING_DIR", "/tmp")
        pipeline_directory = Path(directory) / "pipeline"
        pipeline_file = pipeline_directory / "pipeline.pkl"
        
        if(not pipeline_file.exists()):
            pipeline_directory.mkdir(parents=True, exist_ok=True)
            _download(pipeline_file)
        
        return load(open(pipeline_file, 'rb'))
    
    except Exception as e:
        print(f'#lkj Exception: {e}')
    
def _get_class(prediction, directory):
    """
    Returns the class name of a given prediction. 
    """
    try:
        if(directory is None):
            directory = os.environ.get("CLASSES_DIR", "/tmp")
        
        classes_directory = Path(directory) / "pipeline"
        classes_file = classes_directory / "classes.csv"
        
        if(not classes_file.exists()):
            classes_directory.mkdir(parents=True, exist_ok=True)
            _download(classes_file)
        
        with open(classes_file) as f:
            file = f.readlines()

        classes = list(map(lambda x: x.replace("'", ""), file[0].split(',')))
        return classes[prediction]
    except Exception as e:
        print(f'#lku8 Exception: {e}')

def _download(file):    
    try:
        if(file.exists()):
            return

        s3_uri = os.environ.get("S3_LOCATION", f"s3://{BUCKET}/penguins/preprocessing")

        s3_parts = s3_uri.split('/', 3)
        bucket = s3_parts[2]
        key = s3_parts[3]

        s3.Bucket(bucket).download_file(f"{key}/{file.name}", str(file))
    except Exception as e:
        print(f'#kljkl Exception: {e}')
   
def _process_probabilities(probabilities):
    """
        Returns class and probability
    """
    prediction = np.argmax(probabilities)    
    confidence = probabilities[prediction]
    return prediction, confidence
    
def _process_prediction(input_data, directory):
    """
        Return prediction in JSON format and groundtruth(undesirable hack to generate baseline stats and data)
    """ 
    prediction, confidence = input_data
    species = _get_class(prediction, directory)
    result =  {
        "species": species,
        "prediction": int(prediction),
        "confidence": confidence.item()
    } 

    return result
    
def model_fn(model_dir):
    """
        Loads Pytorch model
    """
    input_shape = 7
    
    model = PenguinModel(input_shape=input_shape)    
    try:
        with open(Path(model_dir) / '001' / 'model.pth', 'rb') as f:
            model.load_state_dict(torch.load(f))

    except Exception as e:
        print(f"#kj9 Exception: {e}")
         
    return model
    
def input_fn(request_body, request_content_type, directory=None):
    
    print(f"Processing input data...{request_body}")
    try:
        if request_content_type in ("application/json", "application/octet-stream"):
            # When the endpoint is running, we will receive a context
            # object. We need to parse the input and turn it into 
            # JSON in that case.
            endpoint_input = json.loads(request_body)
            if isinstance(endpoint_input, dict):
                endpoint_input = [endpoint_input]                
            if endpoint_input is None:
                raise ValueError("There was an error parsing the input request.")
        else:
            raise ValueError(f"Unsupported content type: {request_content_type or 'unknown'}")
 
        pipeline = _get_pipeline(directory)
        transformed_data = []
        
        for data in endpoint_input:
            data.pop("species", None)
            df = pd.json_normalize(data)
            result = pipeline.transform(df)
            tensor = torch.tensor(result, dtype=torch.float32)
            transformed_data.append(tensor)           
        return transformed_data
    except Exception as e:
        print(f"#k88jj Exception: {e}")
    
def predict_fn(input_data_list, model):
    print(f"Sending input data to model to make a prediction...")
    results = []
    for data in input_data_list:
        tensor = data
        with torch.no_grad():
            model.eval()
            out = model.predict(tensor)
        results.append(out)
    return results
    
def output_fn(output_data, directory=None, accept="application/json"):
    print("Processing prediction received from the model...")
    prediction_list = []
    for output in output_data:
        predictions = output
        result = _process_probabilities(predictions[0])
        prediction = _process_prediction(result,directory)
        
        prediction_list.append(prediction)
    
    if accept == 'application/json':
        return json.dumps(prediction_list), accept
    
    raise Exception(f'Requested unsupported ContentType in Accept:{accept}')
