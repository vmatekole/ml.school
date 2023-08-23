
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from pickle import dump
from typing import Tuple


DEFAULT_BASE_DIR = Path('/opt')/'ml'/'processing'

def _preprocess_pipeline(df_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:    
    num_classes = 10

    categorical_transformer = Pipeline(
        steps=[
            ('encoder', OneHotEncoder())            
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('labels', categorical_transformer, ['label'])
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor)
        ]
    )
    data: np.ndarray = pipeline.fit_transform(df_data)
    # OneHotEncoded
    y_data: np.ndarray = data[:, :num_classes]
    # Drop OneHotEncoded target variable
    data = np.delete(data,np.arange(num_classes), axis=1)

    X_train, X_test_validation, y_data, y_test_validation = train_test_split(data, y_data, test_size=0.2, random_state=7)
    X_test, X_validation, y_test, y_validation = train_test_split(X_test_validation, y_test_validation, test_size=0.5, random_state=7)

    return X_train, X_test, X_validation, y_data, y_test, y_validation


def _save_pipeline(base_dir: str, pipeline: Pipeline):  
    
    pipeline_path = Path(base_dir)
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / 'pipeline.pkl', 'wb'))

def preprocess(base_dir = None, data_filepath =  DEFAULT_BASE_DIR):
    
    if base_dir is None:
        base_dir = DEFAULT_BASE_DIR
        
    base_dir = Path(base_dir)
    (base_dir / 'train').mkdir(parents=True, exist_ok=True)
    (base_dir / 'validation').mkdir(parents=True, exist_ok=True)
    (base_dir / 'test').mkdir(parents=True, exist_ok=True)
    (base_dir / 'labels').mkdir(parents=True, exist_ok=True)
        
    df_data: pd.DataFrame =  pd.read_csv(Path(data_filepath) / 'mnist_train.csv')

    df_test: pd.DataFrame =  pd.read_csv(Path(data_filepath) / 'mnist_test.csv')

    df_data: np.ndarray = pd.concat([df_data, df_test], axis=0)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(df_data['label'])

    X_train, X_test, X_validation, y_train, y_test, y_validation  = _preprocess_pipeline(df_data)    

    np.savetxt(base_dir / 'train' / 'mnist_train.csv', np.concatenate([X_train, y_train], axis=1), delimiter=',')
    np.savetxt(base_dir / 'test' / 'mnist_test.csv', np.concatenate([X_test, y_test], axis=1), delimiter=',')
    np.savetxt(base_dir / 'validation' / 'mnist_validation.csv', np.concatenate([X_validation, y_validation], axis=1), delimiter=',')
    np.savetxt(base_dir / 'labels' / 'labels.csv', labels, delimiter=',')

if __name__ == "__main__":
    preprocess(
        base_dir=DEFAULT_BASE_DIR,
    )
