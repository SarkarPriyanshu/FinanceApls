import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from app.model.model import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    
    if file_name == 'test.csv':
        dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
        target = dataframe[config.model_config.target]
        test_data = dataframe[config.model_config.features]
        return target,test_data        

    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))    
    return dataframe.drop('Unnamed: 0',axis=1)

def remove_old_dataset(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old train.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for data_file in DATASET_DIR.iterdir():
        if data_file.name not in do_not_delete:
            data_file.unlink() 


def load_new_dataset(*, data_file: pd.DataFrame) -> dict:
    try:
        old_data = load_dataset(file_name=config.app_config.training_data_file) 
        train = pd.concat([old_data, data_file], ignore_index=True, sort=False)
        
        
        train = train.drop_duplicates(keep='first')
        save_file_name = config.app_config.test_data_file
        save_path = DATASET_DIR / save_file_name
        
        remove_old_dataset(files_to_keep=[save_file_name])
        train.to_csv(save_path,index=False)
    except Exception as e:
        return {'message':'unsuccessfully uploaded','result':False,'error':e}
    else:
         return {'message':'successfully uploaded','result':True}      

def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""
    file_path = TRAINED_MODEL_DIR / f'{file_name}.pkl'
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()