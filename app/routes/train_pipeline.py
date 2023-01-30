import logging
from fastapi import APIRouter
from http import HTTPStatus
from typing import Dict

from sklearn.model_selection import train_test_split

from app.model.model import config
from app.processing.pipeline import finance_pipe
from app.processing.data_manager import load_dataset,save_pipeline

logger = logging.getLogger(__file__)
router = APIRouter()

@router.post('/train_pipeline', status_code=HTTPStatus.OK, tags=['train_pipeline'])
async def train_pipeline():
    """Train the model."""

    # read training data
    data = load_dataset(file_name=config.app_config.training_data_file)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.model_config.features],  # predictors
        data[config.model_config.target],
        test_size=config.model_config.test_size,
        # we are setting the random seed here
        # for reproducibility
        random_state=config.model_config.random_state,
    )


    try:
        # fit model
        finance_pipe.fit(X_train, y_train)

        # persist trained model
        save_pipeline(pipeline_to_persist=finance_pipe)
        logger.info(f'Training model succesfully')
        
        return {'message': 'Model has been trained succesfully :)'}

    except Exception as e:
        logger.info(f'Opps ! : {e}')
        
        return {'message': f'Model failed to train :( due to {e}'}
    