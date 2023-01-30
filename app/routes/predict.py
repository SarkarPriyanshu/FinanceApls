import pandas as pd
import numpy as np

import logging
from fastapi import APIRouter
from http import HTTPStatus
from typing import Dict
from app.model.model import config

from app.processing.data_manager import load_pipeline


logger = logging.getLogger(__file__)
router = APIRouter()

@router.post('/predict', status_code=HTTPStatus.OK, tags=['predict'])
async def predict(age:int,sex:str,bmi:float,children:int,smoker:str,region:str) -> dict:
    finance_pipe = None
    data =  np.array([age,sex,bmi,children,smoker,region]).reshape(1,6)
    try:
        finance_pipe = load_pipeline(file_name=config.app_config.pipeline_save_file)
        logger.info(f'finance pipeline loaded successfully: {finance_pipe}')
        data = pd.DataFrame(data,columns=config.model_config.features)
    except Exception as e:
        logger.info(f'finance pipeline loaded unsuccessfully: {e}')
        return {'error':f'finance pipeline loaded unsuccessfully'}
    else:
        result = finance_pipe.predict(data.astype({'age':int,'bmi':float,'children':int}))
        logger.info(f'predict finance: {result[0]}')
        return {'Output': result[0]}