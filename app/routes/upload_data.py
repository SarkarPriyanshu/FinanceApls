import pandas as pd
from fastapi import APIRouter,UploadFile,File
from http import HTTPStatus
from typing import Dict

from app.processing.data_manager import load_new_dataset

router = APIRouter()

@router.post('/upload_data', status_code=HTTPStatus.OK, tags=['upload_data'])
async def upload_data(data_file: UploadFile = File(...)) -> dict:
    """upload new data for model training."""
    df = pd.read_csv(data_file.file)

    data_file.file.close()
    
    result = load_new_dataset(data_file=df)
    if result['result']:
        return {'message': f'New data uploaded successfully :)'}
    else:
        return {'message': f'failed to upload data :(, {result["error"]}'}  
    