from fastapi import FastAPI,Request
from app.routes import predict,train_pipeline,upload_data

app = FastAPI()

app.include_router(predict.router)
app.include_router(train_pipeline.router)
# app.include_router(upload_data.router)

@app.get('/')
async def index():
    return {'data':'Welcome friends from future.'}

@app.get('/about')
async def about():
    return {'data':f'Hello, this is finance predicting application.'}