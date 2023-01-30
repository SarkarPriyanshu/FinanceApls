import random
import requests

from app.routes.train_pipeline import train_pipeline

def test_predict(sample_input__test_data):

  target,test_data = sample_input__test_data
  index = random.randrange(0, test_data.shape[0])  

  #inputs 
  age = test_data['age'][index]
  sex = test_data['sex'][index]
  bmi = test_data['bmi'][index]
  children = test_data['children'][index]
  smoker = test_data['smoker'][index]
  region = test_data['region'][index]

  response = requests.put(f'http://127.0.0.1:8080/predict?age={age}&sex={sex}&bmi={bmi}&children={children}&smoker={smoker}&region={region}')

  assert isinstance(response,object)


def test_train_pipeline():

  response = train_pipeline()

  assert isinstance(response,object)


