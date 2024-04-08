from src.constants import DATASETS_PATH, DATASET_RAW_FILE
from src.model import DiamondPricePredictor
from src.dataset import DiamondsDataset, load_csv_data
from flask import Flask, request
from dataclasses import dataclass

import json
import pandas as pd
import os

app = Flask(__name__)

dataset = DiamondsDataset.from_csv(DATASET_RAW_FILE)
model = DiamondPricePredictor()
model.fit(dataset)


@dataclass
class DiamondSample:
    carat: float
    cut: str
    color: str
    clarity: str
    depth: float
    table: float
    x: float
    y: float
    z: float
    
@dataclass
class PricedDiamondSample(DiamondSample):
    price: float


@app.route('/datasets', methods=['GET'])
def datasets():
    if request.method == 'GET':
        return [csv for csv in os.listdir(DATASETS_PATH) if csv.endswith('.csv')]


@app.route('/datasets/<dataset_name>', methods=['GET', 'PUT', 'DELETE'])
def dataset(dataset_name):
    if not dataset_name.endswith('.csv'):
        dataset_name += '.csv'

    if request.method == 'GET':
        return load_csv_data(f'{DATASETS_PATH}{dataset_name}').to_dict(orient='records')
    
    elif request.method == 'PUT':
        body = request.get_json()

        try:
            samples = [PricedDiamondSample(**sample) for sample in body]
        except TypeError:
            return f'''Invalid dataset format, it should be a list of dictionaries
                       with the following keys: {list(DiamondSample.__annotations__.keys())
                       + list(PricedDiamondSample.__annotations__.keys())}''', 400

        pd.DataFrame(samples).to_csv(f'{DATASETS_PATH}{dataset_name}', index=False)
        return f'Dataset {dataset_name} created successfully'
    
    elif request.method == 'DELETE':
        os.remove(f'{DATASETS_PATH}{dataset_name}')
        return f'Dataset {dataset_name} deleted successfully'
    
    else:
        return 'Invalid method', 405


@app.route('/train', methods=['POST'])
def train():
    if request.method == 'POST':
        dataset_names = request.get_json()
        
        if not isinstance(dataset_names, list):
            return 'You should provide a list of available dataset names to train the model with', 400
        
        available_datasets = os.listdir(DATASETS_PATH)
        if not all([dataset_name in available_datasets for dataset_name in dataset_names]):
            return f'''All dataset should be available in the {DATASETS_PATH} directory.
                       The following datasets are available: {available_datasets}''', 400
        
        dataset = DiamondsDataset.from_csv(*[f'{DATASETS_PATH}{name}' for name in dataset_names])
        
        global model
        model = DiamondPricePredictor.from_grid_search_cv(dataset)
        
        return model.evaluate(dataset)
    
    else:
        return 'Invalid method', 405


@app.route('/predict', methods=['GET'])
def predict():
    if request.method == 'GET':
        body = request.get_json()
        
        try:
            samples = [DiamondSample(**sample) for sample in body]
        except TypeError:
            return f'''Invalid sample format, it should be a list of dictionaries
                       with the following keys: {list(DiamondSample.__annotations__.keys())}''', 400
        
        dataset = DiamondsDataset(pd.DataFrame(samples))
        
        return model.predict(dataset).tolist()
    
    else:
        return 'Invalid method', 405
    
    
@app.route('/predict-explain', methods=['GET'])
def predict_explain():
    if request.method == 'GET':
        body = request.get_json()
        
        try:
            samples = [DiamondSample(**sample) for sample in body]
        except TypeError:
            return f'''Invalid sample format, it should be a list of dictionaries
                       with the following keys: {list(DiamondSample.__annotations__.keys())}''', 400
        
        dataset = DiamondsDataset(pd.DataFrame(samples))
        
        json_explainations = [{
            'predicted_price': x.predicted_price,
            'decision_steps': [vars(xx) for xx in x.decision_steps]
        } for x in model.predict_explain(dataset)]
        
        return json.dumps(json_explainations, default=int)
    
    else:
        return 'Invalid method', 405


if __name__ == '__main__':
    app.run()