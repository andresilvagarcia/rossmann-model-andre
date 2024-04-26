import pickle
import os
import pandas as pd
from flask import Flask, request, Response
from rossmann.Rossmann import Rossmann

# Loading Model
model = pickle.load(open('model/model_rossmann.pkl', 'rb'))

# Initialize API
app = Flask(__name__)

@app.route('/rossmann/predict', methods=['POST'])

def rossmann_predict():
    test_json = request.get_json()
    if test_json: # There is data
        if isinstance(test_json, dict): # Unique Example
            teste_raw = pd.DataFrame(test_json, index=[0])
        else: # Multiple Examples
            teste_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
    else:
        return Response('{}', status=200, mimetype='application/json')
    
    # Instantiate Rossman Class
    pipeline = Rossmann()

    # Data Cleaning
    df1 = pipeline.data_cleaning(teste_raw)

    # Feature Engineering
    df2 = pipeline.feature_engineering(df1)

    # Data Preparation
    df3 = pipeline.data_preparation(df2)

    # Prediction
    df_response = pipeline.get_prediction(model, teste_raw, df3)
    return df_response

if __name__ == '__main__':
    port = os.environ.get('PORT', 5000)
    app.run(host='0.0.0.0', port=port)