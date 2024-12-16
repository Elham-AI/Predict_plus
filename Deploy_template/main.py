from fastapi import FastAPI,Request
import pandas as pd
import pickle
from autoML import Module
app = FastAPI()
@app.post("/<user>/<model_name>/predict")
async def predict(request:Request):      
    """
    data shape is 
    {
        "data":[
                {<data record as json object>},
                {<data record as json object>},
                ...
        ]
    } 
    """
    data = await request.json()
    data = data['data']
    data = pd.DataFrame(data)
    with open("<model_id>_tuner.pkl",'rb') as f: 
        tuner = pickle.load(f)
    with open("<model_id>_model.pkl",'rb') as f: 
        model = pickle.load(f)
    tuner.model = model
    dates_cols = tuner.features_date
    num_features = tuner.features_num
    for col in dates_cols:
        data[col] = pd.to_datetime(data[col])
    missing = set(dates_cols+tuner.features_cat+tuner.features_bool).difference(set(list(data.columns)))
    if missing:
        return {"ERROR":f"The following columns are missing: {missing}"}
    trained_model = Module(tuner)
    prediction = trained_model.predict(data)
    prediction = {"predictions":prediction.tolist()}
    return prediction

@app.get("/<user>/<model_name>/ping")
async def ping(request:Request):
    return {"message":"healthy"}