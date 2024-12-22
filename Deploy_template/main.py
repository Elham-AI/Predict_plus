from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN, HTTP_500_INTERNAL_SERVER_ERROR, HTTP_400_BAD_REQUEST
import requests
import pandas as pd
import pickle
from autoML import Module

# FastAPI app instance
app = FastAPI()

# API key header name
API_KEY_NAME = "X-API-Key"

# API key validation dependency
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if not api_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="API key missing"
        )
    
    try:
        response = requests.post(
            "https://elham.ai/api/api_keys/auth",
            json={"user_id": <user>, "api_key": api_key},
            timeout=10
        )
        if response.status_code != 200:
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail="Unauthorized"
            )
    except requests.RequestException as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate API key: {str(e)}"
        )

# Prediction endpoint
@app.post("/<user>/<model_name>/predict")
async def predict(request:Request,api_key: str = Depends(verify_api_key)):      
    """
    Endpoint to make predictions using a trained model.
    """
    try:
        # Parse JSON data
        payload = await request.json()
        if "data" not in payload:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail="Invalid request: 'data' field is required"
            )

        data = pd.DataFrame(payload["data"])
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Error parsing input data: {str(e)}"
        )

    try:
        # Load model and tuner
        with open(f"<model_id>_tuner.pkl", "rb") as f:
            tuner = pickle.load(f)
        with open(f"<model_id>_model.pkl", "rb") as f:
            model = pickle.load(f)
        tuner.model = model
    except FileNotFoundError:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Model or tuner not found for <model_name>"
        )
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error loading model: {str(e)}"
        )

    try:
        # Preprocess data
        dates_cols = tuner.features_date
        num_features = tuner.features_num
        for col in dates_cols:
            data[col] = pd.to_datetime(data[col])
        
        missing_columns = set(dates_cols + tuner.features_cat + tuner.features_bool).difference(data.columns)
        if missing_columns:
            return JSONResponse(
                status_code=HTTP_400_BAD_REQUEST,
                content={"error": f"The following columns are missing: {missing_columns}"}
            )

        # Make predictions
        trained_model = Module(tuner)
        predictions = trained_model.predict(data)
        return {"predictions": predictions.tolist()}
    except Exception as e:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error during prediction: {str(e)}"
        )


@app.get("/<user>/<model_name>/ping")
async def ping(request:Request,api_key: str = Depends(verify_api_key)):
    """
    Health check endpoint.
    """
    return {"message": f"Model <model_name> for user <user> is healthy"}