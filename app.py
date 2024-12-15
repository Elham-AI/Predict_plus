from fastapi import FastAPI, UploadFile, HTTPException,BackgroundTasks
import pandas as pd
from autoML import AutoML,Module
import os
import shutil
import uuid
import json
import pickle
import boto3
from docker_utils import build_image, run_container, get_images_and_containers, stop_container, start_container, delete_container, delete_image
from nginx_utils import add_model_to_nginx_config, delete_model_from_nginx_config
from dotenv import load_dotenv
import requests

load_dotenv()
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
BASE_URL = f"http://{API_HOST}:{API_PORT}"
app = FastAPI()
s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)
# Directories setup
os.makedirs('Models', exist_ok=True)
os.makedirs('Deployments', exist_ok=True)
os.makedirs('tmp', exist_ok=True)
os.makedirs('tmp/status', exist_ok=True)
def deploy():
    try:
        dist = os.path.join('Deployments', model_name)
        os.makedirs(dist, exist_ok=True)

        # Copy template files and replace placeholders
        files = os.listdir('Deploy_template')
        for file in files:
            shutil.copyfile(os.path.join('Deploy_template', file), os.path.join(dist, file))

        with open(os.path.join(dist, 'main.py'), 'r') as f:
            file_content = f.read()
        file_content = file_content.replace("<model_name>", model_name)
        file_content = file_content.replace("<user>", user_id)
        with open(os.path.join(dist, 'main.py'), 'w') as f:
            f.write(file_content)

        _, containers = get_images_and_containers()
        port = 8000 if containers.empty else max([int(c[-1]) for c in containers['COMMAND'] if c[-1].isdigit()] or [8000]) + 1

        add_model_to_nginx_config("yourdomain.com", model_name, port)

        image_id = build_image(path=dist, tag=model_name)
        run_container(image=image_id, ports={f'{port}/tcp': port})

        return {"message": "Model deployed successfully", "url": f"http://yourdomain.com:{port}/{model_name}/predict"}
    except Exception as e:
        return None
def training_in_background(tuner:AutoML, training_level,dataset_id,model_id):
    name = f"{dataset_id}-{model_id}"
    with open(f"tmp/status/{name}.json", "w") as file:
        json.dump({"progress":0.0}, file, indent=4)
    
    for i in tuner.optimize(n_trials=training_level):
        with open(f"tmp/status/{name}.json", "r") as file:
            status = json.load(file)
        status['progress'] = i/training_level
        with open(f"tmp/status/{name}.json", "w") as file:
            json.dump(status, file, indent=4)
    tuner.final_training()
    with open(f"tmp/status/{name}.json", "r") as file:
        status = json.load(file)
    status['progress'] = 100
    with open(f"tmp/status/{name}.json", "w") as file:
        json.dump(status, file, indent=4)
    model_score = tuner.score
    
    tuner.save(name)
    
    # Save model
    s3_key_model = f"models/{name}_model.pkl"
    s3_key_tuner = f"models/{name}_tuner.pkl"
    s3_client.upload_file(f"Models/{name}_model.pkl", AWS_S3_BUCKET, s3_key_model)
    s3_client.upload_file(f"Models/{name}_tuner.pkl", AWS_S3_BUCKET, s3_key_tuner)
    os.remove(f"Models/{name}_tuner.pkl")
    os.remove(f"Models/{name}_tuner.pkl")
        
    update_data = {
        "port": 8080,
        "target_column": "updated_column",
        "training_level": 3
    }

    # Make the PUT request
    response = requests.put(f"{BASE_URL}/models/{model_id}", json=update_data)  
    return {"message": "Model trained successfully", "score": model_score}

@app.post("/train")
def train_model(dataset_id: str,model_id: str,dataset_path: str, target_column: str,background_tasks: BackgroundTasks, id_columns: list[str] = [], training_level: int = 500):
    try:
        df = pd.read_csv(dataset_path)
        if id_columns:
            df = df.drop(columns=id_columns)
        
        tuner = AutoML(data=df, data_preprocessing=True, target_column=target_column, interpretability=1)
        tuner.init_study()
        background_tasks.add_task(training_in_background, tuner, training_level,dataset_id,model_id)
        return {"message": f"Training is started!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/status")
def train_model(user_id: str,model_id: str):
    try:
        name = f"{user_id}-{model_id}"
        with open(f"tmp/status/{name}.json", "r") as file:
            status = json.load(file)
        return status    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/predict")
# def predict_model(user_id: str,model_name: str, input_data: dict):
#     try:
#         tuner_path = f"{user_id}-{model_name}_tuner.pkl"
#         model_path = f"{user_id}-{model_name}_model.pkl"
#         # if not os.path.exists(model_path):
#         #     raise HTTPException(status_code=404, detail="Model not found")
#         with open(tuner_path,'rb') as f: 
#             tuner = pickle.load(f)
#         with open(model_path,'rb') as f: 
#             model = pickle.load(f)
#         tuner.model = model
#         trained_model = Module(tuner)
#         input_df = pd.DataFrame([input_data])
#         prediction = trained_model.predict(input_df)
#         return {"prediction": prediction.tolist()}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))



@app.post("/deploy")
def deploy_model(model_name: str,user_id:int):
    try:
        dist = os.path.join('Deployments', model_name)
        os.makedirs(dist, exist_ok=True)

        # Copy template files and replace placeholders
        files = os.listdir('Deploy_template')
        for file in files:
            shutil.copyfile(os.path.join('Deploy_template', file), os.path.join(dist, file))

        with open(os.path.join(dist, 'main.py'), 'r') as f:
            file_content = f.read()
        file_content = file_content.replace("<model_name>", model_name)
        file_content = file_content.replace("<user>", user_id)
        with open(os.path.join(dist, 'main.py'), 'w') as f:
            f.write(file_content)

        _, containers = get_images_and_containers()
        port = 8000 if containers.empty else max([int(c[-1]) for c in containers['COMMAND'] if c[-1].isdigit()] or [8000]) + 1

        add_model_to_nginx_config("yourdomain.com", model_name, port)

        image_id = build_image(path=dist, tag=model_name)
        run_container(image=image_id, ports={f'{port}/tcp': port})

        return {"message": "Model deployed successfully", "url": f"http://yourdomain.com:{port}/{model_name}/predict"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete")
def delete_model(model_name: str):
    try:
        dist = os.path.join('Deployments', model_name)
        shutil.rmtree(dist)
        os.remove(os.path.join('Models', f"{model_name}_model.pkl"))
        os.remove(os.path.join('Models', f"{model_name}_tuner.pkl"))
        delete_model_from_nginx_config("yourdomain.com", model_name)
        return {"message": "Model deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/deployed_models")
def deployed_models():
    try:
        _, containers = get_images_and_containers()
        return containers.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
