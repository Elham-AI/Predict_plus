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
from fastapi.middleware.cors import CORSMiddleware
import requests
from pydantic import BaseModel
from typing import List, Optional
from fastapi import BackgroundTasks, HTTPException
import certifi

load_dotenv(override=True)
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET")
AWS_REGION = os.getenv("AWS_REGION")
API_HOST = os.getenv("API_HOST")
API_PORT = os.getenv("API_PORT")
# BASE_URL = API_HOST
BASE_URL = f"http://{API_HOST}:{API_PORT}"
app = FastAPI()
origins = [
    "http://localhost:3000",  # Frontend local development
    "http://127.0.0.1:3000",  # Frontend local development alternative
    "http://your-production-domain.com",  # Add your production domain
    "*"  # Allow all origins (use with caution in production)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Origins that are allowed to access the API
    allow_credentials=True,  # Allow cookies to be sent
    allow_methods=["*"],  # HTTP methods to allow (GET, POST, DELETE, etc.)
    allow_headers=["*"],  # Headers to allow (Authorization, Content-Type, etc.)
)
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
os.makedirs('tmp/progresses', exist_ok=True)



# Define the request body schema
class TrainRequest(BaseModel):
    model_id: int
    model_name: str
    user_id: int
    dataset_path: str
    target_column: str
    id_columns: Optional[List[str]] = []
    training_level: int = 500

def clean_up(model_id):
    os.remove(f"Models/{model_id}_tuner.pkl")
    os.remove(f"Models/{model_id}_model.pkl")
    os.remove(f"tmp/progresses/{model_id}.json")
    shutil.rmtree(os.path.join('Deployments', str(model_id)))

def deploy(model_id,user_id,model_name):
    # try:
    _, containers = get_images_and_containers()
    port = 8000 if containers.empty else max([int(c[-1]) for c in containers['COMMAND'] if c[-1].isdigit()] or [8000]) + 1
    # since 8001 is mlapi port
    if port == 8001:
        port = 8002
    dist = os.path.join('Deployments', str(model_id))
    os.makedirs(dist, exist_ok=True)

    # Copy template files and replace placeholders
    files = os.listdir('Deploy_template')
    for file in files:
        shutil.copyfile(os.path.join('Deploy_template', file), os.path.join(dist, file))

    with open(os.path.join(dist, 'main.py'), 'r') as f:
        file_content = f.read()
    file_content = file_content.replace("<model_name>", model_name)
    file_content = file_content.replace("<model_id>", str(model_id))
    file_content = file_content.replace("<user>", str(user_id))
    with open(os.path.join(dist, 'main.py'), 'w') as f:
        f.write(file_content)
        
    # with open(os.path.join(dist,'README.md'),'r') as f:
    #     file_content = f.read()
    # file_content = file_content.replace("[API Name]",model_name)
    # for col in dates:
    #     if col != target_column:
    #         input_data[col] = input_data[col].astype(str)
    # file_content = file_content.replace("<input_json>",json.dumps({"data":input_data.to_dict('records')},indent=4))
    # file_content = file_content.replace("<port>",str(port))
    # file_content = file_content.replace("<model_name>",model_name)
    # with open(os.path.join(dist,'README.md'),'w') as f:
    #     f.write(file_content)

    
    with open(os.path.join(dist,'dockerfile'),'r') as f:
        file_content = f.read()

    file_content = file_content.replace("<port>",str(port))
    with open(os.path.join(dist,'dockerfile'),'w') as f:
        f.write(file_content)

    shutil.copyfile(os.path.join('Models',f"{model_id}_model.pkl"), os.path.join(dist,f"{model_id}_model.pkl"))
    shutil.copyfile(os.path.join('Models',f"{model_id}_tuner.pkl"), os.path.join(dist,f"{model_id}_tuner.pkl"))
    shutil.copyfile('autoML.py', os.path.join(dist,'autoML.py'))
    

    add_model_to_nginx_config(user_id=user_id,model_name=model_name,container_port=port)

    image_id = build_image(path=dist, tag=model_name)
    run_container(image=image_id, ports={f'{port}/tcp': port})

    return port,True
    # except Exception as e:
    #     print(e)
    #     return 0,False

   
def training_in_background(tuner:AutoML, training_level,model_id,model_name,user_id):
    name = f"{model_id}"

    for i in tuner.optimize(n_trials=training_level):
        with open(f"tmp/progresses/{name}.json", "r") as file:
            progress = json.load(file)
        progress['progress'] = i/training_level
        with open(f"tmp/progresses/{name}.json", "w") as file:
            json.dump(progress, file, indent=4)

    tuner.final_training()
    with open(f"tmp/progresses/{name}.json", "r") as file:
        progress = json.load(file)
    
    progress['progress'] = 0.99
    with open(f"tmp/progresses/{name}.json", "w") as file:
        json.dump(progress, file, indent=4)
    model_score = tuner.score
    tuner.save(name)
    
    # Save model
    s3_key_model = f"models/{name}_model.pkl"
    s3_key_tuner = f"models/{name}_tuner.pkl"
    s3_client.upload_file(f"Models/{name}_model.pkl", AWS_S3_BUCKET, s3_key_model)
    s3_client.upload_file(f"Models/{name}_tuner.pkl", AWS_S3_BUCKET, s3_key_tuner)
    port,deployed = deploy(model_id=model_id,model_name=model_name,user_id=user_id)
    
    if deployed:
        update_data = {
            "port": port,
            "score": round(model_score,ndigits=3),
            "status":1
            
        }

        # Make the PUT request
        response = requests.put(f"{BASE_URL}/models/{model_id}", json=update_data)
        print(response.text)
    else:
        update_data = {
            "port": port,
            "score": round(model_score,ndigits=3),
            "status":2
            
        }

        # Make the PUT request
        response = requests.put(f"{BASE_URL}/models/{model_id}", json=update_data)
        print(response.text)
    
    with open(f"tmp/progresses/{name}.json", "r") as file:
        progress = json.load(file)
    
    progress['progress'] = 1
    with open(f"tmp/progresses/{name}.json", "w") as file:
        json.dump(progress, file, indent=4)
    clean_up(model_id)
    return {"message": "Model trained successfully", "score": model_score}

@app.post("/train")
def train_model(request: TrainRequest, background_tasks: BackgroundTasks):
    try:
        with open(f"tmp/progresses/{request.model_id}.json", "w") as file:
            json.dump({"progress":0.0}, file, indent=4)

        df = pd.read_csv(f"s3://{AWS_S3_BUCKET}/{request.dataset_path}", storage_options={
    "key": AWS_ACCESS_KEY,
    "secret": AWS_SECRET_KEY,
    "client_kwargs": {"region_name": AWS_REGION}
})
        if request.id_columns:
            df = df.drop(columns=request.id_columns)
        
        tuner = AutoML(data=df, data_preprocessing=True, target_column=request.target_column, interpretability=1)
        tuner.init_study()
        background_tasks.add_task(training_in_background, tuner, request.training_level,request.model_id,request.model_name,request.user_id)
        return {"message": f"Training is started!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/train/progress")
def train_model(model_id: str):
    try:
        name = f"{model_id}"
        with open(f"tmp/progresses/{name}.json", "r") as file:
            progress = json.load(file)
        return progress   
    except FileNotFoundError as e:
        #TODO Make it 1
        return {"progress":0.99}
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

@app.delete("/delete")
def delete_model(user_id:int,model_name: str,model_id:int):
    try:
        _, containers = get_images_and_containers()
        container_id = containers[containers['REPOSITORY']==f"{model_name}:latest"].any()
        delete_container(container_id=container_id)
        port = int(containers[containers['CONTAINER ID']==container_id]['COMMAND'].tolist()[0][-1])
        try:
            delete_model_from_nginx_config(user_id=user_id,model_name=model_id,container_port=port)
        except Exception as e:
            print(e)
        shutil.rmtree(os.path.join('Deployments',str(model_id)))
        update_data = {"status":2,"is_deleted":True}
        response = requests.put(f"{BASE_URL}/models/{model_id}", json=update_data) 
        return {"message": "Model deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def delete_model(model_name: str,model_id:int):
    try:
        _, containers = get_images_and_containers()
        container_id = containers[containers['REPOSITORY']==f"{model_name}:latest"].any()
        stop_container(container_id=container_id)
        update_data = {"status":2}
        response = requests.put(f"{BASE_URL}/models/{model_id}", json=update_data) 
        return {"message": "Model stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.delete("/start")
def delete_model(model_name: str,model_id:int):
    try:
        _, containers = get_images_and_containers()
        container_id = containers[containers['REPOSITORY']==f"{model_name}:latest"].any()
        start_container(container_id=container_id)
        update_data = {"status":1}
        response = requests.put(f"{BASE_URL}/models/{model_id}", json=update_data) 
        return {"message": "Model started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deployed_models")
def deployed_models():
    try:
        _, containers = get_images_and_containers()
        return containers.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

