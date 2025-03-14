from fastapi import FastAPI, UploadFile, HTTPException,BackgroundTasks,Request,Form
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
import uvicorn

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

class PredictRequest(BaseModel):
    port:int
    model_name:str
    user_id:int
    data:list

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

    image_id = build_image(path=dist, tag=f"{user_id}_{model_id}")
    run_container(image=image_id, ports={f'{port}/tcp': port},name=f"{user_id}_{model_id}")

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
        return {"progress":1}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_model(request:PredictRequest):
    try:
        data = request
        api_key_url = f"{BASE_URL}/api_keys"
        payload = json.dumps({
            "user_id" :data.user_id,
            "api_name" :str(uuid.uuid4())[:8]
        })
        response = requests.post(api_key_url,data=payload)
        api_key = json.loads(response.text)["api_key"]
        api_key_id = json.loads(response.text)["api_key_id"]
        url = f"""http://127.0.0.1:{data.port}/{data.user_id}/{data.model_name}/predict"""
        payload = json.dumps({
            "data" :data.data
        })
        headers = {
            "x-api-key":api_key
        }
        response = requests.post(url,data=payload,headers=headers)
        requests.delete(f"{BASE_URL}/api_keys/{api_key_id}")
        if response.status_code!= 200:
            raise HTTPException(status_code=500, detail=str(response.text))
        predictions = json.loads(response.text)
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/upload/predict")
def predict_model(file:UploadFile = Form(...),user_id : int=Form(...),port:int=Form(...),model_name:str=Form(...)):
    try:
        valid_extensions = ["csv", "xlsx", "parquet"]
        file_extension = file.filename.split(".")[-1].lower()

        if file_extension not in valid_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format '{file_extension}'. Allowed formats: {', '.join(valid_extensions)}"
            )
        
        content = file.file.read()

        current_dir = os.getcwd()
        os.makedirs(os.path.join(current_dir,"tmp"),exist_ok=True)
        temp_filename = os.path.join(current_dir,"tmp",f"{str(uuid.uuid4())[:8]}_{file.filename}")

        with open(temp_filename, "wb") as temp_file:
            temp_file.write(content)
        
        if file_extension == "csv":
            data = pd.read_csv(temp_filename)
        elif file_extension == "xlsx":
            data = pd.read_excel(temp_filename)
        elif file_extension == "parquet":
            data = pd.read_parquet(temp_filename)
        os.remove(temp_filename)
        data = data.to_dict("records")

        api_key_url = f"{BASE_URL}/api_keys"
        payload = json.dumps({
            "user_id" :user_id,
            "api_name" :str(uuid.uuid4())[:8]
        })
        response = requests.post(api_key_url,data=payload)
        api_key = json.loads(response.text)["api_key"]
        api_key_id = json.loads(response.text)["api_key_id"]
        url = f"""http://127.0.0.1:{port}/{user_id}/{model_name}/predict"""
        payload = json.dumps({
            "data" :data
        })
        headers = {
            "x-api-key":api_key
        }
        response = requests.post(url,data=payload,headers=headers)
        predictions = json.loads(response.text)
        requests.delete(f"{BASE_URL}/api_keys/{api_key_id}")
        return predictions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete")
def delete_model(user_id:int,model_name:str,model_id:int):
    try:
        try:
            _, containers = get_images_and_containers()
            container_id = containers[containers['NAMES']==f"{user_id}_{model_id}"]['CONTAINER ID'].item()
            delete_container(container_id=container_id)
            port = int(containers[containers['CONTAINER ID']==container_id]['COMMAND'].tolist()[0][-1])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        try:
            delete_model_from_nginx_config(user_id=user_id,model_name=model_name,container_port=port)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        update_data = {"status":2}
        requests.put(f"{BASE_URL}/models/status/{model_id}", json=update_data)
        requests.delete(f"{BASE_URL}/models/{model_id}", json=update_data) 
        return {"message": "Model deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
def stop_model(user_id: int,model_id:int):
    try:
        _, containers = get_images_and_containers()
        container_id = containers[containers['NAMES']==f"{user_id}_{model_id}"]['CONTAINER ID'].item()
        stop_container(container_id=container_id)
        update_data = {"status":2}
        response = requests.put(f"{BASE_URL}/models/status/{model_id}", json=update_data)
        return {"message": "Model stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/start")
def start_model(user_id: str,model_id:int):
    try:
        _, containers = get_images_and_containers()
        container_id = containers[containers['NAMES']==f"{user_id}_{model_id}"]['CONTAINER ID'].item()
        start_container(container_id=container_id)
        update_data = {"status":1}
        response = requests.put(f"{BASE_URL}/models/status/{model_id}", json=update_data)
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

if __name__ == "__main__":
    # Run the server. Adjust host/port if needed.
    uvicorn.run(app, host="0.0.0.0", port=8001)