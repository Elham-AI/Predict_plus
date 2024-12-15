from fastapi import FastAPI, UploadFile, HTTPException
import pandas as pd
from autoML import AutoML,Module
import os
import shutil
import json
import pickle
from docker_utils import build_image, run_container, get_images_and_containers, stop_container, start_container, delete_container, delete_image
from nginx_utils import add_model_to_nginx_config, delete_model_from_nginx_config

app = FastAPI()

# Directories setup
os.makedirs('Models', exist_ok=True)
os.makedirs('Deployments', exist_ok=True)

@app.post("/train")
def train_model(dataset_path: str, target_column: str, id_columns: list[str] = [], training_level: int = 500):
    try:
        df = pd.read_csv(dataset_path)
        if id_columns:
            df = df.drop(columns=id_columns)
        
        tuner = AutoML(data=df, data_preprocessing=True, target_column=target_column, interpretability=1)
        tuner.init_study()
        for _ in tuner.optimize(n_trials=training_level):
            pass
        fig = tuner.final_training()
        model_score = tuner.score

        # Save model
        model_name = f"{target_column}_model"
        tuner.save(model_name)
        
        return {"message": "Model trained successfully", "score": model_score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_model(user_id: str,model_name: str, input_data: dict):
    try:
        tuner_path = f"{user_id}-{model_name}_tuner.pkl"
        model_path = f"{user_id}-{model_name}_model.pkl"
        # if not os.path.exists(model_path):
        #     raise HTTPException(status_code=404, detail="Model not found")
        with open(tuner_path,'rb') as f: 
            tuner = pickle.load(f)
        with open(model_path,'rb') as f: 
            model = pickle.load(f)
        tuner.model = model
        trained_model = Module(tuner)
        input_df = pd.DataFrame([input_data])
        prediction = trained_model.predict(input_df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
