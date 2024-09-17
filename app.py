import streamlit as st
import pandas as pd
from nelc_autoML import *
from datetime import datetime
import time
import shutil
from streamlit_navigation_bar import st_navbar
import subprocess

os.makedirs('Models',exist_ok=True)
os.makedirs('Deployments',exist_ok=True)
# Function to train model
def train_model(df, target_column,training_level):
    tuner = AutoML(data=df,data_preprocessing=True,target_column=target_column,interpretability=1)
    tuner.tune(training_level)
    return tuner

# Function to predict with trained model
def predict_with_model(model, input_data):
    return model.predict(input_data)
# Main Streamlit app
def train_page():
    st.sidebar.title('Auto ML Tool')
    if not st.session_state.get('file'):
        st.session_state['file'] = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
        if st.session_state.get('file'):
            st.rerun()
    else:
        if st.sidebar.button("Upload another file"):
            st.session_state['file'] = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if st.session_state.get('file')  is not None:
        if not isinstance(st.session_state.get('data'),pd.DataFrame):
            df = pd.read_csv(st.session_state['file'] )
            st.session_state['data'] = df
        if isinstance(st.session_state.get('data'),pd.DataFrame):
            st.write("Data Preview", st.session_state['data'].head())
            st.write(f"""The data has {st.session_state['data'].shape[0]} records""")

        # Select target column
        target_column = st.sidebar.selectbox("Select Target Column", options=st.session_state['data'].columns,index = st.session_state.get('TARGET_COLUMN_INDEX') if st.session_state.get('TARGET_COLUMN_INDEX') else 0)
        st.session_state['TARGET_COLUMN_INDEX'] = list(st.session_state['data'].columns).index(target_column)
        training_level = st.sidebar.number_input(f"Training level",max_value=10000,min_value=2,value=st.session_state.get('TRAINING_LEVEL') if st.session_state.get('TRAINING_LEVEL') else 500)
        st.session_state['TRAINING_LEVEL'] = training_level
        
        # Train model
        if st.sidebar.button("Train Model"):
            st.markdown("---")
            start_time = time.time()
            st.error("If you leave the page you will lose the training progress, so please do not leave the page until the training is finished")
            with st.spinner('Training ... !'):       
                tuner = train_model(st.session_state['data'], target_column,training_level)
            st.warning(f"The training elapsed time: {(round(time.time()-start_time)/60)} Minutes")
            st.success(f"Model trained with score: {tuner.score}")
            dates = tuner.features_date
            st.session_state['DATES'] = dates
            st.session_state['TRAINED'] = True
            st.session_state['TUNER'] = tuner
        
        if st.session_state.get('TRAINED'):
            # Test page to input data and predict
            st.subheader("Test Model")
            test_data = {}
            dates = st.session_state['DATES']
            for col in st.session_state['TUNER'].features_cat:
                if col != target_column:
                    test_data[col] = st.selectbox(f"Enter value for {col}",tuple(list(st.session_state['data'][col].unique())))
                    
            for col in st.session_state['TUNER'].features_bool:
                if col != target_column:
                    test_data[col] = st.selectbox(f"Enter value for {col}",tuple(list(st.session_state['data'][col].unique())))        
                    
            for col in st.session_state['TUNER'].features_num:
                flag = False
                for date in dates:
                    if date in col:
                        flag = True
                if flag:
                    continue
                if col != target_column:
                    test_data[col] = st.number_input(f"Enter value for {col}")

            for col in dates:
                if col != target_column:
                    test_data[col] = st.date_input(f"Enter value for {col}")
                    test_data[col] = pd.to_datetime(test_data[col])

            input_data = pd.DataFrame([test_data])
            
            trained_model = Module(st.session_state['TUNER'])
            if st.button("Predict"):
                with st.spinner("Predicting ... !"):
                    prediction = predict_with_model(trained_model, input_data)
                st.success(f"Prediction: {prediction[0]}")
            if st.session_state.get('DEPLOYED'):
                st.success('The model is deployed successfuly')
            if not st.session_state.get('DEPLOYED'):
                st.subheader("Deploy as an API:")
                model_name = st.text_input("Input the model name")
                model_name = model_name.replace(" ","_")
                model_name = model_name.replace("-","_")
                if st.button("Deploy"): 
                    with st.spinner("Deploying...!"):  
                        if model_name:
                            st.session_state['TUNER'].save(model_name)
                            files = os.listdir('Deploy_template')
                            dist = os.path.join('Deployments',model_name)
                            os.makedirs(dist,exist_ok=True)
                            for i in files:
                                shutil.copyfile(os.path.join('Deploy_template',i),os.path.join(dist,i))
                            with open(os.path.join(dist,'main.py'),'r') as f:
                                file_content = f.read()
                            file_content = file_content.replace("<model_name>",model_name)
                            with open(os.path.join(dist,'main.py'),'w') as f:
                                f.write(file_content)

                            with open(os.path.join(dist,'README.md'),'r') as f:
                                file_content = f.read()
                            file_content = file_content.replace("[API Name]",model_name)
                            for col in dates:
                                if col != target_column:
                                    input_data[col] = input_data[col].astype(str)
                            file_content = file_content.replace("<input_json>",json.dumps(input_data.to_dict('records'),indent=4))
                            with open(os.path.join(dist,'README.md'),'w') as f:
                                f.write(file_content)
                            shutil.copyfile(os.path.join('Models',f"{model_name}_model.pkl"), os.path.join(dist,f"{model_name}_model.pkl"))
                            shutil.copyfile(os.path.join('Models',f"{model_name}_tuner.pkl"), os.path.join(dist,f"{model_name}_tuner.pkl"))
                            shutil.copyfile('nelc_autoML.py', os.path.join(dist,'nelc_autoML.py'))
                            command = ["sudo","docker", "build", "-t", f"{model_name}","."]
                            # password = "user@123"
                            # Start the Docker command in the background
                            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,cwd=dist)
                            # stdout, stderr = process.communicate(input=password + '\n')
                            # stdout, stderr = process.communicate()

                            if process.returncode == 0:
                                st.success(f"Docker image built successfully")
                                st.session_state['DEPLOYED'] = True
                                print(process.stdout)
                            else:
                                st.error("Failed to build Docker image")
                                print(process.stderr)
                        else:
                            st.error("Please enter the model name to deploy")

    # Example instructions
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. Upload a CSV file with your data.
    2. Select the target column for prediction.
    3. Click 'Train Model' to train the model.
    4. Use the test inputs below to predict using the trained model.
    """)

def home_page():
    st.header("Welcome to the auto-ML tool")

def deployed_page():
    st.header("Your deployed models")
    deployed_models = os.listdir('Deployments')
    df = pd.DataFrame({"Model Name":deployed_models})
    df['Status'] = 'deployed'
    df['Version'] = 1
    st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
    st.subheader("Make action")
    model_name = st.selectbox("Model name",options=deployed_models)

    if st.button("Stop") and model_name:
        st.success("The model is stopped")

    elif st.button("Run") and model_name: 
        with st.spinner(""):
            command = ["sudo","docker", "run","--name", model_name,"-p","8000:8000","-d",model_name]
            # Start the Docker command in the background
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,cwd=os.path.join('Deployments',model_name))
            # stdout, stderr = process.communicate()

            if process.returncode == 0:
                print("Docker image running successfully:\n", process.stdout)
                st.rerun()
                st.success("The model is running")
            else:
                print("Failed to build Docker image:\n", process.stderr)
        
    elif st.button("Delete") and model_name:
        with st.spinner(""):
            
            command = ["sudo","docker", "rm",model_name]
            # Start the Docker command in the background
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,cwd=os.path.join('Deployments',model_name))
            stdout, stderr = process.communicate()
            command = ["sudo","docker", "rmi",model_name]
            # Start the Docker command in the background
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,cwd=os.path.join('Deployments',model_name))
            stdout, stderr = process.communicate()
            shutil.rmtree(os.path.join('Deployments',model_name))
            os.remove(os.path.join('Models',f"{model_name}_model.pkl"))
            os.remove(os.path.join('Models',f"{model_name}_tuner.pkl"))
            if process.returncode == 0:
                print("Docker image deleted successfully:\n", stdout)
                st.rerun()
                st.success("The model is deleted")
            else:
                print("Failed to build Docker image:\n", stderr)
    if model_name:
        st.divider()
        with open(os.path.join('Deployments',model_name,'README.md'),'r') as f:
            file_content = f.read()
        st.markdown(file_content)
        
        


page = st_navbar(["Home", "Train and deploy", "Deployed models"])
pages = {
    "Home":home_page,
    "Train and deploy":train_page,
    "Deployed models":deployed_page
}
pages[page]()