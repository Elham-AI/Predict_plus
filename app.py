import streamlit as st
import pandas as pd
from nelc_autoML import *
from datetime import datetime
import time
import shutil
from streamlit_navigation_bar import st_navbar
import subprocess
from docker_utils import *
os.makedirs('Models',exist_ok=True)
os.makedirs('Deployments',exist_ok=True)


st.set_page_config(
layout="wide",
page_icon="logo.svg"
)

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
            st.session_state['file_updated'] = True
            st.rerun()
    else:
        if st.sidebar.button("Upload another file"):
            st.session_state['file'] = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
            if st.session_state.get('file'):
                st.session_state['file_updated'] = True
                st.rerun()
    if st.session_state.get('file')  is not None:
        if st.session_state.get('file_updated'):
            df = pd.read_csv(st.session_state['file'] )
            st.session_state['data'] = df
            st.session_state['file_updated'] = False
        if isinstance(st.session_state.get('data'),pd.DataFrame):
            st.write("Data Preview", st.session_state['data'].head())
            st.write(f"""The data has {st.session_state['data'].shape[0]} records""")

        # Select target column
        target_column = st.sidebar.selectbox("Select Target Column", options=st.session_state['data'].columns,index = st.session_state.get('TARGET_COLUMN_INDEX') if st.session_state.get('TARGET_COLUMN_INDEX') else 0)
        st.warning("NOTE: Selecting the ID columns will improve model performence")
        if st.session_state.get('ID_COLUMNS'):
            if isinstance(st.session_state.get('ID_COLUMNS'),list):
                cols = list(set(list(st.session_state['data'].columns) + st.session_state.get('ID_COLUMNS')))
            else:
                cols = list(set(list(st.session_state['data'].columns) + [st.session_state.get('ID_COLUMNS')]))
        else:
            cols = list(st.session_state['data'].columns)
        id_columns = st.sidebar.multiselect("Select ID columns", options=cols,default = st.session_state.get('ID_COLUMNS') if st.session_state.get('ID_COLUMNS') else None)
        st.session_state['ID_COLUMNS'] = id_columns
        st.session_state['TARGET_COLUMN_INDEX'] = list(st.session_state['data'].columns).index(target_column)
        training_level = st.sidebar.number_input(f"Training level",max_value=10000,min_value=2,value=st.session_state.get('TRAINING_LEVEL') if st.session_state.get('TRAINING_LEVEL') else 500)
        st.session_state['TRAINING_LEVEL'] = training_level
        
        # Train model
        if st.sidebar.button("Train Model"):
            if not set(st.session_state['ID_COLUMNS']).difference(list(st.session_state['data'].columns)):
                st.session_state['data'] = st.session_state['data'].drop(columns=st.session_state['ID_COLUMNS'])
                st.session_state['TARGET_COLUMN_INDEX'] = list(st.session_state['data'].columns).index(target_column)
            st.markdown("---")
            start_time = time.time()
            st.error("If you leave the page you will lose the training progress, so please do not leave the page until the training is finished")
            progress_text = "Training in progress {p}%. Please wait."
            progress_bar = st.progress(0, text=progress_text.format(p=0))
            tuner = AutoML(data=st.session_state['data'],data_preprocessing=True,target_column=target_column,interpretability=1)
            tuner.init_study()
            print("study init")
            for i in tuner.optimize(n_trials=training_level):
                print("start")
                time.sleep(0.01)
                percent_complete = (i+1)/training_level
                progress_bar.progress(percent_complete, text=progress_text.format(p=round(percent_complete*100,ndigits=2)))
            fig = tuner.final_training()
            st.subheader("Training visualization")
            st.plotly_chart(fig)
            st.warning(f"The training elapsed time: {(round(round(time.time()-start_time)/60,ndigits=2))} Minutes")
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
                            image_id = build_image(path=dist,tag=model_name)
                            run_container(image=image_id,ports={'8000/tcp': 8000})
                            st.success(f"Docker image built successfully")
                            st.session_state['DEPLOYED'] = True
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
    _,col,_ = st.columns(3)
    with col:
        st.image("logo.svg")
    st.title("Welcome to the Auto ML Tool!")
    st.markdown("""
    ### Automate Your Machine Learning Journey
    This Auto ML tool streamlines the process of training, tuning, and deploying machine learning models. With just a few clicks, you can go from data to predictions without needing extensive ML expertise.

    ### Key Features
    - **Data Upload & Selection**: Upload your data and choose your target column.
    - **Automated Model Tuning**: Select the training level and let the tool find the best model for your data.
    - **Model Testing & Prediction**: Input test data to see real-time predictions.
    - **Deployment as an API**: Deploy your trained models as Dockerized APIs with a single click, and manage them easily.

    ### How to Get Started
    1. Go to the **Train and Deploy** page to upload your dataset.
    2. Select your target column and ID columns, then specify the training level.
    3. Train the model, test it, and deploy it as needed.
    
    Enjoy the seamless experience of automated machine learning!
    """)
    _,col1,_ = st.columns(3)
    with col1:
        st.image("faris_ml_create_an_image_of_a_manager_that_thinking_of_a_data_674c193a-ea42-4f37-a1c2-fb7a21d953b8_3.png", caption="Machine Learning Made Simple", use_column_width=True)

def deployed_page():
    st.header("Your deployed models")
    images, containers = get_images_and_containers()
    images = images[images['REPOSITORY'] != 'python']
    if not images.empty and not containers.empty:
        deployed_models = os.listdir('Deployments') 
        images['IMAGE'] = images['REPOSITORY']+":"+images['TAG']
        df = containers.merge(images,on='IMAGE',how='left',suffixes=('_CONTAINER','_IMAGE'))
        not_built_models = set(deployed_models).difference(images['REPOSITORY'].tolist())
        deleted_models = set(images['REPOSITORY'].tolist()).difference(deployed_models)
        st.markdown(df.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        st.subheader("Make action")
         
        model_name = st.selectbox("Model name",options=images['IMAGE'].tolist())
        col1, col2, col3 = st.columns(3)
        with col1:
            _,sub_col1,_ = st.columns(3)
            with sub_col1:
                stop = st.button("Stop") 
            
        with col2:
            _,sub_col2,_ = st.columns(3)
            with sub_col2:
                start = st.button("Start")

        with col3:
            _,sub_col3,_ = st.columns(3)
            with sub_col3:
                delete = st.button("Delete")   
        
        if stop and model_name:
            with st.spinner(""):
                container_ids = df[df['IMAGE']==model_name]['CONTAINER ID'].tolist()
                for container_id in container_ids:
                    stop_container(container_id=container_id)
                st.success("The model is stopped")
                time.sleep(1)
                st.rerun()

        elif start and model_name: 
            with st.spinner(""):
                container_ids = df[df['IMAGE']==model_name]['CONTAINER ID'].tolist()
                for container_id in container_ids:
                    start_container(container_id=container_id)
                st.success("The model is running")
                time.sleep(1)
                st.rerun()
        elif delete and model_name:
            with st.spinner(""):
                container_ids = df[df['IMAGE']==model_name]['CONTAINER ID'].tolist()
                for container_id in container_ids:
                    delete_container(container_id)
                delete_image(model_name)
                shutil.rmtree(os.path.join('Deployments',model_name.split(':')[0]))
                os.remove(os.path.join('Models',f"""{model_name.split(':')[0]}_model.pkl"""))
                os.remove(os.path.join('Models',f"""{model_name.split(':')[0]}_tuner.pkl"""))
                st.success("The model is deleted")
                time.sleep(1)
                st.rerun()
        if model_name:
            st.divider()
            try:
                with open(os.path.join('Deployments',model_name.split(':')[0],'README.md'),'r') as f:
                    file_content = f.read()
                st.markdown(file_content)
            except FileNotFoundError as e:
                st.warning("There is no documentaion for this container !!")
    else:
        images = images.assign(IMAGE=None)
        st.warning("You do not have deployed models")

page = st_navbar(["Home", "Train and deploy", "Deployed models"],logo_path="logo.svg")
pages = {
    "Home":home_page,
    "Train and deploy":train_page,
    "Deployed models":deployed_page
}
pages[page]()