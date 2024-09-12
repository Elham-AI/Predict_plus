import streamlit as st
import pandas as pd
from nelc_autoML import *
from datetime import datetime
import time
# Function to train model
def train_model(df, target_column,training_level):
    tuner = AutoML(data=df,data_preprocessing=True,target_column=target_column,interpretability=1)
    tuner.tune(training_level)
    return tuner

# Function to predict with trained model
def predict_with_model(model, input_data):
    return model.predict(input_data)
# Main Streamlit app

st.sidebar.title('Auto ML Tool')

st.session_state['file'] = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if st.session_state.get('file')  is not None:
    if not isinstance(st.session_state.get('data'),pd.DataFrame):
        df = pd.read_csv(st.session_state['file'] )
        st.session_state['data'] = df
    if isinstance(st.session_state.get('data'),pd.DataFrame):
        st.write("Data Preview", st.session_state['data'].head())

    # Select target column
    target_column = st.sidebar.selectbox("Select Target Column", options=st.session_state['data'].columns)
    training_level = st.sidebar.number_input(f"Training level",max_value=10000,min_value=2,value=500)
    
    # Train model
    if st.sidebar.button("Train Model"):
        st.markdown("---")
        start_time = time.time()
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

# Example instructions
st.markdown("---")
st.subheader("Instructions")
st.markdown("""
1. Upload a CSV file with your data.
2. Select the target column for prediction.
3. Click 'Train Model' to train the model.
4. Use the test inputs below to predict using the trained model.
""")
