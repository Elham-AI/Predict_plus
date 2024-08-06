import streamlit as st
import pandas as pd
from nelc_autoML import *
from datetime import datetime
# Function to train model
def train_model(df, target_column,training_level):
    tuner = AutoML(data=df,data_preprocessing=True,target_column=target_column,interpretability=1)
    tuner.tune(training_level)
    return tuner

# Function to predict with trained model
def predict_with_model(model, input_data):
    return model.predict(input_data)

# Main Streamlit app
def main():
    st.sidebar.title('Auto ML Tool')

    # Upload CSV file
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview", df.head())

        # Select target column
        target_column = st.sidebar.selectbox("Select Target Column", options=df.columns)
        training_level = st.sidebar.number_input(f"Training level",max_value=10000,min_value=2,value=500)
        
        # Train model
        if st.sidebar.button("Train Model"):
            st.markdown("---")
            with st.spinner('Training ... !'):
                tuner = train_model(df, target_column,training_level)
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
                    test_data[col] = st.selectbox(f"Enter value for {col}",tuple(list(df[col].unique())))
                    
            for col in st.session_state['TUNER'].features_bool:
                if col != target_column:
                    test_data[col] = st.selectbox(f"Enter value for {col}",tuple(list(df[col].unique())))        
                    
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
                
                prediction = predict_with_model(trained_model, input_data)
                print(prediction)
                st.success(f"Prediction: {prediction}")

    # Example instructions
    st.markdown("---")
    st.subheader("Instructions")
    st.markdown("""
    1. Upload a CSV file with your data.
    2. Select the target column for prediction.
    3. Click 'Train Model' to train the model.
    4. Use the test inputs below to predict using the trained model.
    """)

if __name__ == "__main__":
    
    main()