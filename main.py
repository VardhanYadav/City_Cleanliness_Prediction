import streamlit as st
from prediction import run as run_prediction
from prediction_oneDNN import prediction_oneDNN  # Import the oneDNN prediction function
from water_analysis import main
from chatbot import chatbot_ui
from waste_management import waste_management

st.title("Team Gradient")
st.write("Select a task to perform:")


# Task selection
task = st.selectbox("Choose a task:", ["Cleanliness Score Prediction", "Cleanliness Score Factors", "Chatbot 🤖"])

if task == "Cleanliness Score Prediction":
    st.subheader("Prediction using Data libraries.", divider="blue")

    # Model selection between oneDAL and oneDNN
    data_analysis = st.radio("Choose an option:", ["oneDAL", "oneDNN"])

    if data_analysis == "oneDAL":
        # Call the oneDAL prediction function
        run_prediction()

    elif data_analysis == "oneDNN":
        # Call the oneDNN prediction function
        prediction_oneDNN()  # Call prediction_oneDNN for the oneDNN task

elif task == "Cleanliness Score Factors":
    st.subheader("Cleanliness Score Factors", divider="blue")

    # Select between different quality analyses
    quality_analysis = st.radio("Choose an option:", ["Water Quality Analysis", "Waste Management Analysis"])

    if quality_analysis == "Water Quality Analysis":
        # Call the water quality analysis script
        main()
    elif quality_analysis == "Waste Management Analysis":
        waste_management()

elif task == "Chatbot 🤖":
    chatbot_ui()

