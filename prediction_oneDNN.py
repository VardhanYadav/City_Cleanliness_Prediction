import time

import tensorflow as tf
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.graph_objects as go


# Function to create the oneDNN model
def create_oneDNN_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(input_shape,)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    return model


def plot_loss_curves(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(history.history['loss']))),
                             y=history.history['loss'],
                             mode='lines+markers',
                             name='Training Loss', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=list(range(len(history.history['val_loss']))),
                             y=history.history['val_loss'],
                             mode='lines+markers',
                             name='Validation Loss', line=dict(color='red')))
    fig.update_layout(title='Loss Curves', xaxis_title='Epochs', yaxis_title='Loss', template='plotly_white')
    return fig


def add_sidebar_info():
    st.sidebar.header("Additional Information on oneDNN Prediction", divider="blue")
    st.sidebar.write("""
    **oneDNN (Deep Neural Network)**: 

    This model uses a neural network architecture to predict the cleanliness score for cities based on historical data. 
    The model consists of multiple layers:
    - **Input Layer**: The model takes 7 input features representing the cleanliness scores of previous years.
    - **Hidden Layers**: Three fully connected layers with ReLU activation are used to learn patterns in the data.
    - **Output Layer**: A single output neuron is used to predict the cleanliness score for 2024.

    ### Key Steps:
    1. **Data Preprocessing**: The features are scaled using StandardScaler to normalize the input.
    2. **Model Training**: The model is trained on the preprocessed data using Adam optimizer and mean squared error loss function. 
       Early stopping is implemented to avoid overfitting.
    3. **Evaluation**: After training, the model's performance is evaluated using metrics like R² score, MAE, and MSE.

    ### Loss Curves:
    The graph shows the loss during training and validation, helping to understand the model's convergence.

    ### Why oneDNN?
    Deep neural networks are useful for capturing complex patterns in data, making them effective for predicting future cleanliness scores.
    """)


def prediction_oneDNN():
    # Load dataset
    data = pd.read_csv("E:/PyCharm/Projects/pythonProject/Cleanest_Cities_India.csv")

    st.header("Prediction using oneDNN", divider="blue")

    # Call the function to add the sidebar info
    add_sidebar_info()

    # Dropdowns to select cities
    city1 = st.selectbox("Select first city", data['City Name'].unique())
    city2 = st.selectbox("Select second city", data['City Name'].unique())

    # Prepare features and target
    X = data[['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000', '2020_Score_Max6000', '2022_Score_Max7500',
              '2023_Score_Max10000']]
    y = data['2024_Score_Predicted'] = data['2023_Score_Max10000'] + (
            data['2023_Score_Max10000'] - data['2016_Score']) / 7

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and train the oneDNN model
    model_oneDNN = create_oneDNN_model(X_train_scaled.shape[1])

    # Train the model with early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model_oneDNN.fit(X_train_scaled, y_train, epochs=100, validation_data=(X_test_scaled, y_test),
                               callbacks=[early_stopping], verbose=0)

    # Predict for selected cities after the "Predict" button is pressed
    if st.button("Predict"):
        progress_bar = st.progress(0, text=f"Making prediction for {city1}")
        time.sleep(0.5)

        selected_city1 = data[data['City Name'] == city1]
        selected_city2 = data[data['City Name'] == city2]

        X_city1 = selected_city1[
            ['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000', '2020_Score_Max6000', '2022_Score_Max7500',
             '2023_Score_Max10000']]
        X_city2 = selected_city2[
            ['2016_Score', '2017_Score', '2018_Score', '2019_Score_5000', '2020_Score_Max6000', '2022_Score_Max7500',
             '2023_Score_Max10000']]

        # Scale the city data using the same scaler used in training
        X_city1_scaled = scaler.transform(X_city1)
        X_city2_scaled = scaler.transform(X_city2)

          # Simulate prediction delay

        # Make predictions for both cities
        pred_city1 = model_oneDNN.predict(X_city1_scaled)[0][0]

        progress_bar.progress(50, text=f"Making prediction for {city2}")
        time.sleep(0.5)  # Simulate prediction delay

        pred_city2 = model_oneDNN.predict(X_city2_scaled)[0][0]
        progress_bar.progress(100, text="Prediction completed")

        # Display predictions
        st.success(f"Predicted Cleanliness Score for {city1}: {pred_city1:.2f}")
        st.success(f"Predicted Cleanliness Score for {city2}: {pred_city2:.2f}")

        # Evaluate the model performance on the test set
        y_pred_oneDNN = model_oneDNN.predict(X_test_scaled)
        r2_oneDNN = r2_score(y_test, y_pred_oneDNN)

        st.write(f"Accuracy for oneDNN: {r2_oneDNN:.4f}")

        st.write(f"Below is a graph for loss curves for training and validation")
        # Plot the loss curves for training and validation using Plotly
        loss_fig = plot_loss_curves(history)
        st.plotly_chart(loss_fig)


