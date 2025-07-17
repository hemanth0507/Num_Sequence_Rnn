import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

# Get the current directory (important for deployment environments)
model_path = os.path.join(os.path.dirname(__file__), "rnn_sequence_model.h5")

# Load model
model = load_model(model_path)

st.title("📈 RNN Sequence Predictor")
st.write("Enter a numeric sequence to predict the next number.")

user_input = st.text_input("Enter sequence (comma-separated):", "1,2,3")

if user_input:
    try:
        sequence = [float(x.strip()) for x in user_input.split(",")]
        input_array = np.array(sequence).reshape(1, -1, 1)

        prediction = model.predict(input_array)
        next_number = prediction[0][0]

        st.success(f"🔮 Predicted Next Number: {next_number:.2f}")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
