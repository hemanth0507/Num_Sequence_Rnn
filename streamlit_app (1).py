import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os

# Safe path handling (for Streamlit Cloud)
model_path = os.path.join(os.getcwd(), "rnn_sequence_model.h5")

# Load model
model = load_model(model_path)

# UI
st.title("📈 RNN Sequence Predictor")
st.markdown("Enter a numeric sequence (comma-separated) to predict the next number using a trained RNN model.")

user_input = st.text_input("🔢 Enter Sequence:", "1,2,3")

if user_input:
    try:
        # Parse input and reshape
        sequence = [float(x.strip()) for x in user_input.split(",")]
        input_array = np.array(sequence).reshape(1, -1, 1)

        # Prediction
        prediction = model.predict(input_array)
        next_number = prediction[0][0]

        st.success(f"🔮 Predicted Next Number: `{next_number:.2f}`")
    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
