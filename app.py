import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load trained RNN model
model = load_model("rnn_sequence_model.h5")

# Streamlit UI
st.set_page_config(page_title="🔢 RNN Number Predictor", layout="centered")
st.title("🔮 RNN Sequence Predictor")
st.markdown("Enter a sequence of numbers and the model will predict the **next number** using a trained RNN.")

# Input box
user_input = st.text_input("Enter numbers separated by commas", "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20")

# Process input
try:
    sequence = [int(x.strip()) for x in user_input.split(",")]
    window_size = 3

    if len(sequence) < window_size:
        st.warning("⚠️ Please enter at least 3 numbers.")
    else:
        # Prepare input data
        X = []
        y = []
        for i in range(len(sequence) - window_size):
            X.append(sequence[i:i + window_size])
            y.append(sequence[i + window_size])

        X = np.array(X).reshape((len(X), window_size, 1))
        y = np.array(y)

        # Predict full sequence and next number
        y_pred = model.predict(X, verbose=0).flatten()
        next_input = np.array(sequence[-window_size:]).reshape((1, window_size, 1))
        next_number = model.predict(next_input, verbose=0)[0][0]

        st.success(f"✅ Predicted next number: **{next_number:.2f}**")

        # Plot actual vs predicted
        st.subheader("📊 Actual vs Predicted Plot")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(range(window_size, window_size + len(y)), y, label="Actual", marker="o")
        ax.plot(range(window_size, window_size + len(y_pred)), y_pred, label="Predicted", linestyle="--", marker="x")
        ax.set_xlabel("Time Step")
        ax.set_ylabel("Value")
        ax.set_title("Sequence Prediction")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

except Exception as e:
    st.error(f"❌ Error: {e}")
