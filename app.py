
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.exceptions import NotFittedError

# Page config
st.set_page_config(page_title="Health Risk Prediction", layout="centered")
st.title("🧬 Health Risk Prediction App")
st.markdown("Enter the patient's health indicators to predict diabetes risk.")

# Load model
@st.cache_resource
def load_model():
    try:
        with open("diabetes_model.pkl", "rb") as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

model = load_model()

# Feature input form
feature_names = ['Gender', 'Age', 'Hyp', 'HD', 'Smok', 'BMI', 'HbA1c test', 'BGL']

st.subheader("📝 Patient Information")
with st.form("prediction_form"):
    cols = st.columns(2)
    user_input = dict()

    for i, feature in enumerate(feature_names):
        with cols[i % 2]:
            if feature in ["Gender", "Hyp", "HD", "Smok", "HbA1c test"]:
                user_input[feature] = st.selectbox(
                    f"{feature}", options=[0, 1,], help="0 = No / Female, 1 = Yes / Male"
                )
            else:
                user_input[feature] = st.number_input(f"{feature}", value=0.0)

    submitted = st.form_submit_button("🔍 Predict")

# Make prediction
if submitted:
    if model is not None:
        try:
            input_df = pd.DataFrame([user_input])
            prediction = model.predict(input_df)[0]
            label = "🟢 Positive" if prediction == 1 else "🔴 Negative"
            st.success(f"Prediction: {label} (0 = Negative, 1 = Positive)")
        except NotFittedError:
            st.error("❌ Model is not trained yet.")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")