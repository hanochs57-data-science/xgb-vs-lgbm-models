import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load your trained models and label encoders
@st.cache_resource
def load_assets():
    # Load Machine Learning Models
    with open("xgb_model.pkl", "rb") as f:
        xgb_model = joblib.load(f)
    with open("lgb_model.pkl", "rb") as f:
        lgb_model = joblib.load(f)
        
    # Load Label Encoders
    with open("le_embarked.pkl", "rb") as f:
        le_embarked = joblib.load(f)
    with open("le_who.pkl", "rb") as f:
        le_who = joblib.load(f)
    with open("le_sex.pkl", "rb") as f:
        le_sex = joblib.load(f)
        
    return xgb_model, lgb_model, le_embarked, le_who, le_sex

try:
    xgb_model, lgb_model, le_embarked, le_who, le_sex = load_assets()
except FileNotFoundError as e:
    st.error(f"Missing asset file error: {e}. Ensure models and encoders are in the folder.")
    st.stop()

# 2. App Layout Configuration
st.title("🚢 Titanic Survival Prediction App")
st.write("Predict whether a passenger would survive the Titanic disaster using XGBoost and LightGBM.")

st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["XGBoost", "LightGBM"])

# 3. User Input Fields
st.subheader("📋 Enter Passenger Details")
col1, col2, col3 = st.columns(3)

with col1:
    pclass = st.selectbox("Ticket Class (pclass)", [1, 2, 3], index=2)
    # Options are generated safely from the classes found during training
    sex = st.selectbox("Sex", list(le_sex.classes_))
    age = st.number_input("Age", min_value=0.0, max_value=100.0, value=28.0, step=0.5)

with col2:
    sibsp = st.number_input("Siblings/Spouses Aboard (sibsp)", min_value=0, max_value=10, value=0)
    parch = st.number_input("Parents/Children Aboard (parch)", min_value=0, max_value=10, value=0)
    fare = st.number_input("Passenger Fare", min_value=0.0, max_value=600.0, value=32.0, step=1.0)

with col3:
    embarked = st.selectbox("Port of Embarkation", list(le_embarked.classes_))
    who = st.selectbox("Who", list(le_who.classes_))
    alone = st.selectbox("Traveling Alone?", [True, False], index=0)

# 4. Input Transformation & Pipeline Assembly
def preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, who, alone):
    # Map boolean to numeric flag if needed by model framework
    alone_encoded = 1 if alone else 0
    
    # Safely transform categorical values into their trained integer matches
    sex_encoded = le_sex.transform([sex])[0]
    embarked_encoded = le_embarked.transform([embarked])[0]
    who_encoded = le_who.transform([who])[0]
    
    # Construct the array matching your exact training layout 
    input_data = pd.DataFrame([{
        'pclass': pclass,
        'sex': sex_encoded,
        'age': age,
        'sibsp': sibsp,
        'parch': parch,
        'fare': fare,
        'embarked': embarked_encoded,
        'who': who_encoded,
        'alone': alone_encoded
    }])
    
    return input_data

# 5. Model Inference Engine
if st.button("Predict Survival", type="primary"):
    # Preprocess inputs
    features = preprocess_input(pclass, sex, age, sibsp, parch, fare, embarked, who, alone)
    
    # Route data to target model architecture
    selected_model = xgb_model if model_choice == "XGBoost" else lgb_model
    prediction = selected_model.predict(features)
    
    # Calculate inference likelihood
    try:
        prob_array = selected_model.predict_proba(features)
        # Pull index [0][1] for the survival probability class
        confidence = prob_array[0][1] if prediction[0] == 1 else prob_array[0][0]
    except (AttributeError, IndexError):
        confidence = None

    # Render Visual Feedbacks
    st.write("---")
    if prediction[0] == 1:
        st.success("**Survived!**")
        if confidence is not None:
            st.write(f"Confidence score: {confidence * 100:.2f}%")
    else:
        st.error("**Did Not Survive**")
        if confidence is not None:
            st.write(f"Confidence score: {confidence * 100:.2f}%")
