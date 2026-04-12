import streamlit as st
import pandas as pd
import numpy as np
from model import load_data, train_knn, train_random_forest, train_decision_tree
from sklearn.model_selection import train_test_split 

# Load and preprocess data (cached)
@st.cache_data
def get_data():
    return load_data()

X, y = get_data()

# Extract feature names
feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models (cached)
@st.cache_resource
def train_model(model_type):
    if model_type == "K-Nearest Neighbors (KNN)":
        return train_knn(X_train, X_test, y_train, y_test)
    elif model_type == "Random Forest":
        return train_random_forest(X_train, X_test, y_train, y_test)
    elif model_type == "Decision Tree":
        return train_decision_tree(X_train, X_test, y_train, y_test)
    else:
        st.error("Invalid model selected.")
        return None

# Align features
def align_features(user_data, feature_names):
    """
    Align the columns of user_data with the expected feature_names.
    Missing columns are filled with zeros, and extra columns are removed.
    """
    missing_cols = set(feature_names) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0  # Add missing columns with default values
    return user_data[feature_names]  # Reorder columns

# Streamlit App Layout
st.title("Ovarian Cancer Detection Models")

# Sidebar for model selection
st.sidebar.header("Select a Model")
model_choice = st.sidebar.selectbox(
    "Choose a model to train and use:",
    ["K-Nearest Neighbors (KNN)", "Random Forest", "Decision Tree"]
)

# Train the selected model
if st.sidebar.button("Train Model"):
    st.write(f"Training {model_choice}...")
    model = train_model(model_choice)
    if model:
        st.session_state.model = model
        st.session_state.feature_names = feature_names  # Save feature names
        st.success(f"{model_choice} has been trained successfully!")

# Section for predictions
st.header("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for predictions", type="csv")

if uploaded_file:
    try:
        user_data = pd.read_csv(uploaded_file, encoding='utf-8')
        st.write("Uploaded Data Preview:")
        st.write(user_data.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        user_data = None

    if user_data is not None:
        if 'model' in st.session_state and 'feature_names' in st.session_state:
            model = st.session_state.model
            feature_names = st.session_state.feature_names
            try:
                # Align features before prediction
                user_data = align_features(user_data, feature_names)
                predictions = model.predict(user_data)
                st.write("Predictions:")
                st.write(predictions)
            except ValueError as ve:
                st.error(f"Error during prediction: {ve}")
        else:
            st.warning("Please train a model first!")
