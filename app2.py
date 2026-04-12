import streamlit as st
import pandas as pd
from io import BytesIO
from model import load_data, train_knn, train_random_forest, train_decision_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Set Streamlit page configuration
st.set_page_config(page_title="Ovarian Cancer Detection", layout="wide", initial_sidebar_state="expanded")

# Cached functions
@st.cache_data
def get_data():
    return load_data()

@st.cache_data
def convert_df_to_csv(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

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

# Load and split data
X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit app layout
st.title("Ovarian Cancer Detection Models")

# Sidebar for model selection
with st.sidebar:
    st.header("Select a Model")
    model_choice = st.selectbox("Choose a model to train and use:", ["K-Nearest Neighbors (KNN)", "Random Forest", "Decision Tree"])
    if st.button("Train Model"):
        model = train_model(model_choice)
        if model:
            st.session_state.model = model
            st.success(f"{model_choice} trained successfully!")
            # Evaluate the model
            y_pred = model.predict(X_test)
            st.write("**Model Performance on Test Data:**")
            st.text(classification_report(y_test, y_pred))

# File uploader for predictions
st.header("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for predictions (include 'TYPE' column for verification)", type="csv")

if uploaded_file:
    try:
        # Read uploaded file
        user_data = pd.read_csv(uploaded_file)
        st.write("**Uploaded Data Preview:**")
        st.write(user_data.head())

        # Ensure 'TYPE' column exists and drop it
        if "TYPE" in user_data.columns:
            user_data = user_data.drop(columns=["TYPE"], errors="ignore")  # Drop 'TYPE' column for predictions

        # Align features with training data
        feature_names = X.columns.tolist()
        missing_cols = set(feature_names) - set(user_data.columns)
        for col in missing_cols:
            user_data[col] = 0
        user_data = user_data[feature_names]  # Align order of columns

        # Ensure predictions are processed with Pandas
        user_data = pd.DataFrame(user_data)  # Ensure it's a DataFrame

        if 'model' in st.session_state:
            model = st.session_state.model
            predictions = model.predict(user_data)
            user_data['Prediction'] = pd.Series(predictions, index=user_data.index)  # Convert predictions to Series

            # Map predictions: 0 -> BOT, 1 -> OC
            user_data['Prediction'] = user_data['Prediction'].map({0: 'BOT', 1: 'OC'})

            # Display predictions
            col1, col2 = st.columns(2)

            with col1:
                st.write("**Rows Predicted as OC (1):**")
                st.write(user_data[user_data['Prediction'] == 'OC'])

            with col2:
                st.write("**Rows Predicted as BOT (0):**")
                st.write(user_data[user_data['Prediction'] == 'BOT'])

            # Download predictions
            csv_data = convert_df_to_csv(user_data)
            st.download_button("Download Predictions", data=csv_data, file_name="predictions.csv", mime="text/csv")
        else:
            st.warning("Please train a model first!")

    except Exception as e:
        st.error(f"Error processing file: {e}")