import streamlit as st
import pandas as pd
from io import BytesIO
from model import load_data, train_knn, train_random_forest, train_decision_tree
from sklearn.model_selection import train_test_split

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="Ovarian Cancer Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load and preprocess data (cached)
@st.cache_data
def get_data():
    return load_data()

# Convert dataframe to CSV for download
@st.cache_data
def convert_df_to_csv(df):
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
    return buffer.getvalue()

# Align features
def align_features(user_data, feature_names):
    missing_cols = set(feature_names) - set(user_data.columns)
    for col in missing_cols:
        user_data[col] = 0
    return user_data[feature_names]

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

# Load data
X, y = get_data()
feature_names = X.columns.tolist() if isinstance(X, pd.DataFrame) else [f"Feature_{i}" for i in range(X.shape[1])]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Streamlit App Layout with Design
st.title(" Ovarian Cancer Detection Models")

# Sidebar for Model Selection
with st.sidebar:
    st.header("Select a Model")
    model_choice = st.selectbox(
        "Choose a model to train and use:",
        ["K-Nearest Neighbors (KNN)", "Random Forest", "Decision Tree"]
    )
    if st.button("Train Model"):
        st.write(f"Training {model_choice}...")
        model = train_model(model_choice)
        if model:
            st.session_state.model = model
            st.session_state.feature_names = feature_names
            st.success(f"{model_choice} has been trained successfully!")

# Main Prediction Section
st.header("Make Predictions")
uploaded_file = st.file_uploader("Upload a CSV file for predictions", type="csv")

if uploaded_file:
    try:
        user_data = pd.read_csv(uploaded_file, encoding='utf-8')
        st.write("**Uploaded Data Preview:**")
        st.write(user_data.head())
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        user_data = None

    if user_data is not None:
        if 'model' in st.session_state and 'feature_names' in st.session_state:
            model = st.session_state.model
            feature_names = st.session_state.feature_names
            try:
                # Align data
                aligned_data = align_features(user_data, feature_names)

                # Predictions
                predictions = model.predict(aligned_data)
                user_data['Prediction'] = predictions

                # Drop unnecessary features
                columns_to_exclude = [col for col in user_data.columns if "Feature_" in col]
                clean_data = user_data.drop(columns=columns_to_exclude, errors='ignore')

                # Show Results in Two Columns
                col1, col2 = st.columns(2)

                # Display Rows with Predictions = 1
                with col1:
                    st.write("**Rows Predicted as 1:**")
                    rows_with_1 = clean_data[clean_data['Prediction'] == 1]
                    if not rows_with_1.empty:
                        st.write(rows_with_1)
                    else:
                        st.write("No rows were predicted as 1.")

                # Display Rows with Predictions = 0
                with col2:
                    st.write("**Rows Predicted as 0:**")
                    rows_with_0 = clean_data[clean_data['Prediction'] == 0]
                    if not rows_with_0.empty:
                        st.write(rows_with_0)
                    else:
                        st.write("No rows were predicted as 0.")

                # Downloadable Dataset
                csv_data = convert_df_to_csv(clean_data)
                st.download_button(
                    label="Download Data with Predictions",
                    data=csv_data,
                    file_name="predictions.csv",
                    mime="text/csv"
                )

            except ValueError as ve:
                st.error(f"Error during prediction: {ve}")
        else:
            st.warning("Please train a model first!")
