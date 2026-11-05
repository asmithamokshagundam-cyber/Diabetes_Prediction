
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺"
)

st.title("AI-Based Diabetes Prediction System")
st.write("Predict diabetes category using health indicators.")

# Step 1: Load the dataset

@st.cache_data
def load_data():
    return pd.read_csv("diabetes_binary_health.csv")

try:
    data = load_data()
except FileNotFoundError:
    st.error("Dataset not found. Check the CSV file location.")
    st.stop()

# Step 2: Select required columns

features = [
    "HighBP",
    "HighChol",
    "BMI",
    "PhysActivity",
    "Age"
]

target = "Diabetes_binary"

if not all(col in data.columns for col in features + [target]):
    st.error("Required columns are missing from the dataset.")
    st.write("Available columns:", list(data.columns))
    st.stop()

data = data[features + [target]].copy()

# Step 3: Clean the data

data[target] = pd.to_numeric(data[target], errors="coerce")
data = data.dropna(subset=[target])
data = data[data[target].isin([0, 1])]

X = data[features]
y = data[target].astype(int)

if y.nunique() < 2:
    st.error("Dataset must contain both diabetes classes.")
    st.stop()

# Step 4: Split the data

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Step 5: Create and train the model

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Step 6: Evaluate the model

predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(
    y_test, predictions, zero_division=0
)
recall = recall_score(
    y_test, predictions, zero_division=0
)
f1 = f1_score(
    y_test, predictions, zero_division=0
)
auc = roc_auc_score(y_test, probabilities)

st.header("Model Evaluation")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy:.2%}")
col2.metric("Precision", f"{precision:.2%}")
col3.metric("Recall", f"{recall:.2%}")

col1, col2 = st.columns(2)

col1.metric("F1-Score", f"{f1:.2%}")
col2.metric("ROC-AUC", f"{auc:.2%}")

# Step 7: Display confusion matrix

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, predictions, labels=[0, 1])

fig, ax = plt.subplots()

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Non-Diabetic", "Diabetic"]
).plot(ax=ax, colorbar=False)

st.pyplot(fig)
plt.close(fig)

# Step 8: Display ROC curve

st.subheader("ROC Curve")

fpr, tpr, thresholds = roc_curve(y_test, probabilities)

fig, ax = plt.subplots()

ax.plot(fpr, tpr, label="Random Forest")
ax.plot([0, 1], [0, 1], linestyle="--")

ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend()

st.pyplot(fig)
plt.close(fig)

# Step 9: Display feature importance

st.subheader("Feature Importance")

importance = model.named_steps[
    "classifier"
].feature_importances_

importance_data = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values("Importance", ascending=False)

st.bar_chart(importance_data.set_index("Feature"))

# Step 10: Get patient details

st.header("Diabetes Prediction")

with st.form("patient_form"):

    high_bp = st.selectbox(
        "High Blood Pressure",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    high_chol = st.selectbox(
        "High Cholesterol",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    bmi = st.number_input(
        "BMI",
        min_value=12.0,
        max_value=98.0,
        value=25.0
    )

    activity = st.selectbox(
        "Physical Activity",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    age = st.slider(
        "Age Category",
        min_value=1,
        max_value=13,
        value=5
    )

    submitted = st.form_submit_button("Predict")

# Step 11: Make prediction

if submitted:

    patient = pd.DataFrame([{
        "HighBP": high_bp,
        "HighChol": high_chol,
        "BMI": bmi,
        "PhysActivity": activity,
        "Age": age
    }])

    result = model.predict(patient)[0]

    probability = model.predict_proba(patient)[0][1] * 100

    st.subheader("Prediction Result")

    if result == 1:
        st.warning("Model predicts the diabetic category.")
    else:
        st.success("Model predicts the non-diabetic category.")

    st.write(f"Estimated probability: {probability:.2f}%")

    st.caption(
        "Educational prediction only. This is not a medical diagnosis."
    )
