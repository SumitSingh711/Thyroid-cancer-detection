import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
import joblib
import os

# Loading data
df = pd.read_csv('dataset.csv')

df.columns = df.columns.str.lower()
df.rename(columns={'hx smoking': 'hx_smoking', 'hx radiothreapy': 'hx_radiothreapy', 'thyroid function': 'thyroid_function', 'physical examination':'physical_examination'}, inplace=True)

X = df.drop('recurred', axis=1)
y = df['recurred']

model = joblib.load('pipeline.joblib')


# web application
st.set_page_config(
    page_title='Thyroid Cancer Predictor',
    page_icon='🧑🏻‍⚕️',
)

st.title("🧑🏻‍⚕ Thyroid Cancer Predictor")

st.subheader("Welcome to the Thyroid Cancer Predictor App")
st.write("""Here, you can assess your risk of Thyroid cancer based on your thyroid checkups data""")


st.subheader("About")
st.info("This application predicts the likelihood of thyroid cancer reoccurance for survivors by providing details such as age, thyroid function, pathology, and other key health indicators."
        "Our app uses advanced algorithms to predict the chances of cancer recurrence. Get personalized insights and take proactive steps towards better health by understanding your risk of thyroid cancer relapse")


st.subheader("Input Features")

age = st.slider(
    "**Age** *(Years)*",
    min_value=1,  # Minimum year
    max_value=100,  # Maximum year
    value=40,      # Default value
)

gender = st.selectbox(
    '**Gender** *(Male=1 or Female=0)*',
    options=X.gender.unique()
)

smoking = st.selectbox(
    "**Smoke** *(Yes=1 or No=0)*",
    options=X.smoking.unique()
)

hx_smoking = st.selectbox(
    '**Smoking History** *(Yes=1 or No=0)*',
    options=X.hx_smoking.unique()
)

hx_radiothreapy = st.selectbox(
    '**History of radiotherapy treatment** *(Yes=1 or No=0)*',
    options=X.hx_radiothreapy.unique()
)

thyroid_function = st.selectbox(
     '**Status of Thyroid function**',
    options=X.thyroid_function.unique()
)

physical_examination = st.selectbox(
    '**Physical examination**',
    options=X.physical_examination.unique()
)

adenopathy = st.selectbox(
    "**Presence or absence of enlarged lymph nodes**",
    options=X.adenopathy.unique()
)

pathology = st.selectbox(
     '**Type of thyroid cancer determined by the pathological examination**',
     options=X.pathology.unique()
)

focality = st.selectbox(
     '**Whether the cancer is unifocal or multifocal**',
     options=X.focality.unique()
)

risk = st.selectbox(
     '**Risk category of the cancer based on various factors**',
     options=X.risk.unique()
)

t = st.selectbox(
     '**Tumor classification based on its size**',
     options=X.t.unique()
)

n = st.selectbox(
     '**Nodal classification indicating the involvement of lymph nodes**',
     options=X.n.unique()
)

m = st.selectbox(
     '**Metastasis classification indicating the presence or absence of distant metastases**',
     options=X.m.unique()
)

stage = st.selectbox(
     '**The overall stage of the cancer**',
     options=X.stage.unique()
)

response = st.selectbox(
     '**Response to treatment**',
     options=X.response.unique()
)

X_new = pd.DataFrame(dict(
	age = [age],
	gender = [gender],
    smoking = [smoking],
    hx_smoking = [hx_smoking],
    hx_radiothreapy = [hx_radiothreapy],
    thyroid_function = [thyroid_function],
    physical_examination = [physical_examination],
	adenopathy	= [adenopathy],
	pathology = [pathology],
	focality=[focality],
    risk=[risk],
	t=[t],
	n=[n],
	m=[m],
	stage=[stage],
    response = [response]
))


# Binary encoding
binary_mappings = {
    'gender': {'M': 1, 'F': 0},
    'smoking': {'Yes': 1, 'No': 0},
    'hx_smoking': {'Yes': 1, 'No': 0},
    'hx_radiothreapy': {'Yes': 1, 'No': 0},
}

for col, mapping in binary_mappings.items():
    X_new[col] = X_new[col].map(mapping)


# Ordinal encoding
ordinal_mappings = {
    'thyroid_function': ['Euthyroid', 'Subclinical Hypothyroidism', 'Clinical Hypothyroidism', 'Subclinical Hyperthyroidism', 'Clinical Hyperthyroidism'],
    'focality': ['Uni-Focal', 'Multi-Focal'],
    'risk': ['Low', 'Intermediate', 'High'],
    't': ['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'],
    'n': ['N0', 'N1a', 'N1b'],
    'm': ['M0', 'M1'],
    'stage': ['I', 'II', 'III', 'IVA', 'IVB']
}

for column, categories in ordinal_mappings.items():
    encoder = OrdinalEncoder(categories=[categories])
    X_new[column] = encoder.fit_transform(X_new[[column]]).astype(np.int8)

X_new.reset_index(drop=True, inplace=True)


# One hot encoding
ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
cols_to_encode = ['physical_examination', 'adenopathy', 'pathology', 'response']

ohe.fit(X[cols_to_encode])
feature_names = ohe.get_feature_names_out(['physical_examination', 'adenopathy', 'pathology', 'response'])

encoded_features_check = ohe.transform(X_new[cols_to_encode])
df_encoded_check = pd.DataFrame(encoded_features_check, columns=feature_names)

X_new = pd.concat([X_new.reset_index(drop=True), df_encoded_check.reset_index(drop=True)], axis=1)
X_new.drop(cols_to_encode, axis=1, inplace=True)

# Scaling numerical values
scaler = StandardScaler()
scaler.fit(X[['age']])
X_new['age'] = scaler.transform(X_new[['age']])


st.subheader('Encoded input')
st.write(X_new)

if st.button('Predict Patient Health'):

    prediction = model.predict(X_new)[0]

    if prediction==1:
        st.markdown("<h4 style='color: red;'>Prediction: The patient is likely to have Thyroid cancer.</h4>", unsafe_allow_html=True)
    else:
        st.markdown("<h4 style='color: green;'>Prediction: The patient is healthy.</h4>", unsafe_allow_html=True)

    # Log the prediction
    # log_prediction(X_new, prediction)

st.markdown(
    """
    <style>
    .css-1d391kg {
        width: 100px;  /* Adjust width as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.subheader('Description of Columns')
description = '''
● **Age:** The age at the time of diagnosis or treatment.\n
● **Gender:** The gender of the patient (male or female).\n
● **Smoking:** Whether the patient is a smoker or not.\n
● **Hx Smoking:** Smoking history of the patient (e.g., whether they have ever smoked).\n
● **Hx Radiotherapy:** History of radiotherapy treatment for any condition.\n
● **Thyroid Function:** The status of thyroid function, possibly indicating if there are any abnormalities.\n
● **Physical Examination:** Findings from a physical examination of the patient.\n
● **Adenopathy:** Presence or absence of enlarged lymph nodes (adenopathy) in the neck region.\n
● **Pathology:** Specific type of thyroid cancer determined by the pathological examination of biopsy samples.\n
● **Focality:** Whether the cancer is unifocal (limited to one location) or multifocal (present in multiple locations).\n
● **Risk:** The risk category of the cancer based on various factors, such as tumor size, extent of spread, and histological type.\n
● **T:** Tumor classification based on its size and extent of invasion into nearby structures.\n
● **N:** Nodal classification indicating the involvement of lymph nodes.\n
● **M:** Metastasis classification indicating the presence or absence of distant metastases.\n
● **Stage:** The overall stage of the cancer, typically determined by combining T, N, and M classifications.\n
● **Response:** Response to treatment, indicating whether the cancer responded positively, negatively, or remained stable after treatment.\n
● **Recurred:** Has the cancer recurred after initial treatment.'''

st.sidebar.markdown(description)
# st.subheader("Prediction Log History")
# st.dataframe(st.session_state.log_df)