<h1>Thyroid Cancer Detection Project</h1>

This project focuses on predicting the likelihood of thyroid cancer relapse for survivors based on comprehensive patient data. You can access the thyroid cancer detection web application [here](https://thyroid-cancer-detection-mehmkngabvohisubjwq422.streamlit.app/) to input patient details and receive predictions on the risk of cancer recurrence.

<h3>Objective</h3>

The goal of this project was to develop a machine learning model that accurately predicts whether a thyroid cancer survivor is at risk of cancer relapse. The model classifies patients into two categories: at risk or not at risk based on various health metrics and diagnostic features.

<h3>Approach</h3>

<h4>1. Data Exploration and Preprocessing</h4>

The project began with an extensive exploration and preprocessing of the dataset. This included handling missing values, encoding categorical variables, and scaling numerical features to ensure data quality. Key features were selected based on their relevance to thyroid cancer relapse prediction, including:
<ul>
<li>Age</li>
<li>Gender</li>
<li>Smoking History</li>
<li>Thyroid Function</li>
<li>Physical Examination Results</li>
<li>Pathology Details</li>
<li>Cancer Stage etc</li>
</ul>

<h4>2. Model Selection and Training</h4>

Several classification algorithms were tested and evaluated for performance on this binary classification task. These algorithms included:
<ul>
<li>Random Forest</li>
<li>Support Vector Machine</li>
<li>Logistic Regression</li>
<li>Gradient Boosting</li>
<li>Decision Tree</li>
</ul>

Each model was fine-tuned and assessed to determine the best performer based on metrics such as:
<ul>
<li>Accuracy</li>
<li>Precision</li>
<li>Recall</li>
<li>F1 Score</li>
</ul>
As this was a medical disease detection, recall score was main priority to ensure all actual cancer to predict correctly. 
Cross-validation was employed to ensure the model's robustness and avoid overfitting.

<h4>3. Model Deployment</h4>

The final model, selected based on its superior performance, was deployed as an interactive web application using Streamlit. The app allows users to input various health and diagnostic metrics and receive real-time predictions on the risk of cancer relapse.

<h4>4. Evaluation</h4>

The performance of the model was evaluated on a test dataset using the following metrics:
Accuracy: Overall correctness of predictions.
Precision: Proportion of positive identifications that were actually correct.
Recall: Ability of the model to identify all positive cases.
F1 Score: Balance between precision and recall.

<h3>Key Features</h3>

Interactive Web Application: The model is deployed on a server using Streamlit, providing an easy-to-use interface for inputting patient data and obtaining predictions on cancer relapse risk.
Cross-Validation: Ensured the model's generalizability to unseen data.
Real-Time Prediction: The app offers instant results based on user inputs.

<h3>Conclusion</h3>

This end-to-end thyroid cancer detection project showcases the application of machine learning in oncology. It emphasizes the importance of data preprocessing, model selection, and thorough evaluation, while delivering a practical tool for predicting cancer relapse.

<h3>Technologies Used</h3>
<ul>
<li>Python</li>
<li>Pandas, NumPy for data manipulation</li>
<li>Scikit-learn for machine learning models</li>
<li>Streamlit for deployment</li>
<li>Matplotlib, Seaborn for visualizations</li>
</ul>
