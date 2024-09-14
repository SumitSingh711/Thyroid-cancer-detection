Thyroid Cancer Detection Project

This project focuses on predicting the likelihood of thyroid cancer relapse for survivors based on comprehensive patient data. You can access the thyroid cancer detection web application [here](https://thyroid-cancer-detection-mehmkngabvohisubjwq422.streamlit.app/) to input patient details and receive predictions on the risk of cancer recurrence.

Objective

The goal of this project was to develop a machine learning model that accurately predicts whether a thyroid cancer survivor is at risk of cancer relapse. The model classifies patients into two categories: at risk or not at risk based on various health metrics and diagnostic features.

Approach

Data Exploration and Preprocessing

The project began with an extensive exploration and preprocessing of the dataset. This included handling missing values, encoding categorical variables, and scaling numerical features to ensure data quality. Key features were selected based on their relevance to thyroid cancer relapse prediction, including:
Age
Gender
Smoking History
Thyroid Function
Physical Examination Results
Pathology Details
Cancer Stage
Model Selection and Training

Several classification algorithms were tested and evaluated for performance on this binary classification task. These algorithms included:
Random Forest
Support Vector Machine
Logistic Regression
Gradient Boosting
Decision Tree
Each model was fine-tuned and assessed to determine the best performer based on metrics such as:
Accuracy
Precision
Recall
F1 Score
Cross-validation was employed to ensure the model's robustness and avoid overfitting.
Model Deployment

The final model, selected based on its superior performance, was deployed as an interactive web application using Streamlit. The app allows users to input various health and diagnostic metrics and receive real-time predictions on the risk of cancer relapse.
Evaluation

The performance of the model was evaluated on a test dataset using the following metrics:
Accuracy: Overall correctness of predictions.
Precision: Proportion of positive identifications that were actually correct.
Recall: Ability of the model to identify all positive cases.
F1 Score: Balance between precision and recall.
Key Features

Interactive Web Application: The model is deployed on a server using Streamlit, providing an easy-to-use interface for inputting patient data and obtaining predictions on cancer relapse risk.
Cross-Validation: Ensured the model's generalizability to unseen data.
Real-Time Prediction: The app offers instant results based on user inputs.
Conclusion

This end-to-end thyroid cancer detection project showcases the application of machine learning in oncology. It emphasizes the importance of data preprocessing, model selection, and thorough evaluation, while delivering a practical tool for predicting cancer relapse.

Technologies Used

Python
Pandas, NumPy for data manipulation
Scikit-learn for machine learning models
Streamlit for deployment
Matplotlib, Seaborn for visualizations