# CWT-Task
 
Answers to the three main tasks are arranged as follows:

1.	What interesting insights could you draw from this dataset about restaurants and the reviewers? What type of characteristics of the popular restaurants can you learn from the review texts? e.g. cuisine, services etc. Please perform Exploratory Data Analysis techniques to draw the insights.

This exploration is contained in the EDA notebook. 

2.	Could you please design and implement a Machine Learning model to predict the Ratings Review of the restaurants? 

For this task, we trained two models: A **Multinomial Naive Bayes Classifier** using scikit-learn, and a **BERT model** fine-tuned for the task. Modelling code is found in their respective folders. Experiments and metrics were recorded and tracked using MLFlow. Models were assessed based on their accuracy, F1, and ROC AUC scores. 

3.	Please expose the model inference built from step 2 by implementing a RESTful API. The API takes the review_full as request, and returns the predicted rating of a restaurant.

Whilst this can be done in MLFlow, in this project it was implemented using FastAPI. Code can be found in the api folder. The BERT model was used for inference - the model can be added upon request (may require additional steps due to its size.)
