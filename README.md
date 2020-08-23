# Credit Card Fraud Prediction

Trained several models and evaluated how effectively they predict instances of fraud using data based on [this dataset from Kaggle](https://www.kaggle.com/dalpozz/creditcardfraud).
 
* Each row in `fraud_data.csv` corresponds to a credit card transaction. Features include confidential variables `V1` through `V28` as well as `Amount` which is the amount of the transaction. 
* The target is stored in the `class` column, where a value of 1 corresponds to an instance of fraud, and 0 corresponds to an instance of not fraud.

Models:
* Dummy Classifier - strategy = 'most_frequent'
* Support Vector Classifier
* Logistic Regression Classifier

Evaluation Metrics:
* Accuracy
* Precision
* Recall
* Receiver Operating Characteristic
* False Positive Rate
* True Positive Rate

Model Parameters Optimized using GridSearchCV
