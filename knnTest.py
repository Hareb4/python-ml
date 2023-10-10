import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import math
from sklearn import preprocessing
from sklearn import tree
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)

@app.route("/",methods=['POST'])
def hello_world():
    # data = request.get_json()
    # input_strings = data['input_strings']
    
    # Call your machine learning model
    result = your_ml_function()
    
    print(result)
    return "<p>Hello, World!</p>"


# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.get_json()
#     input_strings = data['input_strings']
    
#     # Call your machine learning model
#     result = your_ml_function()
    
#     return jsonify({'result': result})


#Import Dataset
raw_data = pd.read_csv('test_no.csv')

#-----------------------Split into testing and training data and Features Encoding----------------------------------------------
x = raw_data.drop(columns = ['Class'])
y = raw_data['Class']

#Features encoding
transformer = ColumnTransformer(transformers=[('cat', OneHotEncoder(), [0, 1, 2, 3, 4])])

x = transformer.fit_transform(x)
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(x, y, test_size = 0.3)
#-------------------------Choosing the best k-----------------------------------------------------------
num_of_rows = len(raw_data)
k = int(math.sqrt(num_of_rows))
#--------------------------Train the knn model----------------------------------------------------------
model = KNeighborsClassifier(n_neighbors = k)
model.fit(x_training_data, y_training_data)
#joblib

predictions = model.predict(x_test_data)
#-------------------------------------------------------------------------------------------------------
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_training_data, y_training_data)
tree_predictions = clf.predict(x_test_data)
#--------------------------Measure the Accuracy---------------------------------------------------------
print("---------------------KNN Classification Report--------------------------")  
print(classification_report(y_test_data, predictions))
print("-----------------------------------------------")
#print(confusion_matrix(y_test_data, predictions))
#print("-----------------------------------------------")
print("---------------------Tree Classification Report--------------------------")  
print(classification_report(y_test_data, tree_predictions))
print("-----------------------------------------------")
#Predict A user given Sample
sample_cols = ['Interest 1', 'Interest 2', 'Interest 3', 'Interest 4', 'Interest 5']
sample_fields = ["Outdoor", "Cultural", "Casual", "Linguistic", "Media"]
sample_data = {}


def your_ml_function():
    for index in range(len(sample_cols)):
        sample_data[sample_cols[index]] = [sample_fields[index]]

    predictTest = transformer.transform(pd.DataFrame(data=sample_data))

    return model.predict(predictTest[0])



