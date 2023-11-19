import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sea
import math
import json
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
import ast


app = Flask(__name__)
cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"


# Call the method to Get all of the puppies


# Import Dataset
raw_data = pd.read_csv("test_no.csv")

# -----------------------Split into testing and training data and Features Encoding----------------------------------------------
x = raw_data.drop(columns=["Class"])
y = raw_data["Class"]

# Features encoding
transformer = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(), [0, 1, 2, 3, 4])]
)

x = transformer.fit_transform(x)
x_training_data, x_test_data, y_training_data, y_test_data = train_test_split(
    x, y, test_size=0.3
)
# -------------------------Choosing the best k-----------------------------------------------------------
num_of_rows = len(raw_data)
k = int(math.sqrt(num_of_rows))
# --------------------------Train the knn model----------------------------------------------------------
model = KNeighborsClassifier(n_neighbors=k)
model.fit(x_training_data, y_training_data)
# joblib

predictions = model.predict(x_test_data)
# -------------------------------------------------------------------------------------------------------
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_training_data, y_training_data)
tree_predictions = clf.predict(x_test_data)
# --------------------------Measure the Accuracy---------------------------------------------------------
print("---------------------KNN Classification Report--------------------------")
print(classification_report(y_test_data, predictions))
print("-----------------------------------------------")
# print(confusion_matrix(y_test_data, predictions))
# print("-----------------------------------------------")
print("---------------------Tree Classification Report--------------------------")
print(classification_report(y_test_data, tree_predictions))
print("-----------------------------------------------")
# Predict A user given Sample
sample_cols = ["Interest 1", "Interest 2", "Interest 3", "Interest 4", "Interest 5"]
sample_fields = ["Outdoor", "Cultural", "Casual", "Linguistic", "Media"]
sample_data = {}


def your_ml_function(samp):
    for index in range(len(sample_cols)):
        sample_data[sample_cols[index]] = [samp[index]]

    predictTest = transformer.transform(pd.DataFrame(data=sample_data))

    predictions = model.predict(predictTest)
    if predictions:
        result = predictions[0]  # Extract the first element from the list
        return result
    else:
        return "No prediction available"


@app.route("/pyml", methods=["GET"])
def hello_world():
    # input_strings = request.get_json()['input_strings']
    # print(input_strings)

    d = {}
    query = request.args["query"]
    selected_hobbies = json.loads(query)
    print(type(selected_hobbies))
    print("hareb")
    answer = str(your_ml_function(selected_hobbies))
    print(answer)
    d["output"] = answer
    d["list"] = selected_hobbies
    # Respond with a success message and HTTP status code 201 (Created)
    return d

if __name__ == "__main__":
    app.run()
