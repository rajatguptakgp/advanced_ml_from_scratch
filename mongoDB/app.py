import pickle
import pymongo
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from flask import Flask, request, render_template

def finetune_model(collection):
    all_records = list(collection.find())

    if len(all_records) % 5 == 0:
        df = pd.DataFrame(all_records)
        X = df[feature_names].values
        y = df['target'].values
        clf.fit(X, y)

        # save updated model
        with open('saved_model.pkl', 'wb') as file:
            pickle.dump(clf, file)    

# getting feature names
dataset = load_iris()
feature_names = dataset['feature_names']

# connecting MongoDB 
connection_string = 'mongodb://localhost:27017'
client = pymongo.MongoClient(connection_string)

# loading MongoDB collection
db = client['iris_db']
collection_name = 'iris_inference'
collection = db[collection_name]

# loading model
with open('saved_model.pkl', 'rb') as file:
    clf = pickle.load(file)

# initialize Flask app
app = Flask(__name__)

# creating endpoints
@app.route('/')
def home(): 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    vals = np.array(features).reshape(1, -1)
    y_pred = clf.predict(vals)[0]

    # key-value pair to insert in MongoDB
    data = dict(zip(feature_names + ['target'], features + [y_pred]))
    collection.insert_one(data)

    # finetuning model with incoming predictions
    finetune_model(collection)
    return render_template('index.html', prediction_text=f'Predicted class: {y_pred}')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3000)