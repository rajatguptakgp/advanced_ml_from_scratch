import pickle
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier

def make_data():
    dataset = load_iris()
    df = pd.DataFrame(dataset['data'], columns = dataset['feature_names'])

    target_names = dataset['target_names']
    target_names_dict = dict(zip(range(len(target_names)), target_names))

    df['target'] = dataset['target']
    df['target'].replace(target_names_dict, inplace = True)
    return df

if __name__ == '__main__':
    # make data
    df = make_data()

    cols = df.columns.tolist()
    cols.remove('target')
    X = df[cols].values
    y = df['target'].values

    # train model
    clf = DecisionTreeClassifier()
    clf.fit(X, y)

    # save model
    with open('saved_model.pkl', 'wb') as file:
        pickle.dump(clf, file)    
