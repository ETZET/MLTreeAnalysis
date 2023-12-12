import numpy as np
import pandas as pd
import openml
from data_pre import standardize_binary_labels
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier,\
    AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import make_scorer, precision_score, recall_score, \
                            f1_score, accuracy_score
from sklearn.model_selection import cross_validate, cross_val_score
import json

def credit_data():
    data = pd.read_csv('credit.csv')
    X, Y = data.iloc[:,:-1], data.iloc[:,-1]
    return X, Y

# def cross_validate(model, X, y, k=5):
#     return cross_val_score(model, X, y, cv=k)

def compare_trees(X, y):
    clf = DecisionTreeClassifier()
    clf = clf.fit(X, y)
    score = cross_val_score(clf, X, y)
    print(score)

    clf = RandomForestClassifier()
    clf = clf.fit(X, y)
    score = cross_val_score(clf, X, y)
    print(score)

    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    score = cross_val_score(clf, X, y)
    print(score)

    clf = AdaBoostClassifier()
    clf = clf.fit(X, y)
    score = cross_val_score(clf, X, y)
    print(score)

    clf = GradientBoostingClassifier() 
    clf = clf.fit(X, y)
    score = cross_val_score(clf, X, y)
    print(score)

def experiment():
    openml.config.apikey = '6adbeab9f7c8a0f9268bb142c3df21f3'  # set the OpenML Api Key
    # SUITE_ID = 336 # Regression on numerical features
    SUITE_ID = 337 # Classification on numerical features
    #SUITE_ID = 335 # Regression on numerical and categorical features
    #SUITE_ID = 334 # Classification on numerical and categorical features
    benchmark_suite = openml.study.get_suite(SUITE_ID)  # obtain the benchmark suite
    results = {}
    i = 0
    for task_id in benchmark_suite.tasks:  # iterate over all tasks
        task = openml.tasks.get_task(task_id)  # download the OpenML task
        dataset = task.get_dataset()
        X, y, categorical_indicator, attribute_names = dataset.get_data(
            dataset_format="dataframe", target=dataset.default_target_attribute
        )
        y_labels = np.unique(y)
        if len(y_labels) != 2:
            print(task_id, "not a binary dataset, skipped")
            continue
        y = standardize_binary_labels(y, y_labels[0], y_labels[1])
        classifiers = [
            ("Decision Tree", DecisionTreeClassifier()),
            ("Random Forest", RandomForestClassifier()),
            ("AdaBoost", AdaBoostClassifier()),
            ("Gradient Boosting", GradientBoostingClassifier())
        ]

        for clf_name, clf in classifiers:
            clf = clf.fit(X, y)
            print(f"{clf_name}")
            scores = model_statistics(X, y, clf)
            # Store results in the 'results' dictionary
            if clf_name not in results:
                results[clf_name] = {}
            results[clf_name][task_id] = scores
            results[clf_name][task_id]['dataset_len'] = len(X)
    
    with open(f'classifier_results_{SUITE_ID}.json', 'w') as json_file:
        json.dump(results, json_file, indent=2)
                  


def model_statistics(X, y, model):
    y_int = y.astype(int)
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='binary', pos_label=1),
        'precision_macro': make_scorer(precision_score, average='binary', pos_label=1),
        'recall_macro': make_scorer(recall_score, average='binary', pos_label=1)
    }
    scores = cross_validate(model, X, y_int, cv=10, scoring = scorers, return_train_score=True)
    return {key: np.mean(scores[key]) for key in scores}
    

if __name__ == "__main__":
    experiment()