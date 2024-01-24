# Imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras import Input, layers
from tensorflow.keras.models import Sequential
import time
from xgboost import XGBClassifier

# Setup
accuracyData = "./accuracyTraining.csv"
accuracyData2 = "./accuracyTraining(w_oconfidence).csv"
accuracyData3 = "./accuracyTrainingMerge.csv"
finalData = "../Results/All/outputs-all.csv"

sourceData = accuracyData3

modelMap = {
    "NN": {"name": "neural network", "trials": 100},
    "RF": {"name": "random forest", "trials": 500},
    "XG": {"name": "xgboost", "trials": 1000},
    "KN": {"name": "k-nearest neighbors", "trials": 1000},
}


# Data Accuracy
def averageAccuracy(modelType, include, sourceData=sourceData, modelMap=modelMap):
    """
    Description
        - Calculates the average accuracy of the specified machine-learning model 
        in assessing whether the given output for a candidate row is true or false.
    Parameters
        - modelType: a string specifying the type of machine-learning model 
        being trained. “NN” indicates neural network, “RF” indicates random forest, 
        “XG” indicates XGBoost, and “KN” indicates k-nearest neighbors.
        - include: a string that indicates whether to include candidate output 
        rows with no data. If include is set to “all”, then all candidate output 
        rows are included. If include is set to “output”, only the candidate rows 
        with some output present are included.
        - sourceData: a string representing the path to the csv containing the 
        source data. This is set to the global definition by default.
        - modelMap: a dictionary that stores the string abbreviation of 
        machine-learning models as keys, and a dictionary indicating the full 
        name and number of trials to perform for the corresponding model as values. 
        This is set to the global definition by default.
    Return
        - No return value.
    """
    
    # verifies parameters
    assert modelType in ["NN", "RF", "XG", "KN"]
    assert include in ["all", "output"]

    start = time.perf_counter()
    
    # calculates average accuracy of model across trials
    accuracies = []
    for i in range(modelMap[modelType]["trials"]):
        print(f"\ntrial {i+1}\n")
        accuracies.append(accuracy(modelType, include, sourceData))
    avgAccuracy = sum(accuracies) / len(accuracies)
    print(
        f"\naverage accuracy of {modelMap[modelType]['trials']} {modelMap[modelType]['name']}s: {avgAccuracy * 100:.2f}%"
    )

    finish = time.perf_counter()
    print(f"finished accuracy sampling in {finish - start} seconds")


def accuracy(modelType, include, sourceData=sourceData):
    """
    Description
        - Determines the accuracy of the specified machine-learning model in 
        assessing whether the given output for a candidate row is true or false.
    Parameters
        - modelType: a string specifying the type of machine-learning model being 
        trained. “NN” indicates neural network, “RF” indicates random forest, 
        “XG” indicates XGBoost, and “KN” indicates k-nearest neighbors.
        - include: a string that indicates whether to include candidate output 
        rows with no data. If include is set to “all”, then all candidate output 
        rows are included. If include is set to “output”, only the candidate rows 
        with some output present are included.
        - sourceData:  a string representing the path to the csv containing the 
        source data. This is set to the global definition by default.
    Return
        - A float indicating the accuracy of the model.
    """
    
    # verifies parameters
    assert modelType in ["NN", "RF", "XG", "KN"]
    assert include in ["all", "output"]

    # reads the source data
    data = readData(sourceData, include)
    
    # prepares the training data
    X_train, X_test, y_train, y_test = prepareData(data)
    
    # trains the machine-learning model
    model = trainModel(X_train, y_train, modelType)
    
    # evaluates the accuracy of the model
    acc = evaluateModel(model, modelType, X_test, y_test)
    return acc


def readData(file, include):
    """
    Description
        - Reads the csv containing the source data and creates a dataframe that 
        contains the desired information.
    Parameters
        - file: a string representing the path to the csv containing the source 
        data.
        - include: a string that indicates whether to include candidate output 
        rows with no data. If include is set to “all”, then all candidate output 
        rows are included. If include is set to “output”, only the candidate rows 
        with some output present are included.
    Return
        - A pandas dataframe containing the accuracy training data.
    """

    # verifies parameters
    assert include in ["all", "output"]

    # reads all candidate output rows
    if include == "all":
        df = pd.read_csv(file, index_col=None, encoding="latin-1")
        
    # only reads candidate output rows with data
    else:
        df = pd.read_csv(file, index_col=None, encoding="latin-1")
        df = df.drop(
            df[
                (pd.isnull(df["College Major"]))
                & (pd.isnull(df["Undergraduate Institution"]))
                & (pd.isnull(df["Highest Degree and Institution"]))
                & (pd.isnull(df["Work History"]))
            ].index
        )
    return df


def prepareData(data):
    """
    Description
        - Prepares the training data for the machine-learning models by encoding 
        string datatypes, dropping unnecessary columns, creating a train-test 
        split, and scaling the feature data.
    Parameters
        - data: a pandas dataframe containing the accuracy training data.
    Return
        - A list containing the training data for the features, a list containing 
        the testing data for the features, a list of binary values representing 
        the output of the training data, and a list of binary values representing 
        the output of the testing data.
    """
    
    # encodes all columns with string datatype
    label_encoders = {}
    for column in [
        "Name",
        "State",
        "College Major",
        "Undergraduate Institution",
        "Highest Degree and Institution",
        "Work History",
        "Sources",
        "Party",
    ]:
        label_encoders[column] = LabelEncoder()
        data[column] = label_encoders[column].fit_transform(data[column].astype(str))
        
    # drops unnecessary column
    for column in ["Candid"]:
        data = data.drop(column, axis=1)
    
    # defines features and target
    X = data.iloc[:, :-1]
    y = data["Accuracy"]

    # performs train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # scales feature data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


def trainModel(X_train, y_train, modelType):
    """
    Description
        - Trains the specified machine-learning model based on the training data.
    Parameters
        - X_train: a list containing the training data for the features.
        - y_train: a list of binary values representing the training data output.
        - modelType: a string specifying the type of machine-learning model being 
        trained. “NN” indicates neural network, “RF” indicates random forest, 
        “XG” indicates XGBoost, and “KN” indicates k-nearest neighbors.
    Return
        - A machine-learning model as specified by the modelType and trained on 
        the accuracy data. 
    """
    
    # verifies parameters
    assert modelType in ["NN", "RF", "XG", "KN"]
    
    # trains neural network model
    if modelType == "NN":
        model = Sequential(
            [
                Input(shape=X_train.shape[1]),
                layers.Dense(64, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(32, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                layers.Dropout(0.2),
                layers.Dense(1, activation="sigmoid"),
            ]
        )
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        model.fit(X_train, y_train, epochs=35, batch_size=64)

    # trains random forest model
    elif modelType == "RF":
        params = {
            "n_estimators": 100,
            "min_samples_split": 5,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "max_depth": 10,
            "bootstrap": True,
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

    # trains XGBoost model
    elif modelType == "XG":
        params = {
            "learning_rate": 0.05,
            "n_estimators": 100,
            "max_depth": 3,
            "min_child_weight": 3,
            "subsample": 0.7,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1,
            "scale_pos_weight": 1,
            "objective": "binary:logistic",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

    # trains k-nearest neighbors model
    elif modelType == "KN":
        params = {
            "n_neighbors": 7,
            "weights": "uniform",
            "algorithm": "auto",
            "metric": "euclidean",
            "leaf_size": 30,
            "n_jobs": -1,
        }
        model = KNeighborsClassifier(**params)
        model.fit(X_train, y_train)

    return model


def evaluateModel(model, modelType, X_test, y_test):
    """
    Description
        - Evaluates the accuracy of the machine-learning model and prints the 
        test accuracy and classification report (if present).
    Parameters
        - model: a machine-learning model.
        - modelType: a string specifying the type of machine-learning model being 
        trained. “NN” indicates neural network, “RF” indicates random forest, 
        “XG” indicates XGBoost, and “KN” indicates k-nearest neighbors.
        - X_test: a list containing the testing data for the features.
        - y_test: a list of binary values representing the output of the testing 
        data.
    Return
        - A float indicating the accuracy of the model.
    """
    
    # verifies parameters
    assert modelType in ["NN", "RF", "XG", "KN"]

    # evalutes neural network model
    if modelType == "NN":
        loss, accuracy = model.evaluate(X_test, y_test)
        print(f"neural network test accuracy: {accuracy * 100:.2f}%")

    # evaluates random forest model
    elif modelType == "RF":
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"random forest test accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predictions))

    # evaluates XGBoost model
    elif modelType == "XG":
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"xgboost test accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predictions))

    # evaluates k-nearest neighbors model
    elif modelType == "KN":
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"k-nearest neighbors test accuracy: {accuracy * 100:.2f}%")
        print(classification_report(y_test, predictions))

    return accuracy
