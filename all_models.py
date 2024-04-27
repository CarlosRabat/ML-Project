import numpy as np
import random
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.regularizers import l2
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


"""
File contains a function to build the best model for each implementation
 each function is going to return the predictions to evaluate in the
 overall results notebook.

"""


def optimal_MLP():
    """Function return the predictions of the best MLP model we could build

    Returns:
        list: predcitions
    """
    data = np.load("data.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]

    y_test = data["y_test"]
    y_train = np.array(y_train).flatten()

    # Set Random Seeds
    np.random.seed(50)
    random.seed(50)

    # Assuming your classes are labeled as 0 and 1
    class_weights = compute_class_weight(
        "balanced", classes=np.unique(y_train), y=y_train
    )
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    # Create the file structure
    model = Sequential(
        [
            Input(shape=(X_train.shape[1],)),
            Dense(46, activation="relu"),
            Dense(24, activation="relu", kernel_regularizer=l2(0.001)),
            Dense(1, activation="sigmoid"),
        ]
    )

    # Compile the Model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=32,
        class_weight=class_weight_dict,
        shuffle=False,
    )

    # Get the predictions
    y_pred = model.predict(X_test)

    # Change it to binary
    pred = (y_pred > 0.5).astype(int)

    return pred


def optimal_SVM():
    """Function Returns the predictions of the best SVM model we could build

    Returns:
        matrix: predictions
    """

    # Load the data
    data = np.load("data.npz")
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    # TRansorm to a list
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Build the model
    model = SVC(class_weight="balanced", C=1, gamma="scale", kernel="linear")
    model.fit(X_train, y_train)

    # Make the predictions
    y_pred = model.predict(X_test)

    return y_pred


# Cody Write your models here
def optimal_Decision_Tree():
    """_summary_

    Returns:
        _type_: _description_
    """
    return None


def optimal_Logistic_Regression():
    """_summary_

    Returns:
        _type_: _description_
    """
    return None
