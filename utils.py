"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y) and y.nunique() > 10

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    proportions = Y.value_counts(normalize=True)
    return -np.sum(proportions * np.log2(proportions + 1e-10))

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    proportions = Y.value_counts(normalize=True)
    return 1 - np.sum(proportions**2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    impurity_before = entropy(Y) if criterion == "entropy" else gini_index(Y)
    weighted_impurity_after = 0
    
    for value in attr.unique():
        subset = Y[attr == value]
        weight = len(subset) / len(Y)
        impurity_after = entropy(subset) if criterion == "entropy" else gini_index(subset)
        weighted_impurity_after += weight * impurity_after
    
    return impurity_before - weighted_impurity_after


def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -1
    best_attribute = None
    
    for feature in features:
        if check_ifreal(X[feature]):
            thresholds = X[feature].unique()
            for threshold in thresholds:
                left = y[X[feature] <= threshold]
                right = y[X[feature] > threshold]
                gain = information_gain(y, X[feature] <= threshold, criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = (feature, threshold)
        else:
            gain = information_gain(y, X[feature], criterion)
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
    
    return best_attribute

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.
    if isinstance(value, (int, float)): 
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
        return (X[left_mask], y[left_mask]), (X[right_mask], y[right_mask])
    else:
        mask = X[attribute] == value
        return (X[mask], y[mask]), (X[~mask], y[~mask])