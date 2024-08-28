from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    correct = (y_hat == y).sum()
    total = y.size
    accuracy = correct / total
    
    return accuracy


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    cls = str(cls)
    
    tp = ((y_hat == cls) & (y == cls)).sum()
    fp = ((y_hat == cls) & (y != cls)).sum()
    
    if (tp + fp) == 0:
        return 0.0
    
    precision = tp / (tp + fp)
    
    return precision


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    cls = str(cls)

    tp = ((y_hat == cls) & (y == cls)).sum()
    fn = ((y_hat != cls) & (y == cls)).sum()
    
    if (tp + fn) == 0:
        return 0.0
    
    recall = tp / (tp + fn)
    
    return recall


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    rmse = ((y_hat - y) ** 2).mean()  
    rmse = np.sqrt(mse)
    
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    mae = (y_hat - y).abs().mean()
    
    return mae
