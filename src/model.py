from src.constants import RANDOM_STATE
from src.dataset import DiamondsDataset, Dataset

from sklearn.base import RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from typing import Type

import numpy as np
import pandas as pd

class DiamondPricePredictor:
    """
    A class to represent a diamond price predictor.

    Parameters
    ----------
    model_class : Type[RegressorMixin], default=DecisionTreeRegressor
        The class of the model to use for prediction.
        e.g. `LinearRegression`, `DecisionTreeRegressor`, etc.

    random_state : int, default=RANDOM_STATE
        The random seed to use for reproducibility.

    hparams : dict, default={}
        The hyperparameters to use for the model.

    Attributes
    ----------
    model : RegressorMixin
        The model to use for prediction.

    Raises
    ------
    ValueError
        If the model is not a regressor.

    Examples
    --------
    >>> from sklearn.tree import DecisionTreeRegressor
    >>> from src.model import DiamondPricePredictor
    >>> from src.dataset import DiamondsDataset
    >>> dataset = DiamondsDataset.from_csv('datasets/diamonds/diamonds.csv')
    >>> model = DiamondPricePredictor(DecisionTreeRegressor)
    >>> model.fit(dataset.train_set)
    >>> predictions = model.predict(dataset.test_set)
    """

    def __init__(self, model_class: Type[RegressorMixin] = DecisionTreeRegressor,
                 random_state: int = RANDOM_STATE, hparams: dict = {}) -> None:
        if not issubclass(model_class, RegressorMixin):
            raise ValueError('The model must be a regressor.')

        self.model = model_class(**hparams, random_state=random_state)

    def fit(self, dataset: DiamondsDataset | Dataset) -> None:
        """
        Fit the model to the data.

        Parameters
        ----------
        dataset : DiamondsDataset or Dataset
            The dataset containing the training data.
            If `dataset` is an instance of `DiamondsDataset`,
            the `train_set` attribute will be used as the training data. 
            Otherwise, `dataset` will be used directly.
            
        Raises
        ------
        ValueError
            If the dataset is not a `Dataset` or `DiamondsDataset` object.
        """
        if not isinstance(dataset, Dataset | DiamondsDataset):
            raise ValueError('The dataset must be a Dataset or DiamondsDataset object.')
        
        train_set = dataset.train_set if isinstance(dataset, DiamondsDataset) else dataset
        
        self.model.fit(train_set.X, train_set.y)

    def predict(self, data: pd.DataFrame | np.ndarray | Dataset | DiamondsDataset) -> np.ndarray:
        """
        Predict the target variables.

        Parameters
        ----------
        data : pd.DataFrame | np.ndarray | Dataset | DiamondsDataset
            The input data for prediction. It can be a pandas DataFrame,
            numpy array, Dataset, or DiamondsDataset object.

        Returns
        -------
        np.ndarray
            The predicted target variables.

        Raises
        ------
        ValueError
            If the data is not a DataFrame, numpy array, Dataset, or DiamondsDataset object.
        """
        if not isinstance(data, pd.DataFrame | np.ndarray | Dataset | DiamondsDataset):
            raise ValueError('The data must be a DataFrame, numpy array, Dataset, or DiamondsDataset object.')
        
        if isinstance(data, pd.DataFrame | np.ndarray):
            return self.model.predict(data)
        
        if isinstance(data, Dataset):
            return self.model.predict(data.X)
        
        if isinstance(data, DiamondsDataset):
            return self.model.predict(data.test_set.X)

if __name__ == '__main__':
    from src.constants import DATASET_RAW_FILE

    dataset: DiamondsDataset = DiamondsDataset.from_csv(DATASET_RAW_FILE)

    model = DiamondPricePredictor(DecisionTreeRegressor)
    model.fit(dataset)
    predictions = model.predict(dataset.test_set.X)
    print(predictions)