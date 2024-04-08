from src.constants import (
    RANDOM_STATE,
    DEFAULT_HPARAMS,
    DEFAULT_GRID_SEARCH_PARAM_GRID,
    DEFAULT_GRID_SEARCH_CV,
    DEFAULT_GRID_SEARCH_SCORING_STRATEGIES
)
from src.dataset import DiamondsDataset, Dataset

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class DecisionStep:
    """
    A class to represent a decision step taken by a DecisionTree model to reach a prediction.

    Attributes
    ----------
    feature_name : str
        The name of the feature used for splitting.

    feature_value : float
        The value of the feature in the test sample.

    threshold_value : float
        The threshold value used for splitting.
    """
    feature_name: str
    feature_value: float
    threshold_value: float


@dataclass
class PredictionExplanation:
    """
    A class to represent a prediction explanation of a DecisionTree model,
    it contains the predicted price and the decision steps taken to reach the prediction.

    Attributes
    ----------
    predicted_price : float
        The price predicted by the model.

    decision_steps : List[DecisionStep]
        The decision steps taken by the model to reach the prediction.
    """
    predicted_price: float
    decision_steps: List[DecisionStep]


class DiamondPricePredictor:
    """
    A class to represent a diamond price predictor based on the DecisionTreeRegressor model.

    Parameters
    ----------
    random_state : int, default=RANDOM_STATE
        The random seed to use for reproducibility.

    hparams : dict, default=DEFAULT_HPARAMS
        The hyperparameters to use for the model.

    Attributes
    ----------
    model : DecisionTreeRegressor
        The actual DecisionTreeRegressor model used for prediction.

    Examples
    --------
    >>> from src.model import DiamondPricePredictor
    >>> from src.dataset import DiamondsDataset
    >>> dataset = DiamondsDataset.from_csv('datasets/diamonds/diamonds.csv')
    >>> model = DiamondPricePredictor()
    >>> model.fit(dataset.train_set)
    >>> predictions = model.predict(dataset.test_set)
    >>> explainations = model.predict_explain(dataset.test_set)
    """

    def __init__(self, random_state: int = RANDOM_STATE, hparams: dict = DEFAULT_HPARAMS) -> None:
        self.model = DecisionTreeRegressor(**hparams, random_state=random_state)
        
    @classmethod
    def from_grid_search_cv(cls, dataset: DiamondsDataset | Dataset,
                            random_state: int = RANDOM_STATE,
                            cv: int = DEFAULT_GRID_SEARCH_CV,
                            param_grid: dict = DEFAULT_GRID_SEARCH_PARAM_GRID,
                            scoring = DEFAULT_GRID_SEARCH_SCORING_STRATEGIES) -> 'DiamondPricePredictor':
        """
        Perform a grid search cross-validation to find the best hyperparameters for the model.

        Parameters
        ----------
        dataset : DiamondsDataset or Dataset
            The dataset containing the training data.
            If `dataset` is an instance of `DiamondsDataset`,
            the `train_set` attribute will be used as the training data. 
            Otherwise, `dataset` will be used directly.

        random_state : int, default=RANDOM_STATE
            The random seed for reproducibility.

        cv : int, default=DEFAULT_GRID_SEARCH_CV
            The number of folds to use for cross-validation.

        param_grid : dict, default=DEFAULT_GRID_SEARCH_PARAM_GRID
            The hyperparameters grid to search.

        scoring : str or callable or list, default=DEFAULT_GRID_SEARCH_SCORING_STRATEGIES
            The scoring strategy(s) to use for evaluating the model performance during grid search.

        Returns
        -------
        DiamondPricePredictor
            A DiamondPricePredictor object with the best hyperparameters found.

        Raises
        ------
        ValueError
            If the dataset is not a `Dataset` or `DiamondsDataset` object.

        Examples
        --------
        >>> from src.model import DiamondPricePredictor
        >>> from src.dataset import DiamondsDataset
        >>> dataset = DiamondsDataset.from_csv('datasets/diamonds/diamonds.csv')
        >>> model = DiamondPricePredictor.from_grid_search_cv(dataset)
        >>> predictions = model.predict(dataset.test_set)
        >>> explanations = model.predict_explain(dataset.test_set)
        """
        if not isinstance(dataset, Dataset | DiamondsDataset):
            raise ValueError('The dataset must be a Dataset or DiamondsDataset object.')
        
        train_set = dataset.train_set if isinstance(dataset, DiamondsDataset) else dataset
        
        grid_search = GridSearchCV(
            estimator=DecisionTreeRegressor(random_state=random_state),
            param_grid=param_grid,
            scoring=scoring,
            refit=scoring[0],
            cv=cv
        )
        
        grid_search.fit(train_set.X, train_set.y)
        
        instance = cls(hparams=grid_search.best_params_)
        instance.model = grid_search.best_estimator_

        return instance

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
        
    def predict_explain(self, data: DiamondsDataset | Dataset) -> List[PredictionExplanation]:
        """
        Generates a list of predictions and corresponding explanations, 
        containing the decision steps leading to the prediction.

        Parameters
        ----------
        data : DiamondsDataset or Dataset
            The dataset containing the test data.
            If `data` is an instance of `DiamondsDataset`,
            the `X` attribute of `test_set` will be used.
            Otherwise, `X` will be used directly.

        Returns
        -------
        List[PredictionExplanation]:
            A list of PredictionExplanation objects, each containing the predicted price
            and the decision steps leading to the prediction.
        """
        if not isinstance(data, Dataset | DiamondsDataset):
            raise ValueError('The dataset must be a Dataset or DiamondsDataset object.')
        
        test_set_X = data.test_set.X if isinstance(data, DiamondsDataset) else data.X
    
        # features used for splitting of each node
        features = self.model.tree_.feature
        # threshold values of each node
        thresholds = self.model.tree_.threshold
        # decision paths of each test sample
        decision_paths = self.model.decision_path(test_set_X)
        # leaf node ids reached by each sample
        leaf_ids = self.model.apply(test_set_X)
        
        explainations: List[PredictionExplanation] = list()
        
        for i in range(len(test_set_X)):

            test_sample_node_ids = decision_paths.indices[decision_paths.indptr[i]:
                                                          decision_paths.indptr[i + 1]]
            
            decision_steps: List[DecisionStep] = list()
            
            for node_id in test_sample_node_ids:

                # if have not reached the leaf node
                if leaf_ids[i] != node_id:
                    
                    decision_steps.append(DecisionStep(
                        feature_name=data.feature_names[features[node_id]],
                        feature_value=test_set_X.iat[i, features[node_id]],
                        threshold_value=thresholds[node_id]
                    ))
                         
            explainations.append(PredictionExplanation(
                predicted_price=self.model.predict(test_set_X.iloc[i:i + 1])[0],
                decision_steps=decision_steps
            ))
            
        return explainations


if __name__ == '__main__':
    from src.constants import DATASET_RAW_FILE

    dataset: DiamondsDataset = DiamondsDataset.from_csv(DATASET_RAW_FILE)

    model = DiamondPricePredictor()
    model.fit(dataset)
    predictions = model.predict(dataset.test_set.X)
    # print(predictions)
    
    explainations = model.predict_explain(dataset.test_set)
    # print(explainations[0])
    
    assert model.model.get_depth() == DiamondPricePredictor.from_grid_search_cv(dataset).model.get_depth()