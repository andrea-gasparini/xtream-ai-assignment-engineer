from src.constants import RANDOM_STATE, TEST_SIZE, Y_COLUMN, DATASET_RAW_FILE

from sklearn.model_selection import train_test_split as sklearn_train_test_split
from typing import Tuple, List

import pandas as pd  

    
class DiamondsDataset:
    """
    A class to represent the diamonds dataset.

    The dataset is loaded from a Pandas DataFrame and preprocessed,
    including cleaning and ordinal encoding of the categorical features.

    See `from_csv` function to create a DiamondsDataset object from a CSV file.

    Parameters
    ----------
    data : sequence of pd.DataFrame
        The data to load.

    remove_duplicates : bool, default=True
        Whether to remove duplicated rows.

    Attributes
    ----------
    CUT_GRADE_SCALE : tuple
        The scale of cut grades for diamonds:
        'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'

    CLARITY_SCALE : tuple
        The scale of clarity grades for diamonds:
        'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'

    COLOR_GRADING_SCALE : tuple
        The scale of color grades for diamonds:
        'Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P',
        'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F', 'E', 'D'

    CUT_GRADE_ENCODER : dict
        A dictionary mapping cut grades to ascending ordinal values for encoding.

    CLARITY_ENCODER : dict
        A dictionary mapping clarity grades to ascending ordinal values for encoding.

    COLOR_GRADE_ENCODER : dict
        A dictionary mapping color grades to ascending ordinal values for encoding.

    data : pd.DataFrame
        The diamonds dataset as a Pandas DataFrame.

    X : pd.DataFrame
        The features of the dataset.
        
    y : pd.Series
        The target of the dataset.
    """

    CUT_GRADE_SCALE = 'Fair', 'Good', 'Very Good', 'Premium', 'Ideal'
    CLARITY_SCALE = 'I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'
    COLOR_GRADING_SCALE = ('Z', 'Y', 'X', 'W', 'V', 'U', 'T', 'S', 'R', 'Q', 'P',
                           'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F', 'E', 'D')

    CUT_GRADE_ENCODER = {grade: index for index, grade in enumerate(CUT_GRADE_SCALE)}
    CLARITY_ENCODER = {clarity: index for index, clarity in enumerate(CLARITY_SCALE)}
    COLOR_GRADE_ENCODER = {grade: index for index, grade in enumerate(COLOR_GRADING_SCALE)}

    def __init__(self, *data: pd.DataFrame, remove_duplicates: bool = True) -> None:
        self.data = pd.concat(data)
        self.data = clean_data(self.data, remove_duplicates=remove_duplicates)
        self.__encode_categorical_features()        
        self.X, self.y = features_target_split(self.data)
        
    @property
    def feature_names(self) -> List[str]:
        """Return the feature names of the dataset."""
        return self.X.columns.tolist()
       
    @classmethod
    def from_csv(cls, *paths: str, remove_duplicates: bool = True) -> 'DiamondsDataset':
        """
        Create a DiamondsDataset object from a CSV file.

        Parameters
        ----------
        paths : sequence of str
            The path(s) to the CSV file(s).

        remove_duplicates : bool, default=True
            Whether to remove duplicated rows.

        Returns
        -------
        DiamondsDataset
            A DiamondsDataset object created from the CSV file.
            
        Examples
        --------
        >>> from src.dataset import DiamondsDataset
        >>> dataset = DiamondsDataset.from_csv('datasets/diamonds/diamonds.csv')
        """
        return cls(*[load_csv_data(path) for path in paths], remove_duplicates=remove_duplicates)
        
    @staticmethod
    def train_test_split(*data: pd.DataFrame,
                         test_size: float = TEST_SIZE,
                         random_state: int = RANDOM_STATE) -> Tuple['DiamondsDataset', 'DiamondsDataset']:
        """
        Split the data into training and testing sets.

        Parameters
        ----------
        data : sequence of pd.DataFrame
            The data to split.

        test_size : float, default=TEST_SIZE
            The proportion of the data to include in the test set.

        random_state : int, default=RANDOM_STATE
            The random seed to use for reproducibility.

        Returns
        -------
        Tuple[DiamondsDataset, DiamondsDataset]
            The training and testing sets.
            
        Examples
        --------
        >>> from src.dataset import DiamondsDataset
        >>> train_set, test_set = DiamondsDataset.train_test_split(pd.read_csv('datasets/diamonds/diamonds.csv'), test_size=0.2)
        """
        train_data, test_data = train_test_split(pd.concat(data), test_size=test_size, random_state=random_state)
        return DiamondsDataset(train_data), DiamondsDataset(test_data)

    def __encode_categorical_features(self) -> None:
        """
        This method encodes the categorical features 'cut', 'clarity', and 'color'
        using the respective encoders: `CUT_GRADE_ENCODER`, `CLARITY_ENCODER`, and
        `COLOR_GRADE_ENCODER`. The encoded values are then assigned back to the
        corresponding columns in the dataset.
        """
        self.data['cut'] = self.data['cut'].map(self.CUT_GRADE_ENCODER)
        self.data['clarity'] = self.data['clarity'].map(self.CLARITY_ENCODER)
        self.data['color'] = self.data['color'].map(self.COLOR_GRADE_ENCODER)
    

def load_csv_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(path)


def clean_data(data: pd.DataFrame, remove_duplicates: bool = False) -> pd.DataFrame:
    """
    Clean the data by removing rows with null values, negative prices, and negative dimensions.
    Optionally remove duplicated rows.

    Parameters
    ----------
    data : pd.DataFrame
        The data to clean.
    remove_duplicates : bool, default=False
        Whether to remove duplicated rows.

    Returns
    -------
    pd.DataFrame
        The cleaned data.
    """
    if remove_duplicates: data.drop_duplicates(inplace=True)
    
    # null values
    data.drop(data[data.isnull().any(axis=1)].index, inplace=True)
    # negative prices
    data.drop(data[data['price'] < 0].index, inplace=True)
    # negative dimensions
    data.drop(data[(data['x'] <= 0) | (data['y'] <= 0) | (data['z'] <= 0)].index, inplace=True)
    
    return data


def features_target_split(data: pd.DataFrame, y_column: str = Y_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split the data into features and target.
    
    Parameters
    ----------
    data : pd.DataFrame
        The data to create the dataset from.
    y_column : str, default=Y_COLUMN
        The name of the target column.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        A tuple containing the features and target as pandas DataFrame and Series respectively.
    """    
    return data.drop(columns=[y_column]), data[y_column]


def train_test_split(data: pd.DataFrame, test_size: float = TEST_SIZE,
               random_state: int = RANDOM_STATE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the data into training and testing sets.

    Parameters
    ----------
    data : pd.DataFrame
        The data to split.

    test_size : float, default=TEST_SIZE
        The proportion of the data to include in the test set.

    random_state : int, default=RANDOM_STATE
        The random seed to use for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The training and testing sets.
    """
    return sklearn_train_test_split(data, test_size=test_size, random_state=random_state)
    