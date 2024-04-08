RANDOM_STATE: int = 42

DEFAULT_HPARAMS: dict = { 'max_depth': 9 }
DEFAULT_GRID_SEARCH_PARAM_GRID: dict = { 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 50, 100, 150] }
DEFAULT_GRID_SEARCH_CV: int = 10
DEFAULT_GRID_SEARCH_SCORING_STRATEGIES = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']

TEST_SIZE: float = 0.2
Y_COLUMN: str = 'price'

DATASETS_PATH = 'datasets/diamonds/'
PREPROCESSED_DATASETS_PATH = f'{DATASETS_PATH}preprocessed/'
DATASET_RAW_FILE = f'{DATASETS_PATH}diamonds.csv'