RANDOM_STATE: int = 42

DEFAULT_HPARAMS: dict = { 'max_depth': 9 }

TEST_SIZE: float = 0.2
Y_COLUMN: str = 'price'

DATASETS_PATH = 'datasets/diamonds/'
PREPROCESSED_DATASETS_PATH = f'{DATASETS_PATH}preprocessed/'
DATASET_RAW_FILE = f'{DATASETS_PATH}diamonds.csv'