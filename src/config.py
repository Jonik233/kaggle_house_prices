ENV_FILE_PATH = ".env"

TARGET = "SalePrice"

LOG_SCALE_COLS = ["1stFlrSF", "GrLivArea"]

STANDARD_SCALE_COLS = ["YearBuilt", "YearRemodAdd", "LowQualFinSF",  "2ndFlrSF", "1stFlrSF", "GrLivArea"]

COLS_TO_USE = ["YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea", TARGET]

NUMERICAL_COLS_TO_FILL = []

CATEGORICAL_COLS_TO_FILL = []

COLS_TO_ENCODE = []
