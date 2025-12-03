################### DatasetVersion: V1 ###################


# Path to env file with dir keys
ENV_FILE_PATH = ".env"

# Label of a target variable
TARGET = "SalePrice"

# Columns that will be scaled using log1p
LOG_SCALE_COLUMNS = ["1stFlrSF", "GrLivArea"]

# Columns that will be scaled using StandardScaler
STANDARD_SCALE_COLUMNS = ["YearBuilt", "YearRemodAdd", "LowQualFinSF",  "2ndFlrSF", "1stFlrSF", "GrLivArea"]

# All Columns that will be used during training
COLUMNS_TO_USE = ["YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea"]

NUMERICAL_COLUMNS_TO_FILL = []

CATEGORICAL_COLUMNS_TO_FILL = []

COLUMNS_TO_ENCODE = []
