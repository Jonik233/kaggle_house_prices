####### Dataset Version: V3.1 #######


# Path to env file with dir keys
ENV_FILE_PATH = ".env"

# Label of a target variable
TARGET = "SalePrice"

# Columns that will be scaled using log1p
LOG_SCALE_COLUMNS = ["1stFlrSF", "GrLivArea"]

# Columns that will be scaled using StandardScaler
STANDARD_SCALE_COLUMNS = ["YearBuilt", "YearRemodAdd", "LowQualFinSF",  "2ndFlrSF", "1stFlrSF", "GrLivArea",
                          "FullBath", "HalfBath", "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd"]

# Columns the values of which will be one-hot encoded
COLUMNS_TO_ENCODE = ["MSSubClass", "BldgType", "HouseStyle", "Functional", "KitchenQual"]

# Categories to encode in MSSubClass column
MSSubClass_CATEGORIES_TO_ENCODE = [20, 50, 60, 120]

# Categories to encode in BldgType column
BldgType_CATEGORIES_TO_ENCODE = ["1Fam", "TwnhsE"]

# Categories to encode in HouseStyle column
HouseStyle_CATEGORIES_TO_ENCODE = ["1Story", "2Story"]

# Categories to encode in Functional column
Functional_CATEGORIES_TO_ENCODE = ["Typ"]

# Categories to encode KitchenQual column
KitchenQual_CATEGORIES_TO_ENCODE = ["TA", "Gd", "Ex", "Fa"]

# Columns of numerical type that will be used during training
NUMERICAL_COLUMNS_TO_USE = ["YearBuilt", "YearRemodAdd", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
                            "BedroomAbvGr"]

# Columns of categorical type that will be used during training
CATEGORICAL_COLUMNS_TO_USE = ["OverallQual", "OverallCond",
                             "MSSubClass_20", "MSSubClass_50", "MSSubClass_60", "MSSubClass_120",
                             "BldgType_1Fam", "BldgType_TwnhsE",
                             "HouseStyle_1Story", "HouseStyle_2Story",
                             "Functional_Typ",
                             "KitchenQual_TA", "KitchenQual_Gd", "KitchenQual_Ex", "KitchenQual_Fa"]

# All Columns that will be used during training
COLUMNS_TO_USE = [*NUMERICAL_COLUMNS_TO_USE, *CATEGORICAL_COLUMNS_TO_USE]

NUMERICAL_COLUMNS_TO_FILL = []

CATEGORICAL_COLUMNS_TO_FILL = []