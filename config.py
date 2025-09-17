"""
Spam Classifier Konfigürasyon Dosyası
"""

# Veri yolları
DATA_RAW_PATH = "data/raw/"
DATA_PROCESSED_PATH = "data/processed/"
MODELS_PATH = "models/"
RESULTS_PATH = "results/"

# Model parametreleri
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2
CV_FOLDS = 10

# Metin işleme parametreleri
MAX_FEATURES = 5000
MIN_DF = 2
MAX_DF = 0.95

# Model hiperparametreleri (başlangıç değerleri)
MODELS_CONFIG = {
    "naive_bayes": {
        "alpha": [0.1, 0.5, 1.0, 2.0]
    },
    "svm": {
        "C": [0.1, 1, 10, 100],
        "kernel": ["linear", "rbf"]
    },
    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5, 10]
    }
}
