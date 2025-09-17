"""
Spam E-posta Sınıflandırıcı - Proje Kurulumu ve Veri İndirme
Bu script proje klasör yapısını oluşturur ve veri setini indirir.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import shutil
from pathlib import Path

def create_project_structure():
    """
    Proje için gerekli klasör yapısını oluşturur
    """
    # Ana klasörler
    folders = [
        'data/raw',
        'data/processed', 
        'src',
        'models',
        'notebooks',
        'results/plots',
        'results/metrics'
    ]
    
    # Klasörleri oluştur
    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)
        print(f"✓ Klasör oluşturuldu: {folder}")

def download_dataset():
    """
    SMS Spam Collection veri setini indirir ve açar
    """
    # Veri seti URL'si
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "data/raw/smsspamcollection.zip"
    
    try:
        print("Veri seti indiriliyor...")
        urllib.request.urlretrieve(url, zip_path)
        print("✓ Veri seti başarıyla indirildi")
        
        # Zip dosyasını aç
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall("data/raw/")
        print("✓ Veri seti açıldı")
        
        # Zip dosyasını sil
        os.remove(zip_path)
        print("✓ Geçici dosyalar temizlendi")
        
    except Exception as e:
        print(f"❌ Veri seti indirme hatası: {e}")
        return False
    
    return True

def load_and_preview_data():
    """
    Veri setini yükler ve önizleme yapar
    """
    try:
        # SMS Spam Collection veri setini yükle
        # Bu veri seti tab-separated values formatında
        data_path = "data/raw/SMSSpamCollection"
        
        # Veri setini yükle (sütun isimleri olmayan TSV dosyası)
        df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
        
        print("=== VERİ SETİ ÖNİZLEMESİ ===")
        print(f"Toplam örnek sayısı: {len(df)}")
        print(f"Sütunlar: {list(df.columns)}")
        print(f"Sınıf dağılımı:\n{df['label'].value_counts()}")
        print(f"Eksik değer sayısı:\n{df.isnull().sum()}")
        
        print("\n=== İLK 5 ÖRNEK ===")
        print(df.head())
        
        print("\n=== SPAM ÖRNEKLERİ ===")
        spam_examples = df[df['label'] == 'spam'].head(3)
        for idx, row in spam_examples.iterrows():
            print(f"Spam {idx}: {row['message'][:100]}...")
            
        print("\n=== HAM (Normal) ÖRNEKLERİ ===")
        ham_examples = df[df['label'] == 'ham'].head(3)
        for idx, row in ham_examples.iterrows():
            print(f"Ham {idx}: {row['message'][:100]}...")
        
        # Veriyi processed klasörüne kaydet
        df.to_csv('data/processed/spam_dataset.csv', index=False)
        print("\n✓ Veri seti data/processed/ klasörüne kaydedildi")
        
        return df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        return None

def create_config_file():
    """
    Konfigürasyon dosyasını oluşturur
    """
    config_content = '''"""
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
'''
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(config_content)
    print("✓ config.py dosyası oluşturuldu")

def create_requirements_file():
    """
    requirements.txt dosyasını oluşturur
    """
    requirements = '''pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
matplotlib>=3.5.0
seaborn>=0.11.0
nltk>=3.7
wordcloud>=1.8.0
joblib>=1.1.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    print("✓ requirements.txt dosyası oluşturuldu")

def main():
    """
    Ana fonksiyon - proje kurulumunu gerçekleştirir
    """
    print("🚀 SPAM E-POSTA SINIFLANDIRICI PROJESİ KURULUMU")
    print("=" * 50)
    
    # 1. Klasör yapısını oluştur
    print("\n1. Klasör yapısı oluşturuluyor...")
    create_project_structure()
    
    # 2. Konfigürasyon dosyalarını oluştur
    print("\n2. Konfigürasyon dosyaları oluşturuluyor...")
    create_config_file()
    create_requirements_file()
    
    # 3. Veri setini indir
    print("\n3. Veri seti indiriliyor...")
    if download_dataset():
        # 4. Veri setini yükle ve önizle
        print("\n4. Veri seti önizleniyor...")
        df = load_and_preview_data()
        
        if df is not None:
            print(f"\n🎉 Proje kurulumu tamamlandı!")
            print(f"📊 {len(df)} adet e-posta örneği hazır")
            print(f"📁 Proje klasörü: {os.getcwd()}")
            print(f"\n📋 Sonraki adım: Veri ön işleme ve özellik çıkarımı")
        else:
            print("❌ Veri yükleme başarısız")
    else:
        print("❌ Veri seti indirilemedi")

if __name__ == "__main__":
    main()