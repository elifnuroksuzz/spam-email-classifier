"""
Spam E-posta Sınıflandırıcı - Basit Veri Ön İşleme
Hızlı ve etkili metin ön işleme
"""

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import joblib

class SimpleTextPreprocessor:
    """
    Basit ve hızlı metin ön işleyici
    """
    
    def __init__(self):
        # SMS kısaltmaları
        self.abbreviations = {
            'u': 'you', 'ur': 'your', 'r': 'are', 'n': 'and',
            'txt': 'text', 'msg': 'message', 'pls': 'please',
            '2': 'to', '4': 'for', 'luv': 'love', 'wat': 'what'
        }
        print("✓ Basit ön işleyici hazırlandı")
    
    def clean_text(self, text):
        """Temel metin temizleme"""
        if pd.isna(text):
            return ""
        
        # Küçük harf
        text = text.lower()
        
        # Kısaltmaları genişlet
        for abbr, full in self.abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        # Noktalama ve sayıları kaldır
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def add_features(self, df):
        """Basit özellikler ekle"""
        df_copy = df.copy()
        
        # Temel özellikler
        df_copy['length'] = df_copy['message'].str.len()
        df_copy['word_count'] = df_copy['message'].str.split().str.len()
        df_copy['exclamation_count'] = df_copy['message'].str.count('!')
        df_copy['uppercase_count'] = df_copy['message'].str.count(r'[A-Z]')
        df_copy['has_numbers'] = df_copy['message'].str.contains(r'\d').astype(int)
        
        return df_copy

def main():
    """Ana fonksiyon"""
    print("🚀 BASİT VERİ ÖN İŞLEME")
    print("=" * 40)
    
    # 1. Veri yükle
    print("1. Veri yükleniyor...")
    df = pd.read_csv('data/processed/spam_dataset.csv')
    print(f"✓ {len(df)} örnek yüklendi")
    
    # 2. Ön işleyici oluştur
    print("2. Ön işleyici hazırlanıyor...")
    preprocessor = SimpleTextPreprocessor()
    
    # 3. Özellikler ekle
    print("3. Özellikler ekleniyor...")
    df = preprocessor.add_features(df)
    
    # 4. Metinleri temizle
    print("4. Metinler temizleniyor...")
    print("   İşlenen örnekler:")
    
    processed_messages = []
    for i, message in enumerate(df['message']):
        cleaned = preprocessor.clean_text(message)
        processed_messages.append(cleaned)
        
        # İlk 3 örneği göster
        if i < 3:
            print(f"   {i+1}. Orijinal: '{message[:50]}...'")
            print(f"      Temizlenmiş: '{cleaned[:50]}...'")
    
    df['processed_message'] = processed_messages
    
    # 5. Label'ları sayısal hale getir
    print("5. Label'lar kodlanıyor...")
    df['label_encoded'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # 6. TF-IDF özellik çıkarımı
    print("6. TF-IDF özellikleri çıkarılıyor...")
    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),  # 1-gram ve 2-gram
        min_df=2,
        max_df=0.95
    )
    
    # TF-IDF fit et
    tfidf_features = vectorizer.fit_transform(df['processed_message'])
    print(f"✓ {tfidf_features.shape[1]} TF-IDF özelliği çıkarıldı")
    
    # 7. Sonuçları kaydet
    print("7. Veriler kaydediliyor...")
    
    # İşlenmiş DataFrame'i kaydet
    df.to_csv('data/processed/final_preprocessed_data.csv', index=False)
    
    # TF-IDF vectorizer'ı kaydet
    joblib.dump(vectorizer, 'models/tfidf_vectorizer.joblib')
    
    # TF-IDF features'ları kaydet (numpy array olarak)
    np.save('data/processed/tfidf_features.npy', tfidf_features.toarray())
    
    # 8. Özet istatistikler
    print("\n📊 ÖZET İSTATİSTİKLER")
    print("=" * 30)
    print(f"Toplam örnek sayısı: {len(df)}")
    print(f"Ham mesaj sayısı: {len(df[df['label'] == 'ham'])}")
    print(f"Spam mesaj sayısı: {len(df[df['label'] == 'spam'])}")
    print(f"TF-IDF özellik sayısı: {tfidf_features.shape[1]}")
    
    # Özellik istatistikleri
    print(f"\nÖzellik İstatistikleri:")
    numeric_features = ['length', 'word_count', 'exclamation_count', 'uppercase_count', 'has_numbers']
    for feature in numeric_features:
        ham_mean = df[df['label'] == 'ham'][feature].mean()
        spam_mean = df[df['label'] == 'spam'][feature].mean()
        print(f"{feature:20s} | Ham: {ham_mean:6.2f} | Spam: {spam_mean:6.2f}")
    
    # En önemli TF-IDF özellikleri
    print(f"\nEn Yüksek TF-IDF Skorlu Kelimeler:")
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = np.mean(tfidf_features.toarray(), axis=0)
    top_indices = np.argsort(tfidf_scores)[-10:][::-1]
    
    for i, idx in enumerate(top_indices):
        print(f"{i+1:2d}. {feature_names[idx]:15s} ({tfidf_scores[idx]:.4f})")
    
    print(f"\n✅ VERİ ÖN İŞLEME TAMAMLANDI!")
    print(f"📁 İşlenmiş veri: data/processed/final_preprocessed_data.csv")
    print(f"🔧 TF-IDF model: models/tfidf_vectorizer.joblib")
    print(f"📊 TF-IDF özellikler: data/processed/tfidf_features.npy")
    
    print(f"\n🎯 Sonraki adım: Model eğitimi ve değerlendirme")

if __name__ == "__main__":
    main()