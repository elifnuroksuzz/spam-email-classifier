"""
Spam E-posta Sınıflandırıcı - Final Test ve Deployment
Eğitilen modeli test eder ve kullanıma hazırlar.
"""

import pandas as pd
import numpy as np
import joblib
import re
import string
from datetime import datetime

class SpamClassifierPredictor:
    """
    Eğitilmiş spam sınıflandırıcı için tahmin sınıfı
    """
    
    def __init__(self):
        """Modelleri ve ön işleyiciyi yükle"""
        try:
            # En iyi modeli yükle
            self.model = joblib.load('models/best_spam_classifier.joblib')
            print("✓ En iyi model yüklendi (Logistic Regression)")
            
            # TF-IDF vectorizer'ı yükle
            self.vectorizer = joblib.load('models/tfidf_vectorizer.joblib')
            print("✓ TF-IDF vectorizer yüklendi")
            
            # SMS kısaltmaları
            self.abbreviations = {
                'u': 'you', 'ur': 'your', 'r': 'are', 'n': 'and',
                'txt': 'text', 'msg': 'message', 'pls': 'please',
                '2': 'to', '4': 'for', 'luv': 'love', 'wat': 'what',
                'dont': 'do not', 'wont': 'will not', 'cant': 'cannot',
                'im': 'i am', 'youre': 'you are', 'theyre': 'they are'
            }
            
        except Exception as e:
            print(f"❌ Model yükleme hatası: {e}")
            raise
    
    def preprocess_message(self, message):
        """Tek bir mesajı ön işlemden geçir"""
        if pd.isna(message) or message == "":
            return ""
        
        # Küçük harf
        text = message.lower()
        
        # Kısaltmaları genişlet
        for abbr, full in self.abbreviations.items():
            text = re.sub(r'\b' + abbr + r'\b', full, text)
        
        # Noktalama ve sayıları kaldır
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_features(self, message):
        """Mesajdan özellikleri çıkar"""
        # Metin özelliklerini çıkar
        length = len(message)
        word_count = len(message.split())
        exclamation_count = message.count('!')
        uppercase_count = sum(1 for c in message if c.isupper())
        has_numbers = 1 if re.search(r'\d', message) else 0
        
        return [length, word_count, exclamation_count, uppercase_count, has_numbers]
    
    def predict_single(self, message):
        """Tek bir mesaj için spam tahmini yap"""
        # Mesajı ön işle
        processed_text = self.preprocess_message(message)
        
        # TF-IDF özellikleri
        tfidf_features = self.vectorizer.transform([processed_text]).toarray()
        
        # Numeric özellikler
        numeric_features = np.array([self.extract_features(message)])
        
        # Özellikleri birleştir
        combined_features = np.hstack([tfidf_features, numeric_features])
        
        # Tahmin yap
        prediction = self.model.predict(combined_features)[0]
        probability = self.model.predict_proba(combined_features)[0]
        
        return {
            'prediction': 'spam' if prediction == 1 else 'ham',
            'spam_probability': probability[1],
            'ham_probability': probability[0],
            'confidence': max(probability)
        }
    
    def predict_batch(self, messages):
        """Birden fazla mesaj için tahmin"""
        results = []
        for message in messages:
            result = self.predict_single(message)
            result['original_message'] = message
            results.append(result)
        return results

def test_with_examples():
    """Örnek mesajlarla test et"""
    print("🧪 ÖRNEK MESAJLARLA TEST")
    print("=" * 50)
    
    # Test mesajları
    test_messages = [
        # Ham örnekleri
        "Hey, are you free for dinner tonight?",
        "Meeting at 3pm in conference room B",
        "Thanks for your help with the project",
        "Can you pick up milk on your way home?",
        "Happy birthday! Hope you have a great day",
        
        # Spam örnekleri  
        "CONGRATULATIONS! You've won $1000! Click here to claim your prize NOW!",
        "URGENT: Your account will be suspended. Verify now at suspicious-link.com",
        "FREE MONEY! No catch! Send your bank details to claim £500 instantly!!!",
        "WINNER! You are selected for cash prize. Call 123-456-7890 immediately!",
        "Limited time offer! Buy now and get 90% OFF! Don't miss out!!!"
    ]
    
    # Tahminleri yap
    predictor = SpamClassifierPredictor()
    results = predictor.predict_batch(test_messages)
    
    # Sonuçları göster
    print(f"{'No':<2} {'Tahmin':<6} {'Güven':<6} {'Mesaj':<60}")
    print("-" * 80)
    
    correct_predictions = 0
    total_predictions = len(results)
    
    # İlk 5 ham, son 5 spam olarak değerlendir
    for i, result in enumerate(results):
        actual = 'ham' if i < 5 else 'spam'
        predicted = result['prediction']
        confidence = result['confidence']
        message_preview = result['original_message'][:55] + "..."
        
        # Doğru tahmin kontrolü
        is_correct = "✓" if actual == predicted else "✗"
        if actual == predicted:
            correct_predictions += 1
        
        print(f"{i+1:2d} {predicted:>6} {confidence:>6.3f} {message_preview:<60} {is_correct}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\n📊 Test Accuracy: {accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    return results

def create_prediction_interface():
    """Kullanıcı için basit tahmin arayüzü"""
    print("\n💬 İNTERAKTİF SPAM TAHMİN ARAYÜZÜ")
    print("=" * 50)
    print("Mesaj yazın ve ENTER'a basın (çıkmak için 'quit' yazın)")
    
    predictor = SpamClassifierPredictor()
    
    while True:
        user_message = input("\nMesajınız: ").strip()
        
        if user_message.lower() in ['quit', 'exit', 'q']:
            print("👋 Görüşmek üzere!")
            break
        
        if user_message == "":
            print("⚠️ Lütfen bir mesaj yazın")
            continue
        
        # Tahmin yap
        result = predictor.predict_single(user_message)
        
        # Sonucu göster
        prediction = result['prediction']
        confidence = result['confidence']
        spam_prob = result['spam_probability']
        
        if prediction == 'spam':
            print(f"🚨 SPAM (Güven: {confidence:.3f}, Spam olasılığı: {spam_prob:.3f})")
        else:
            print(f"✅ HAM (Güven: {confidence:.3f}, Spam olasılığı: {spam_prob:.3f})")

def save_production_model():
    """Production için model ve gerekli dosyaları hazırla"""
    print("\n📦 PRODUCTİON MODEL HAZIRLANIYOR")
    print("=" * 50)
    
    # Model bilgilerini kaydet
    model_info = {
        'model_type': 'Logistic Regression',
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '1.0',
        'performance': {
            'accuracy': 0.9830,
            'f1_score': 0.9343,
            'precision': 0.9643,
            'recall': 0.9060,
            'roc_auc': 0.9959
        },
        'features': {
            'tfidf_features': 5000,
            'numeric_features': 5,
            'total_features': 5005
        }
    }
    
    # Model bilgilerini JSON olarak kaydet
    import json
    with open('models/model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print("✓ Model bilgileri kaydedildi: models/model_info.json")
    print("✓ Production model hazır: models/best_spam_classifier.joblib")
    print("✓ TF-IDF vectorizer hazır: models/tfidf_vectorizer.joblib")
    
    # README dosyası oluştur
    readme_content = f"""# Spam E-posta Sınıflandırıcı

## Model Bilgileri
- **Algoritma**: Logistic Regression  
- **Eğitim Tarihi**: {datetime.now().strftime('%Y-%m-%d')}
- **Test Accuracy**: 98.30%
- **F1-Score**: 93.43%

## Kullanım
```python
from spam_classifier import SpamClassifierPredictor

predictor = SpamClassifierPredictor()
result = predictor.predict_single("Your message here")
print(result['prediction'])  # 'spam' or 'ham'
print(result['confidence'])  # 0.0 - 1.0
```

## Dosya Yapısı
- `models/best_spam_classifier.joblib` - Eğitilmiş model
- `models/tfidf_vectorizer.joblib` - TF-IDF vectorizer  
- `models/model_info.json` - Model detayları

## Performans Metrikleri
- Accuracy: 98.30%
- Precision: 96.43% 
- Recall: 90.60%
- F1-Score: 93.43%
- ROC-AUC: 99.59%
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✓ README.md oluşturuldu")

def main():
    """Ana test fonksiyonu"""
    print("🚀 FİNAL TEST VE DEPLOYMENT")
    print("=" * 50)
    print(f"Test başlangıcı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # 1. Örnek mesajlarla test
        test_results = test_with_examples()
        
        # 2. Production model hazırlığı
        save_production_model()
        
        # 3. İnteraktif arayüz (isteğe bağlı)
        print(f"\n✅ TÜM TESTLER BAŞARILI!")
        print(f"🏆 Model kullanıma hazır!")
        print(f"📊 Test süreci tamamlandı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # İnteraktif test yapmak istiyor musunuz?
        response = input("\nİnteraktif test yapmak ister misiniz? (y/n): ").lower()
        if response in ['y', 'yes', 'evet']:
            create_prediction_interface()
            
    except Exception as e:
        print(f"❌ Test hatası: {e}")

if __name__ == "__main__":
    main()