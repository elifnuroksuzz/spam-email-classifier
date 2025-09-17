# Spam E-posta S�n�fland�r�c�

## Model Bilgileri
- **Algoritma**: Logistic Regression  
- **E�itim Tarihi**: 2025-09-17
- **Test Accuracy**: 98.30%
- **F1-Score**: 93.43%

## Kullan�m
```python
from spam_classifier import SpamClassifierPredictor

predictor = SpamClassifierPredictor()
result = predictor.predict_single("Your message here")
print(result['prediction'])  # 'spam' or 'ham'
print(result['confidence'])  # 0.0 - 1.0
```

## Dosya Yap�s�
- `models/best_spam_classifier.joblib` - E�itilmi� model
- `models/tfidf_vectorizer.joblib` - TF-IDF vectorizer  
- `models/model_info.json` - Model detaylar�

## Performans Metrikleri
- Accuracy: 98.30%
- Precision: 96.43% 
- Recall: 90.60%
- F1-Score: 93.43%
- ROC-AUC: 99.59%
