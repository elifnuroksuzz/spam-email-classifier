"""
Spam E-posta Sınıflandırıcı - Model Eğitimi ve Optimizasyonu
Bu modül farklı ML algoritmalarını eğitir ve optimize eder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
import joblib
import time
from datetime import datetime

class SpamClassifierTrainer:
    """
    Spam sınıflandırma modeli eğitici sınıfı
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.results = {}
        
        # Model konfigürasyonları
        self.model_configs = {
            'naive_bayes': {
                'model': MultinomialNB(),
                'params': {
                    'alpha': [0.1, 0.5, 1.0, 2.0]
                }
            },
            'logistic_regression': {
                'model': LogisticRegression(random_state=random_state, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'svm': {
                'model': SVC(random_state=random_state, probability=True),
                'params': {
                    'C': [1, 10],  # Daha az parametre
                    'kernel': ['linear']  # Sadece linear kernel
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=random_state),
                'params': {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5]
                }
            }
        }
        
        print("✓ Model eğitici hazırlandı")
    
    def load_data(self):
        """Veriyi yükle ve hazırla"""
        print("📂 Veri yükleniyor...")
        
        # TF-IDF özelliklerini yükle
        X_tfidf = np.load('data/processed/tfidf_features.npy')
        
        # İşlenmiş DataFrame'i yükle
        df = pd.read_csv('data/processed/final_preprocessed_data.csv')
        
        # Ek özellikler (numerical features)
        numeric_features = ['length', 'word_count', 'exclamation_count', 'uppercase_count', 'has_numbers']
        X_numeric = df[numeric_features].values
        
        # TF-IDF ve numeric özellikleri birleştir
        X = np.hstack([X_tfidf, X_numeric])
        y = df['label_encoded'].values
        
        print(f"✓ Veriler yüklendi: {X.shape[0]} örnek, {X.shape[1]} özellik")
        print(f"✓ Sınıf dağılımı: Ham={np.sum(y==0)}, Spam={np.sum(y==1)}")
        
        return X, y, df
    
    def split_data(self, X, y):
        """Veriyi eğitim/test olarak ayır"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        print(f"✓ Veri bölünmesi:")
        print(f"  Eğitim: {X_train.shape[0]} örnek")
        print(f"  Test: {X_test.shape[0]} örnek")
        
        return X_train, X_test, y_train, y_test
    
    def perform_cross_validation(self, model, X_train, y_train, cv_folds=10):
        """K-fold çapraz doğrulama"""
        cv_strategy = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        # Farklı metrikler için CV skorları
        cv_scores = {}
        cv_scores['accuracy'] = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='accuracy')
        cv_scores['precision'] = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='precision')
        cv_scores['recall'] = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='recall')
        cv_scores['f1'] = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1')
        
        return cv_scores
    
    def train_and_optimize_model(self, model_name, X_train, y_train):
        """Modeli eğit ve hiperparametreleri optimize et"""
        print(f"\n🤖 {model_name.upper()} modeli eğitiliyor...")
        
        start_time = time.time()
        
        config = self.model_configs[model_name]
        model = config['model']
        param_grid = config['params']
        
        # Grid Search ile hiperparametre optimizasyonu
        grid_search = GridSearchCV(
            model, 
            param_grid, 
            cv=5,  # 5-fold CV for hyperparameter tuning
            scoring='f1',  # F1-score optimize et
            n_jobs=-1,  # Paralel işlem
            verbose=0
        )
        
        # Fit işlemi
        grid_search.fit(X_train, y_train)
        
        # En iyi modeli al
        best_model = grid_search.best_estimator_
        
        # Çapraz doğrulama skorları
        cv_scores = self.perform_cross_validation(best_model, X_train, y_train)
        
        training_time = time.time() - start_time
        
        # Sonuçları kaydet
        self.models[model_name] = best_model
        self.results[model_name] = {
            'best_params': grid_search.best_params_,
            'best_cv_score': grid_search.best_score_,
            'cv_scores': cv_scores,
            'training_time': training_time
        }
        
        print(f"✓ En iyi parametreler: {grid_search.best_params_}")
        print(f"✓ En iyi CV F1-score: {grid_search.best_score_:.4f}")
        print(f"✓ Eğitim süresi: {training_time:.2f} saniye")
        
        return best_model
    
    def evaluate_model(self, model, model_name, X_test, y_test):
        """Modeli test seti üzerinde değerlendir"""
        print(f"\n📊 {model_name.upper()} değerlendiriliyor...")
        
        # Tahminler
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Metrikler
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # ROC-AUC (eğer probability tahminleri varsa)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        
        # Sonuçları kaydet
        self.results[model_name]['test_metrics'] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        if roc_auc:
            print(f"  ROC-AUC:   {roc_auc:.4f}")
        
        return y_pred, y_pred_proba
    
    def plot_confusion_matrices(self, X_test, y_test):
        """Tüm modeller için confusion matrix grafiği"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, (model_name, model) in enumerate(self.models.items()):
            y_pred = model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Ham', 'Spam'], 
                       yticklabels=['Ham', 'Spam'],
                       ax=axes[i])
            axes[i].set_title(f'{model_name.title()} - Confusion Matrix')
            axes[i].set_ylabel('Gerçek')
            axes[i].set_xlabel('Tahmin')
        
        plt.tight_layout()
        plt.savefig('results/plots/confusion_matrices.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ Confusion matrix grafiği kaydedildi")
    
    def plot_roc_curves(self, X_test, y_test):
        """ROC curve grafiği"""
        plt.figure(figsize=(10, 8))
        
        for model_name, model in self.models.items():
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                roc_auc = roc_auc_score(y_test, y_pred_proba)
                
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name.title()} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.savefig('results/plots/roc_curves.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("✓ ROC curve grafiği kaydedildi")
    
    def create_results_summary(self):
        """Sonuçları özetleyen tablo oluştur"""
        summary_data = []
        
        for model_name, results in self.results.items():
            cv_scores = results['cv_scores']
            test_metrics = results['test_metrics']
            
            summary_data.append({
                'Model': model_name.title(),
                'CV Accuracy': f"{np.mean(cv_scores['accuracy']):.4f} ± {np.std(cv_scores['accuracy']):.4f}",
                'CV F1-Score': f"{np.mean(cv_scores['f1']):.4f} ± {np.std(cv_scores['f1']):.4f}",
                'Test Accuracy': f"{test_metrics['accuracy']:.4f}",
                'Test F1-Score': f"{test_metrics['f1_score']:.4f}",
                'Test Precision': f"{test_metrics['precision']:.4f}",
                'Test Recall': f"{test_metrics['recall']:.4f}",
                'ROC-AUC': f"{test_metrics['roc_auc']:.4f}" if test_metrics['roc_auc'] else "N/A",
                'Training Time': f"{results['training_time']:.2f}s"
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('results/metrics/model_comparison.csv', index=False)
        
        print("\n📊 MODEL KARŞILAŞTIRMA TABLOSU")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def select_best_model(self):
        """En iyi modeli seç"""
        best_f1_score = 0
        best_model_name = None
        
        for model_name, results in self.results.items():
            f1_score = results['test_metrics']['f1_score']
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_model_name = model_name
        
        self.best_model = self.models[best_model_name]
        
        print(f"\n🏆 EN İYİ MODEL: {best_model_name.upper()}")
        print(f"   F1-Score: {best_f1_score:.4f}")
        
        # En iyi modeli kaydet
        joblib.dump(self.best_model, 'models/best_spam_classifier.joblib')
        print("✓ En iyi model kaydedildi: models/best_spam_classifier.joblib")
        
        return best_model_name, self.best_model

def main():
    """Ana eğitim fonksiyonu"""
    print("🤖 MODEL EĞİTİMİ VE OPTİMİZASYONU")
    print("=" * 50)
    print(f"Başlangıç zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Trainer'ı başlat
    trainer = SpamClassifierTrainer()
    
    # 2. Veriyi yükle
    X, y, df = trainer.load_data()
    
    # 3. Veriyi böl
    X_train, X_test, y_train, y_test = trainer.split_data(X, y)
    
    # 4. Tüm modelleri eğit
    print("\n🎯 MODEL EĞİTİM SÜRECİ")
    print("=" * 30)
    
    model_names = ['naive_bayes', 'logistic_regression', 'random_forest']  # SVM atlandı
    
    for model_name in model_names:
        trainer.train_and_optimize_model(model_name, X_train, y_train)
        trainer.evaluate_model(trainer.models[model_name], model_name, X_test, y_test)
    
    # 5. Görselleştirmeler
    print("\n📈 GRAFİKLER OLUŞTURULUYOR...")
    trainer.plot_confusion_matrices(X_test, y_test)
    trainer.plot_roc_curves(X_test, y_test)
    
    # 6. Sonuç tablosu
    summary_df = trainer.create_results_summary()
    
    # 7. En iyi modeli seç
    best_model_name, best_model = trainer.select_best_model()
    
    # 8. Sonuç özeti
    print(f"\n✅ MODEL EĞİTİMİ TAMAMLANDI!")
    print(f"🏆 En iyi model: {best_model_name.upper()}")
    print(f"📁 Sonuçlar: results/metrics/model_comparison.csv")
    print(f"📊 Grafikler: results/plots/")
    print(f"💾 En iyi model: models/best_spam_classifier.joblib")
    print(f"⏱️ Bitiş zamanı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return trainer, best_model

if __name__ == "__main__":
    trainer, best_model = main()