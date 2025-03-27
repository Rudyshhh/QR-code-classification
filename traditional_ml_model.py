import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score, 
    precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns

class QRCodeClassifier:
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
    def train_models(self, X_train, y_train):
        """
        Train multiple models with cross-validation
        """
        # Random Forest Hyperparameter Tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        rf_grid = GridSearchCV(
            self.models['random_forest'], 
            rf_params, 
            cv=5, 
            scoring='f1'
        )
        rf_grid.fit(X_train, y_train)
        self.models['random_forest'] = rf_grid.best_estimator_
        
        # SVM Hyperparameter Tuning
        svm_params = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
        svm_grid = GridSearchCV(
            self.models['svm'], 
            svm_params, 
            cv=5, 
            scoring='f1'
        )
        svm_grid.fit(X_train, y_train)
        self.models['svm'] = svm_grid.best_estimator_
        
    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive model evaluation
        """
        results = {}
        
        for name, model in self.models.items():
            y_pred = model.predict(X_test)
            
            results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_recall_fscore_support(y_test, y_pred)[0],
                'recall': precision_recall_fscore_support(y_test, y_pred)[1],
                'f1_score': precision_recall_fscore_support(y_test, y_pred)[2]
            }
            
            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'{name.replace("_", " ").title()} Confusion Matrix')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.tight_layout()
            plt.savefig(f'{name}_confusion_matrix.png')
            plt.close()
        
        return results

# Comprehensive Model Training and Evaluation Script
def main():
    from data_preprocessing import QRCodeDataProcessor
    from sklearn.model_selection import train_test_split
    
    processor = QRCodeDataProcessor('./data')
    X_features, y_labels = processor.feature_extraction()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.2, random_state=42
    )
    
    classifier = QRCodeClassifier()
    classifier.train_models(X_train, y_train)
    results = classifier.evaluate_models(X_test, y_test)
    
    # Print detailed results
    for model_name, metrics in results.items():
        print(f"\n{model_name.replace('_', ' ').title()} Model Results:")
        for metric, value in metrics.items():
            print(f"{metric.replace('_', ' ').title()}: {value}")

if __name__ == "__main__":
    main()