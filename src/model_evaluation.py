# src/model_evaluation.py
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, target_names):
    """
    تقييم أداء النموذج
    """
    y_pred = model.predict(X_test)
    
    print("📊 تقرير التصنيف:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    print(f"🎯 دقة النموذج: {accuracy_score(y_test, y_pred):.4f}")
    
    return y_pred

def plot_confusion_matrix(y_test, y_pred, target_names, save_path='data/confusion_matrix.png'):
    """
    رسم مصفوفة الالتباس
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, 
                yticklabels=target_names)
    plt.title('مصفوفة الالتباس')
    plt.ylabel('التصنيف الحقيقي')
    plt.xlabel('التصنيف المتوقع')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
    print(f"✅ تم حفظ مصفوفة الالتباس في {save_path}")

def plot_feature_importance(model, feature_names, save_path='data/feature_importance.png'):
    """
    رسم أهمية الميزات (إذا كان النموذج يدعمها)
    """
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
        indices = np.argsort(importance)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.title('أهمية الميزات')
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        
        print(f"✅ تم حفظ مخطط أهمية الميزات في {save_path}")
    else:
        print("ℹ️ النموذج لا يدعم عرض أهمية الميزات")

# يمكنك إضافة هذا الجزء للاختبار
if __name__ == "__main__":
    print("✅ تم تحميل model_evaluation.py بنجاح!")
    print("الدوال المتاحة:")
    print("- evaluate_model")
    print("- plot_confusion_matrix") 
    print("- plot_feature_importance")