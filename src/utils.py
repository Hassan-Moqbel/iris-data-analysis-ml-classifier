import pickle
import os
from datetime import datetime

def save_model(model, model_name, accuracy, folder='models'):
    """
    حفظ النموذج في ملف
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{folder}/{model_name}_{timestamp}.pkl"
    
    with open(filename, 'wb') as file:
        pickle.dump({
            'model': model,
            'accuracy': accuracy,
            'timestamp': timestamp
        }, file)
    
    print(f"✅ تم حفظ النموذج في {filename}")
    return filename

def load_model(filename):
    """
    تحميل النموذج من ملف
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    
    print(f"✅ تم تحميل النموذج (دقة: {data['accuracy']:.4f})")
    return data

def predict_new_sample(model, features, target_names):
    """
    توقع عينة جديدة
    """
    if len(features) != 4:
        raise ValueError("يجب إدخال 4 قيم (sepal length, sepal width, petal length, petal width)")
    
    prediction = model.predict([features])
    probabilities = model.predict_proba([features])
    
    result = {
        'prediction': target_names[prediction[0]],
        'probabilities': {target_names[i]: prob for i, prob in enumerate(probabilities[0])}
    }
    
    return result

if __name__ == "__main__":
    print("وحدة الأدوات المساعدة - استيرد هذه الدوال في الكود الرئيسي")