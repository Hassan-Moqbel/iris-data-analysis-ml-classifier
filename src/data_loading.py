# إضافة في بداية الملف
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'Iris.csv')

# ثم استخدام DATA_PATH بدلاً من 'data/Iris.csv'
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import os
from pathlib import Path

def load_iris_data(source='auto'):
    """
    تحميل بيانات Iris من مصادر مختلفة
    """
    # المسارات المحتملة للبيانات
    base_dir = Path(__file__).parent.parent
    possible_paths = [
        base_dir / 'data' / 'Iris.csv',
        base_dir / 'Iris.csv',
        Path('data/Iris.csv'),
        Path('Iris.csv')
    ]
    
    if source == 'auto' or source == 'csv':
        # البحث عن ملف CSV في المواقع المحتملة
        for csv_path in possible_paths:
            if csv_path.exists():
                print(f"📁 تم العثور على البيانات في: {csv_path}")
                try:
                    df = pd.read_csv(csv_path, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(csv_path, encoding="latin1")
                
                # تنظيف البيانات - إزالة عمود Id إذا existed
                if 'Id' in df.columns:
                    df = df.drop('Id', axis=1)
                if 'id' in df.columns:
                    df = df.drop('id', axis=1)
                
                # فصل الميزات والهدف
                X = df.drop('Species', axis=1).values
                
                # تحويل التصنيفات إلى أرقام
                species_mapping = {
                    'Iris-setosa': 0,
                    'Iris-versicolor': 1, 
                    'Iris-virginica': 2
                }
                y = df['Species'].map(species_mapping).values
                
                # تحويل أسماء الميزات إلى التنسيق القياسي
                feature_mapping = {
                    'SepalLengthCm': 'sepal length (cm)',
                    'SepalWidthCm': 'sepal width (cm)',
                    'PetalLengthCm': 'petal length (cm)',
                    'PetalWidthCm': 'petal width (cm)',
                    'sepal length (cm)': 'sepal length (cm)',
                    'sepal width (cm)': 'sepal width (cm)',
                    'petal length (cm)': 'petal length (cm)',
                    'petal width (cm)': 'petal width (cm)'
                }
                
                feature_names = []
                for col in df.columns:
                    if col != 'Species':
                        feature_names.append(feature_mapping.get(col, col))
                
                target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
                
                return X, y, feature_names, target_names
        
        # إذا لم يتم العثور على ملف CSV، استخدام البيانات من scikit-learn
        print("⚠️  لم يتم العثور على ملف CSV، استخدام بيانات scikit-learn")
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names, iris.target_names
    
    elif source == 'sklearn':
        # من مكتبة scikit-learn مباشرة
        iris = load_iris()
        return iris.data, iris.target, iris.feature_names, iris.target_names
    
    else:
        raise ValueError("المصدر المطلوب غير متوفر")

def create_iris_dataframe(source='auto'):
    """
    إنشاء DataFrame من مصدر البيانات المحدد
    """
    X, y, feature_names, target_names = load_iris_data(source)
    
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    df['species'] = df['target'].apply(lambda x: target_names[x])
    
    return df

# دالة مساعدة للتحقق من البيانات
def check_data_files():
    """
    التحقق من وجود ملفات البيانات
    """
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data'
    
    print("🔍 التحقق من ملفات البيانات:")
    
    if not data_dir.exists():
        print(f"❌ مجلد data غير موجود: {data_dir}")
        data_dir.mkdir()
        print(f"✅ تم إنشاء مجلد data: {data_dir}")
    
    csv_files = list(data_dir.glob("*.csv"))
    if csv_files:
        print(f"✅ تم العثور على ملفات CSV: {[f.name for f in csv_files]}")
    else:
        print("❌ لم يتم العثور على أي ملفات CSV في مجلد data")
        
    return csv_files

if __name__ == "__main__":
    check_data_files()
    df = create_iris_dataframe('auto')
    print(f"✅ تم تحميل {len(df)} عينة بنجاح")