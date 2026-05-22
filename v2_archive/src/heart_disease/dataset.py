import pandas as pd
from pathlib import Path
import logging
from typing import Tuple

# إعداد الـ Logger لتسجيل الأحداث (ممارسة هندسية احترافية)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    """
    تكتشف هذه الدالة المسار الجذري (Root) للمشروع ديناميكياً.
    تصعد 3 مستويات من مكان هذا الملف (src/heart_disease/dataset.py) لتصل للجذر.
    """
    return Path(__file__).resolve().parents[2]

def load_raw_data(filename: str = "heart.csv") -> pd.DataFrame:
    """
    تحميل البيانات الخام من مجلد data/raw.
    
    Args:
        filename (str): اسم ملف البيانات. الافتراضي هو 'heart.csv'.
        
    Returns:
        pd.DataFrame: البيانات الخام كإطار بيانات.
        
    Raises:
        FileNotFoundError: إذا لم يكن الملف موجوداً في المسار المحدد.
    """
    root_dir = get_project_root()
    data_path = root_dir / "data" / "raw" / filename
    
    if not data_path.exists():
        logger.error(f"لم يتم العثور على ملف البيانات في: {data_path}")
        raise FileNotFoundError(f"الملف {filename} غير موجود في {data_path}")
        
    logger.info(f"تم تحميل البيانات بنجاح من: {data_path}")
    df = pd.read_csv(data_path, sep='\t')
    logger.info(f"حجم البيانات المحملة: {df.shape[0]} صف و {df.shape[1]} عمود.")
    
    return df

def get_X_y(df: pd.DataFrame, target_col: str = 'HeartDisease') -> Tuple[pd.DataFrame, pd.Series]:
    """
    فصل إطار البيانات إلى ميزات (X) والهدف (y).
    """
    if target_col not in df.columns:
        logger.error(f"العمود الهدف '{target_col}' غير موجود في البيانات.")
        raise ValueError(f"Target column '{target_col}' not found in the dataset.")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    logger.info(f"تم فصل البيانات بنجاح: مدخلات (X) {X.shape} ، هدف (y) {y.shape}")
    return X, y

if __name__ == "__main__":
    # اختبار سريع للتأكد من عمل الدوال
    try:
        df = load_raw_data()
        X, y = get_X_y(df)
        print("\n--- عينة من الميزات (X) ---")
        print(X.head(2))
    except Exception as e:
        logger.error(f"حدث خطأ أثناء تحميل البيانات: {e}")