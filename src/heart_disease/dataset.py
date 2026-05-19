import pandas as pd
from pathlib import Path
import logging

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

if __name__ == "__main__":
    # هذا الجزء يعمل فقط إذا قمت بتشغيل هذا الملف مباشرة للتجربة
    try:
        df = load_raw_data()
        print(df.head())
    except Exception as e:
        print(f"حدث خطأ أثناء تحميل البيانات: {e}")