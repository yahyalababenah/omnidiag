import sys
from pathlib import Path
import joblib
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# إعداد مسجل الأحداث
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# تحديد المسار الجذري للمشروع ديناميكياً (نصعد 3 مستويات من هذا الملف)
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))

# استدعاء أدواتنا الخاصة من الملفات التي صنعناها
from src.heart_disease.dataset import load_raw_data
from src.heart_disease.features import get_preprocessor

def train_model():
    # 1. جلب البيانات
    logger.info("جاري تحميل البيانات من المحرك...")
    df = load_raw_data()
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease']

    # 2. تقسيم البيانات (تدريب واختبار) مع الحفاظ على التوازن الطبقي (stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. جلب الـ Pipeline الخاص بالمعالجة
    preprocessor = get_preprocessor()

    # 4. بناء معمارية V2: Stacking Ensemble
    logger.info("جاري بناء معمارية الذكاء المجمّع (Stacking Ensemble)...")
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ]
    
    # النموذج النهائي الذي سيتخذ القرار بناءً على مخرجات النماذج الثلاثة
    stacking_clf = StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(),
        cv=5
    )

    # 5. دمج المعالجة والتدريب في خط إنتاج واحد (Pipeline نهائي)
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', stacking_clf)
    ])

    # 6. بدء التدريب
    logger.info("جاري تدريب النماذج... (قد يستغرق بعض الوقت)")
    final_pipeline.fit(X_train, y_train)

    # 7. التقييم وعرض النتائج
    logger.info("جاري تقييم النموذج...")
    y_pred = final_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"الدقة النهائية (Accuracy): {accuracy * 100:.2f}%")
    logger.info("\n" + classification_report(y_test, y_pred))

    # 8. حفظ النموذج في مجلد models
    model_path = root_dir / "models" / "final_model.pkl"
    joblib.dump(final_pipeline, model_path)
    logger.info(f"تم حفظ النموذج الجاهز بنجاح في: {model_path}")

if __name__ == "__main__":
    train_model()