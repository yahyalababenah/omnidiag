import sys
import logging
from pathlib import Path
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# إعداد الـ Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- خدعة برمجية للوصول للملفات الأخرى ---
# نضيف المسار الجذري (Root) لبيئة بايثون لكي نتعرف على (dataset و features)
root_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(root_dir))

# الآن يمكننا الاستيراد براحة تامة
from src.heart_disease.dataset import load_raw_data, get_X_y
from src.heart_disease.features import get_preprocessor

def build_classic_stacking_model() -> Pipeline:
    """
    يبني المعمارية الكلاسيكية المستقرة (RF + GB + SVC -> Logistic Regression)
    ويدمجها مع الـ Preprocessor في خط أنابيب واحد.
    """
    logger.info("جاري بناء معمارية Stacking Ensemble الكلاسيكية...")
    
    preprocessor = get_preprocessor()
    
    base_estimators = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svc', SVC(probability=True, random_state=42))
    ]
    meta_learner = LogisticRegression()
    
    classic_stacking = StackingClassifier(
        estimators=base_estimators, 
        final_estimator=meta_learner, 
        cv=5, 
        n_jobs=-1
    )
    
    final_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), 
        ('classifier', classic_stacking)
    ])
    
    return final_pipeline

def evaluate_with_clinical_threshold(pipeline: Pipeline, X_test, y_test, threshold: float = 0.30):
    """
    يقيم النموذج بناءً على العتبة السريرية (مثلاً 30%) لرفع الـ Recall، ويرسم مصفوفة الالتباس.
    """
    logger.info(f"جاري تقييم النموذج بالعتبة السريرية: {threshold*100}%")
    
    # 1. سحب الاحتماليات وتطبيق العتبة
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    y_pred_clinical = (y_probs >= threshold).astype(int)
    
    # 2. طباعة التقرير
    acc = accuracy_score(y_test, y_pred_clinical)
    print(f"\n--- الأداء النهائي مع العتبة السريرية ({threshold*100}%) ---")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_clinical))
    
    # 3. رسم وحفظ مصفوفة الالتباس
    plt.figure(figsize=(6, 5))
    cm_clinical = confusion_matrix(y_test, y_pred_clinical)
    sns.heatmap(cm_clinical, annot=True, fmt='d', cmap='Greens', cbar=False, 
                xticklabels=['Normal (0)', 'Disease (1)'], yticklabels=['Normal (0)', 'Disease (1)'])
    plt.title(f'Classic Stacking + Clinical Threshold ({threshold*100}%)', weight='bold')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # حفظ الصورة بدلاً من عرضها فقط (ممارسة احترافية)
    plot_path = root_dir / "notebooks" / "clinical_confusion_matrix.png"
    plt.savefig(plot_path)
    logger.info(f"تم حفظ رسمة مصفوفة الالتباس في: {plot_path}")
    plt.close()

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    
    try:
        # 1. جلب البيانات
        df = load_raw_data()
        X, y = get_X_y(df)
        
        # 2. تقسيم البيانات (استخدمنا نفس التقسيم السابق 80/20)
        logger.info("جاري تقسيم البيانات إلى 80% تدريب و 20% اختبار...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # 3. بناء وتدريب النموذج
        model_pipeline = build_classic_stacking_model()
        logger.info("بدأ تدريب النموذج. يرجى الانتظار...")
        model_pipeline.fit(X_train, y_train)
        logger.info("تم التدريب بنجاح!")
        
        # 4. التقييم السريري
        evaluate_with_clinical_threshold(model_pipeline, X_test, y_test, threshold=0.30)
        
    except Exception as e:
        logger.error(f"حدث خطأ أثناء تشغيل خط أنابيب التدريب: {e}")