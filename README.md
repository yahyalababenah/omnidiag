# مشروع توقّع خطر أمراض القلب

هذا القالب يجهّز بايبلاين تعلّم آلي كامل من التحضير حتى واجهة تفاعلية.

## المتطلبات
- Python 3.10+
- إنشاء بيئة عمل ثم تثبيت المتطلبات:
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

## البيانات
ضع ملف CSV في `data/heart.csv` ويجب أن يحتوي عمود الهدف باسم `target`. يمكن أن تكون بقية الأعمدة رقمية أو فئوية.

## التدريب
```
python src/train.py
```
المخرجات:
- نموذج نهائي في `models/final_model.pkl`
- وصف الأعمدة في `models/metadata.json`
- مقاييس وتقييمات في `results/metrics.json` و `results/cv_scores.csv`
- منحنى ROC في `results/roc_curve.png`

## تحليل غير مُراقب
```
python src/unsupervised.py
```
المخرجات:
- إسقاط PCA في `results/pca_scatter.png`
- نتائج KMeans في `results/kmeans_summary.csv`

## الواجهة
```
cd ui
streamlit run app.py
```
بعد التدريب سيتم التعرّف تلقائيًا على الأعمدة من `metadata.json` لعرض مدخلات ديناميكية.