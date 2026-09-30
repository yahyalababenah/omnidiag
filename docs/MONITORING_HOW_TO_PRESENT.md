# كيف تشغّل نتائج المراقبة وتعرضها

## 0. ما هو حي وما هو محلي
| الشيء | أين | ملاحظة |
|---|---|---|
| مقاييس Prometheus (`/metrics`) | **حي** على https://yahyoha-omnidiag.hf.space/metrics | تشمل الآن زمن الطلب والكاش |
| لوحة Grafana | **محلي فقط** (localhost:3001) | الـ Space حاوية واحدة، لا يشغّل Grafana |
| كشف الانحراف | حي عبر `POST /api/v4/admin/drift/{disease}/run` (يتطلب أدمن) | لا يُنفَّذ تلقائياً |
| الحفظ التلقائي | جهازك: `reports/monitoring_history/` كل 15 دقيقة | مؤقّت systemd |

## 1. عرض سريع بلا تشغيل شيء (30 ثانية)
افتح اللقطات في `reports/monitoring_local/` بالترتيب 01 → 05، واعرض `monitoring_status.json`.

## 2. عرض حي على الـ Space (بدون Docker)
1. افتح https://yahyoha-omnidiag.hf.space/metrics في المتصفح.
2. شغّل تنبؤاً من الواجهة (أو `curl`) ثم أعد تحميل الصفحة: يزيد `omnidiag_predictions_total`.
3. كرّر التنبؤ نفسه: يظهر `cache-hit: true` ويزيد `omnidiag_cache_hits_total`.
4. اعرض `omnidiag_request_duration_seconds_count` (عدد الطلبات لكل مسار وحالة).

## 3. عرض Grafana الكامل (محلي)
```bash
cd ~/Desktop/Projects/Heart_Disease_Project
# 1) الباك إند (SQLite مؤقتة لأن asyncpg غير مثبت)
DATABASE_URL="sqlite+aiosqlite:////tmp/omnidiag_demo.db" \
  backend/.venv/bin/python -m uvicorn backend.main:app --port 7860 &
# 2) Prometheus + Grafana (أوامر التشغيل الدقيقة في monitoring_status.json > environment)
```
ثم:
- Prometheus: http://localhost:9090/targets (يجب أن يكون الهدف `UP`)
- Grafana: http://localhost:3001 — `admin` / `omnidiag_grafana`، اللوحة "OmniDiag — Clinical AI Platform"
- ولّد حركة قبل العرض: نحو 30 طلب `/predict` لكل مرض (الحد 30 طلب/دقيقة). بدون حركة تبدو الرسوم فارغة.

## 4. تشغيل الانحراف وعرضه
```bash
TOK=$(curl -s -X POST localhost:7860/auth/login -H 'Content-Type: application/json' \
  -d '{"email":"admin@omnidiag.com","password":"Admin@123"}' | python3 -c "import sys,json;print(json.load(sys.stdin)['access_token'])")
curl -X POST localhost:7860/api/v4/admin/drift/heart_disease/run -H "Authorization: Bearer $TOK" \
  -H 'Content-Type: application/json' -d '{}'
```
شروط: 10 تنبؤات مسجَّلة على الأقل (القلب وNHANES)، و50 للسكري. التنبؤ المجهول **لا يُخزَّن**، فسجّل الدخول قبل التنبؤ.

## 5. الحفظ التلقائي
- الملفات: `reports/monitoring_history/<وقت UTC>.json` و`latest.json`.
- فحص المؤقّت: `systemctl --user list-timers omnidiag-monitoring-snapshot.timer`
- لقطة فورية: `python3 scripts/monitoring_snapshot.py`
- إيقافه: `systemctl --user disable --now omnidiag-monitoring-snapshot.timer`
- العدّادات تراكمية منذ آخر إعادة تشغيل للـ Space، فقارن اللقطات بالفرق لا بالقيمة المطلقة.

## 6. ما تقوله للجنة (وما لا تقوله)
قل: "المراقبة تعمل: عدّاد التنبؤات، زمن الاستجابة، إصابات الكاش، وانحراف المدخلات بـ KS/chi-square/PSI مع تصحيح Holm."
لا تقل: "النموذج تدهور/لم يتدهور" — الانحراف هنا **مدخلات فقط**، ولا توجد نتائج متابعة لقياس التدهور.
لا تقل: إن Grafana حية على الـ Space؛ هي محلية.
لا تعرض أرقام الانحراف 1.0/0.95 على أنها اكتشاف؛ كانت حركة اصطناعية.
