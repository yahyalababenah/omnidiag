# OmniDiag — Testing Checklist

> **الهدف**: تغطية شاملة لكل طبقات النظام — من الدوال الصغيرة حتى سلوك الـ API الكامل.  
> **الحالة الراهنة**: الاختبارات الموجودة في `tests/` تغطي Auth + RBAC + Clinical + Patients + Admin + Cache على مستوى HTTP.  
> **ما ينقص**: Unit tests، اختبارات الأمان، اختبارات النماذج، اختبارات Active Learning، والـ Audit log.

---

## الملفات الموجودة (لا تُعاد كتابتها — فقط تُكمَّل)

| الملف | ما يغطيه الآن |
|-------|--------------|
| `tests/test_auth.py` | Register, Login, GetMe, Refresh, Logout |
| `tests/test_rbac.py` | Unauthenticated / Viewer / Doctor / Admin |
| `tests/test_clinical.py` | Predict, Explain, Counterfactuals, Schema, Diseases list |
| `tests/test_patients.py` | Create, List, Search patients |
| `tests/test_admin.py` | Admin endpoints |
| `tests/test_cache.py` | Cache hit/miss |

---

## 1. Unit Tests — ما يجب إنشاؤه

### `tests/test_unit_auth.py`
**المكونات**: `backend/auth/hashing.py` · `backend/auth/jwt.py`

| # | الحالة | الوصف |
|---|--------|-------|
| U-1 | `hash_password` يُنتج hash مختلف عن النص الأصلي | |
| U-2 | `verify_password(plain, hash)` يُرجع `True` للكلمة الصحيحة | |
| U-3 | `verify_password(wrong, hash)` يُرجع `False` | |
| U-4 | `create_access_token` يحتوي على `type=access` في الـ payload | |
| U-5 | `create_refresh_token` يحتوي على `type=refresh` في الـ payload | |
| U-6 | `decode_token` يُرجع الـ payload الصحيح | |
| U-7 | `decode_token` يرفع 401 على token منتهي الصلاحية | |
| U-8 | `decode_token` يرفع 401 على token مزيف | |
| U-9 | استخدام refresh token كـ access token يرفع 401 (token type confusion) | |
| U-10 | `extract_token_from_request` يقرأ من Cookie أولاً ثم Header | |

---

### `tests/test_unit_active_learning.py`
**المكون**: `backend/active_learning/sampler.py`

| # | الحالة | الوصف |
|---|--------|-------|
| U-11 | `prediction_entropy(0.5)` يُرجع `1.0` (أعلى قيمة) | |
| U-12 | `prediction_entropy(0.0)` و `prediction_entropy(1.0)` يُرجعان `0.0` | |
| U-13 | `should_queue_for_review(0.5)` يُرجع `True` | |
| U-14 | `should_queue_for_review(0.99)` يُرجع `False` | |
| U-15 | `uncertainty_band(0.5)` يُرجع `"UNCERTAIN"` | |
| U-16 | `uncertainty_band(0.95)` يُرجع `"CERTAIN"` | |
| U-17 | `uncertainty_band(0.72)` يُرجع `"CONFIDENT"` | |
| U-18 | `uncertainty_band(0.62)` يُرجع `"BORDERLINE"` | |

---

### `tests/test_unit_counterfactuals.py`
**المكون**: `backend/counterfactual_generator.py`

| # | الحالة | الوصف |
|---|--------|-------|
| U-19 | الـ features الـ IMMUTABLE لا تتغير في أي counterfactual ناتج | |
| U-20 | `BMI` في الناتج دائماً بين `15.0` و`50.0` | |
| U-21 | الـ binary features لا تأخذ قيماً كسرية (فقط 0 أو 1) | |
| U-22 | `MentHlth` و`PhysHlth` في النطاق `[0, 30]` | |
| U-23 | الناتج يحتوي على `status=success` و`counterfactuals` كـ list | |

---

### `tests/test_unit_cache.py`
**المكون**: `backend/cache.py`

| # | الحالة | الوصف |
|---|--------|-------|
| U-24 | `predict_cache_key` ينتج مفتاحاً مختلفاً لمدخلات مختلفة | |
| U-25 | `predict_cache_key` ينتج نفس المفتاح لنفس المدخلات | |
| U-26 | `schema_cache_key` يتضمن اسم المرض | |

---

## 2. Integration Tests — ما يجب إكماله

### إضافات على `tests/test_clinical.py`

| # | الحالة | الوصف |
|---|--------|-------|
| I-1 | `/api/v4/diabetes/predict` يعمل بمدخلات صحيحة | |
| I-2 | `/api/v4/diabetes/explain` يُرجع `chart_data` | |
| I-3 | `/api/v4/heart_disease/predict` بمدخلات ناقصة يُرجع 422 | |
| I-4 | `confidence` في الاستجابة دائماً بين `0.0` و`1.0` | |
| I-5 | `prediction` في الاستجابة دائماً `0` أو `1` | |
| I-6 | `diagnosis` في الاستجابة يطابق قيمة `prediction` (Positive/Negative) | |

---

### `tests/test_audit_log.py` — جديد بالكامل

| # | الحالة | الوصف |
|---|--------|-------|
| I-7 | كل طلب مصادق عليه يُضيف صفاً في `audit_logs` | |
| I-8 | الـ `endpoint` المُسجَّل يطابق المسار الفعلي | |
| I-9 | الـ `status_code` المُسجَّل يطابق كود الاستجابة | |
| I-10 | طلب غير مصادق لا يُضيف صفاً في `audit_logs` (أو يُضيف بـ user_id=null) | |
| I-11 | `duration_ms` مسجَّل وقيمته أكبر من صفر | |

---

### `tests/test_active_learning_api.py` — جديد

| # | الحالة | الوصف |
|---|--------|-------|
| I-12 | predict بثقة عالية (`0.99`) لا يُضيف في `review_queue` | |
| I-13 | predict بثقة منخفضة (`~0.5`) يُضيف في `review_queue` | |
| I-14 | GET `/active-learning/queue` يُرجع القائمة للـ doctor | |
| I-15 | PATCH على review item يُغير الـ status | |

---

### إضافات على `tests/test_patients.py`

| # | الحالة | الوصف |
|---|--------|-------|
| I-16 | Soft delete: DELETE مريض لا يحذفه فعلياً — `deleted_at` يُضبط | |
| I-17 | مريض محذوف لا يظهر في list إلا مع `?include_deleted=true` | |
| I-18 | تاريخ تنبؤات المريض `GET /patients/{id}/predictions` | |
| I-19 | Viewer لا يستطيع إنشاء مريض (403) | |

---

## 3. Security Tests — `tests/test_security.py` — جديد

| # | الحالة | الوصف |
|---|--------|-------|
| S-1 | token منتهي الصلاحية يُرجع 401 | |
| S-2 | استخدام refresh token في endpoint يتطلب access token → 401 | |
| S-3 | تعديل payload الـ JWT يُبطل التوقيع → 401 | |
| S-4 | Rate limiting: 6 طلبات login في دقيقة → 429 | |
| S-5 | Security headers موجودة: `X-Content-Type-Options`, `X-Frame-Options` | |
| S-6 | `Content-Security-Policy` header موجود في الاستجابة | |
| S-7 | حقل `search` في `GET /patients/` يعمل بأمان مع محارف خاصة (`' OR 1=1`) | |
| S-8 | Token للـ viewer لا يُقبل في admin endpoints حتى لو عُدِّل يدوياً | |

---

## 4. Database Tests — `tests/test_database.py` — جديد

| # | الحالة | الوصف |
|---|--------|-------|
| D-1 | إنشاء مريضين بنفس الـ MRN يُطلق `IntegrityError` (unique constraint) | |
| D-2 | Soft delete يضبط `deleted_at` دون حذف فعلي | |
| D-3 | حذف user له predictions لا يكسر `predictions` (FK behavior) | |
| D-4 | `Prediction` يحفظ `input_features` كـ JSON قابل للاسترجاع | |
| D-5 | `audit_logs.user_id` يقبل NULL (طلبات بدون auth) | |

---

## 5. ML Model Tests — `tests/test_ml_models.py` — جديد

> **ملاحظة**: هذه الاختبارات تحتاج تحميل النماذج الفعلية — استخدم `pytest.mark.slow` عليها.

| # | الحالة | الوصف |
|---|--------|-------|
| M-1 | `model_loader.py` يُحمِّل النموذج بدون استثناء | |
| M-2 | `predict(valid_input)` يُرجع dict يحتوي `prediction` و`confidence` | |
| M-3 | `confidence` دائماً في `[0.0, 1.0]` | |
| M-4 | `prediction` دائماً `0` أو `1` | |
| M-5 | Ensemble: نتيجة `voting_ensemble` متسقة مع base models على حالة واضحة | |
| M-6 | SHAP: `shap_values` تُحسَب بدون استثناء | |
| M-7 | SHAP: مجموع القيم + `base_value` قريب من `confidence` (±0.1) | |
| M-8 | Snapshot regression: مدخلات معروفة تُنتج نفس النتيجة المحفوظة | |

---

## 6. LLM Report Tests — `tests/test_llm_report.py` — جديد

> تعتمد على mock للـ Anthropic API.

| # | الحالة | الوصف |
|---|--------|-------|
| L-1 | بدون `ANTHROPIC_API_KEY` يُرجع fallback template بدون استثناء | |
| L-2 | الـ fallback يحتوي على `disease_display` و`probability` في النص | |
| L-3 | مع mock للـ API يُرجع النص الذي أعاده Claude | |
| L-4 | `shap_summary` مبني بشكل صحيح من قائمة الـ SHAP values | |

---

## 7. Drift Monitoring Tests — `tests/test_drift.py` — جديد

| # | الحالة | الوصف |
|---|--------|-------|
| V-1 | `DriftMonitor` يُنشأ بدون استثناء | |
| V-2 | `run(current_df)` يُرجع dict يحتوي `drift_detected` و`dataset_drift_score` | |
| V-3 | بدون `evidently` يُعيد رسالة واضحة بدل استثناء غامض | |
| V-4 | `GET /admin/drift/status` يُرجع 200 للـ admin | |
| V-5 | `GET /admin/drift/status` يُرجع 403 للـ viewer | |

---

## إعداد البيئة

```bash
# تشغيل كل الاختبارات
cd /path/to/project
pytest tests/ -v

# بدون الاختبارات البطيئة (ML models)
pytest tests/ -v -m "not slow"

# مع تقرير التغطية
pytest tests/ --cov=backend --cov-report=html

# ملف واحد
pytest tests/test_unit_auth.py -v
```

---

## أهداف التغطية المستهدفة

| الطبقة | التغطية الحالية (تقريبي) | الهدف |
|--------|--------------------------|-------|
| `backend/auth/` | ~60% | **90%** |
| `backend/active_learning/` | 0% | **85%** |
| `backend/counterfactual_generator.py` | 0% | **80%** |
| `backend/monitoring/` | 0% | **70%** |
| `backend/llm/` | 0% | **70%** |
| `backend/patients/` | ~50% | **85%** |
| `backend/middleware/` | ~30% | **75%** |
| **المجموع** | ~40% | **80%** |
