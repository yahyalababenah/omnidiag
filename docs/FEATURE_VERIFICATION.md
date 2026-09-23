# OmniDiag — تحقّق الميزات قبل عرض التحكيم (FEATURE VERIFICATION)

**التاريخ:** 2026-09-23 · **الفرع:** `fix/feature-verification` (17 commit فوق `fix/demo-blockers` @ `c51fead`) · **النوع:** جولة **إصلاح** — كود مُعدَّل، اختبارات جديدة، لا إعادة تدريب، لا تغيير في ملفات النماذج أو العتبات.

هذا التقرير يحلّ محلّ نسخة 2026-09-22. كل صفّ أدناه أُعيد التحقق منه بعد الإصلاح، والدليل مذكور صراحةً. **لا يوجد صفّ مكتوب WORKS بلا دليل**؛ ما لم يُتحقَّق منه مكتوب UNVERIFIED مع السبب في §F.

## البيئات التي اختُبرت فعلياً

| البيئة | الواجهة | الخلفية | ما الذي تحقّق |
|---|---|---|---|
| LOCAL (بناء إنتاجي) | `vite build` + `vite preview` على `127.0.0.1:4173` — **نفس الحزمة التي تُنشر**، لا `npm run dev` | `uvicorn backend.main:app` على `127.0.0.1:8000`، قاعدة SQLite نظيفة، `backend/.venv` (Python 3.13، الإصدارات المثبَّتة في `requirements.txt`) | **57/57 فحص Playwright** + **537 اختبار pytest** + golden master |
| LIVE | لم يُنشر بعد في هذه الجولة | لم يُنشر بعد | — انظر §G |

**الأدوات وأدلّتها:**

| الأداة | ما تغطّيه | المخرجات |
|---|---|---|
| `scratch/ui_verify.py` (Playwright، Chromium 1243) | 57 فحص نقر على البناء الإنتاجي: الصياغة، القائمة المنسدلة، Scales/Randomize/Reset، Compare، Batch (المرضان + السقف + التبديل)، محلل الملاحظات (Apply + العربية)، الملاحظة السريرية، History، طابور Admin | `ui_results.json` + `shots/*.png` |
| `pytest` | 537 اختباراً (كانت 422 — **+115 جديد**) | أدناه |
| `scratch/golden_master.py` | كل مخرجات `/predict` `/explain` `/counterfactuals` `/batch` `/schema` للمرضين | **1474 قيمة، فرقان اثنان فقط** وكلاهما الحقل الإضافي `prevalence_corrected` على `/diseases`. `/predict` و`/explain` **مطابقان تماماً** بسماحية 1e-9 |
| `scripts/warmup_demo_cache.py` | تسخين الكاش لكل المرضى التجريبيين | أدناه |
| `scratch/verify_randomize.mjs` | 500 مريض عشوائي/مرض: داخل حدود الـ schema والنطاق السريري، ومقبولون من `/predict` | أدناه |
| قياس تزامن مباشر | `GET /` أثناء What-If سكري حقيقي غير مخزّن | أدناه |

---

## ⚠️ الحقائق الخمس (مُحدَّثة)

1. **X-3 — مُصلَح ومقيس.** الـ endpoints كانت `async def` وتستدعي كود ML المتزامن مباشرةً، فتجمّد الخادم بالكامل. القياس السابق: `GET /` استغرق **17,981 ms** أثناء دفعة. القياس اليوم على الخادم الفعلي أثناء What-If سكري حقيقي غير مخزّن استغرق **10.16 ث**: 31 فحص `GET /` متوازٍ، **الأسوأ 16.3 ms**، المتوسط 8.4 ms. الآلية: `run_in_threadpool` لـ predict/explain/counterfactuals و`_run_batch_predictions` للدفعات. 4 اختبارات في `tests/test_event_loop_responsiveness.py` **تفشل جميعها على الكود القديم** وتنجح بعده.
2. **X-5 — مُصلَح.** لم يكن أي شيء في المنتج يُنشئ صفّ `Patient` قط، لذا `GET /api/v4/patients/P-001/predictions` كان يعيد **404** — History لم تكن "فارغة"، بل **مستحيلة**. `backend/demo_seed.py` يُنشئ المرضى السبعة و3 نقاط زمنية لكلٍّ منهم عند الإقلاع، بشكل idempotent وفي الخلفية. تحقّق على قاعدة ممسوحة: «Demo history seeded — 21 prediction rows across 7 patients, 3 queued for review»، والنقطة الأحدث لكل مريض تطابق قيمة الشاشة بالضبط (P-001 29.2%، P-002 96.5%، D-003 63.7%، D-004 14.4%).
3. **14 — القيم الخام لم تعد تغادر الخادم.** كان كل تقرير يُرسل أول 20 قيمة مريض خام إلى DeepSeek. كتلة `Patient Features` حُذفت من الـ prompt بالكامل (القلب 11/11، السكري 20/21). ما يزال يُرسَل: المرض، الاحتمال، العتبة، التصنيف، النطاق، وأسماء أعلى 5 ميزات SHAP مع قيمها + مسرد يشرح ما تقيسه كل ميزة. اختبار حيّ على DeepSeek لـ D-004/D-003/P-002: الثلاثة `source: llm`، بلا أي قيمة مقيسة في النص، وبلا "LDL"/"HbA1c".
4. **C-1 — خارج نطاق هذه الجولة بطلب صريح.** كلمات المرور الافتراضية والمستودع العام لم تُمسّ ولم تُفحص. **لا تزال قائمة كما كانت.**
5. **14b — Evidently مثبَّت فعلاً على الـ Space، والرسالة كانت كاذبة.** `requirements.txt:70` فيه `evidently>=0.4.0` والـ Dockerfile يثبّته، لكن `drift.py` يستورد واجهة 0.4 (`ColumnMapping`, `evidently.report.Report`) بينما القيد يحلّ إلى 0.7.23 التي حذفتهما. فرع `except ImportError` كان يكتب "evidently not installed" فيُرسل القارئ يبحث عن حزمة موجودة أصلاً. **لم يُثبَّت شيء**؛ التكلفة موثّقة في [docs/EVIDENTLY_COST.md](EVIDENTLY_COST.md) والرسالة صارت صادقة.

---

## A. جدول الملخص

الحالات: WORKS / PARTIAL / BROKEN / NOT DEPLOYED / OUT OF SCOPE.

| # | الميزة | الحالة السابقة | الحالة الآن | الدليل |
|---|---|---|---|---|
| 1a | Engineering Mode — سكري | UNVERIFIED-carried | **WORKS** | Playwright: Randomize + Scales + Reset + Run Inference على البناء الإنتاجي |
| 1b | Engineering Mode — قلب | WORKS | **WORKS** | نفسه، `ui_results.json` |
| 1c | Scales / Randomize / Reset | PARTIAL | **WORKS** | تذييل Scales صار لكل مرض (القلب: UCI + «Age is in years»؛ السكري: BRFSS + شرح نطاق العمر) — Playwright يؤكّد **غياب BRFSS عن صفحة القلب**. صفر صفوف "Categorical" بلا مدى (فحص صفوف الجدول مباشرةً، 11/11 قلب و21/21 سكري). Randomize: `verify_randomize.mjs`، 500 مريض/مرض داخل الحدود والنطاق السريري، 12/12 مقبولون من `/predict` |
| 1d | القائمة المنسدلة للأمراض | عيب مُرحَّل (مفاتيح خام + "No description") | **WORKS (مُصلَح)** | كانت تقرأ `d.info?.display_name` بينما `DiseaseContext` يسطّح الشكل إلى `{name, ...info}` — كل خيار كان يسقط إلى المفتاح الخام. Playwright: «dropdown shows display names, not raw keys» + «has real descriptions» + «shows a version» |
| 2a | EMR — اختيار المريض + الملخص | WORKS | **WORKS** | Playwright، بلا أخطاء وحدة تحكم |
| 2b | EMR — زر History | **BROKEN** | **WORKS** | سببان أُصلحا معاً: `PatientTimeline.jsx` كان `BASE='/api/v4'` نسبياً (يذهب إلى Vercel ويعيد HTML) → صار `API_BASE`؛ والمرضى لم يكونوا موجودين في القاعدة → X-5. Playwright: «History opens without a JSON parse error» + «shows timeline entries» |
| 3 | Clinical Notes Parser | PARTIAL | **PARTIAL (محدود صراحةً)** | الـ12 ملاحظة أُعيد تشغيلها؛ الأخطاء المنهجية الأربعة زالت (§B.3). العربية **غير مدعومة ويُقال ذلك صراحةً** الآن بدل "0 fields extracted". 33 اختبار جديد |
| 4 | SHAP chart + ملخص التأثير | WORKS | **WORKS** | golden master: `explain` مطابق بايتاً |
| 5a | What-If — العرض | WORKS | **WORKS** | Playwright؛ الصياغة موحّدة الآن عبر `utils/whatIfSummary.js` |
| 5b | What-If — منطق الخلفية | WORKS | **WORKS** | golden master: `counterfactuals` مطابق |
| 5c | What-If — الزمن والكاش | WORKS (بطيء) | **WORKS** | TTL 3600 ث → **43200 ث (12 ساعة)**، قابل للضبط. `warmup_demo_cache.py`: تشغيل أول 60.9 ث (21/21)، تشغيل ثانٍ **5.6 ث** وكل الطلبات "already cached" — What-If السكري من 9–14 ث إلى **0.00 ث** |
| 6 | AI Report (DeepSeek) | WORKS (محتوى مضبوط جزئياً) | **WORKS** | القيم الخام لم تعد تُرسَل؛ "LDL" مُصلَح بمسرد + حارس؛ تناقض Positive/LOW مُصلَح بـ `band_for_report()`. 27 اختبار جديد + اختبار حيّ على 3 مرضى |
| 7 | ملاحظات الطبيب (نص حر) | **NOT IMPLEMENTED** | **WORKS** | عمود `predictions.notes` + `POST /api/v4/patients/{id}/notes` + صندوق «Clinical Note» في EMR + عرضها في History وفي PDF. Playwright: «note saves against the screening» + «the saved note appears in History» |
| 8 | Export PDF + Print | PARTIAL | **WORKS** | PDF صار يحتوي قسم What-If **ورسالة الإحالة** (كانا غائبين كلياً)؛ العنوان «Recommended Lifestyle Interventions» → «Modelled What-If Scenarios»؛ Print لم يعد يُخفي اسم المريض (`data-print-keep`)؛ MRN/Gender/Age تقرأ حقول المريض التجريبي الفعلية |
| 9 | Batch Prediction | UNVERIFIED-this-round | **WORKS** | Playwright بالنقر للمرضين: عمود واحد «Screening result» (لا ازدواج)، سقف 20 صفاً للسكري برسالة تذكر عدد الصفوف، إعادة تعيين النتائج عند تبديل المرض، وعنوان الاحتمال في CSV يتبع `prevalence_corrected` لكل مرض |
| 10 | Patient Comparison (كان Before/After) | UNVERIFIED-this-round | **WORKS (أُعيدت تسميته)** | يقارن مريضين مختلفين، لا قبل/بعد لمريض واحد. أُعيدت التسمية في التنقّل والعنوان والمنتقيات وأعمدة الجدول وشارة الفرق. Playwright للمرضين |
| 11 | Admin — طابور الوسم | PARTIAL (صفوف فارغة، الملاحظات تُهمل) | **WORKS** | الصفوف الفارغة كانت مُصلَحة سلفاً وتأكّد ذلك (0 شرطات في كتلة الطابور). الملاحظات تُخزَّن الآن (`review_queue.notes`). **تحقّق كامل من طرف إلى طرف:** حالة غير مؤكدة دخلت الطابور عند الإقلاع → وُسمت من واجهة Admin → القراءة من القاعدة: `status=reviewed label=1 reviewer=yes notes='Playwright check — borderline, agrees with model.'` |
| 12 | Auth & RBAC | PARTIAL (أمني) | **PARTIAL — X-6 مُصلَح، C-1 خارج النطاق** | الرمز من 15 دقيقة إلى **1440 دقيقة**، قابل للضبط بـ `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`. 13 اختباراً تغطّي حدّ 15/16 دقيقة. **C-1 لم يُمسّ بطلب صريح** |
| 13 | Active learning | PARTIAL | **WORKS** | أعلاه (11) + X-7: التنبؤ المخزّن صار يُسجَّل ويدخل الطابور |
| 14a | Prometheus `/metrics` | live WORKS | **WORKS** | بلا تغيير؛ العدّاد الآن يحسب التنبؤات المُقدَّمة لا استدعاءات النموذج (X-7) |
| 14b | Evidently drift | NOT DEPLOYED | **NOT DEPLOYED (بسبب مُصحَّح)** | مثبَّت لكن واجهته غير متوافقة؛ الرسالة صارت تقول ذلك وتشير إلى تقرير التكلفة. **لم يُثبَّت شيء** |
| 14c | MLflow | NOT DEPLOYED | **NOT DEPLOYED** | بلا تغيير |
| 15a | Dark mode | WORKS | **WORKS** | بلا تغيير |
| 15b | PWA / الأيقونات | PARTIAL | **WORKS** | `favicon.svg` مصدراً، ومنه `favicon.ico` (32) و`apple-touch-icon.png` (180) و192/512 — كلها مُولَّدة بمقاسها لا مُصغَّرة. البناء يحتوي 15 مُدخل precache |
| 15c | الجوال 390px | PARTIAL (547px) | **WORKS** | السبب: صفّ أزرار النموذج 462px بلا التفاف. بعد الإصلاح: `scrollWidth = 390 = innerWidth` على الأوضاع الأربعة |
| 16 | صياغة "Diagnosis" | عيب مُرحَّل | **WORKS** | العنوان الفرعي و`<title>` → «Multi-Disease Clinical Decision Support»؛ 4 رسائل الإقلاع البارد → «the screening service». أسماء حقول الـ API لم تُمسّ |

## مشاكل عابرة للميزات

| ID | الحالة | الدليل |
|---|---|---|
| X-1 | مُصلَح سابقاً | — |
| X-2 | مُصلَح سابقاً | — |
| X-3 | **مُصلَح** | 10.16 ث What-If / أسوأ `GET /` = 16.3 ms (كان 17,981 ms) |
| X-4 | **مُصلَح** | `initialised` ref + علم `cancelled` كانا يتعطّلان معاً تحت StrictMode؛ الحارس الصحيح هو `cancelled` وحده |
| X-5 | **مُصلَح** | 21 صفاً + 3 في الطابور عند كل إقلاع، idempotent |
| X-6 | **مُصلَح** | 1440 دقيقة افتراضياً، قابلة للضبط |
| X-7 | **مُصلَح** | الكاش يحجب استدعاء النموذج فقط؛ السجل والطابور وPrometheus تعمل على الإصابة تماماً كما على الإخفاق. اختباران يفشلان على الكود القديم |
| X-8 | **مُصلَح** | `LOG_LEVEL` افتراضه INFO، و`sqlalchemy.engine` مثبَّت على WARNING |
| C-1 | **خارج النطاق** | لم يُمسّ ولم يُفحص بطلب صريح |

---

## B. تفاصيل ما تغيّر

### 3. Clinical Notes Parser — نتائج الـ12 ملاحظة

مجموعة الملاحظات استُعيدت في `tests/fixtures/clinical_notes.json`. **الملف الأصلي `notes.json` كان في مجلد جلسة لم يعد موجوداً**، فكل ملاحظة تُعيد إنتاج الفخّ الموثّق في §3 من التقرير السابق لا الصياغة الأصلية حرفياً — والملف يقول ذلك.

| Note | اللغة | الحقول | النتيجة |
|---|---|---|---|
| H-EN-1 | EN | 8 | `ChestPainType=TA` — **كان ASY** (كل ذكر لألم الصدر كان يُسطَّح إلى ASY) |
| H-EN-2 | EN | 5 | `Cholesterol=230` لا 160 — **LDL لم يعد يُقرأ ككوليسترول كلي**؛ **لا MaxHR** من "Pulse 88 at rest"؛ لا `FastingBS` من `hypertension` |
| H-EN-3 | EN | 5 | `ChestPainType=ASY` (صحيح، ولسبب صحيح الآن)؛ **لا MaxHR** من "HR 72 at rest" |
| H-AR-1 / H-AR-3 | AR | 0 | **يُبلَّغ عنها كغير مدعومة** بدل "0 fields extracted" |
| H-AR-2 | AR+EN | 2 | قراءة جزئية، تُبلَّغ كذلك صراحةً |
| D-EN-1 | EN | 6 | `HighBP=1, Smoker=1, HeartDiseaseorAttack=1` — الإثبات ما زال يعمل |
| D-EN-2 | EN | 7 | **`Smoker=0, HighBP=0, Stroke=0, HeartDiseaseorAttack=0`** — كانت كلها 1 وتقلب نتيجة الفحص |
| D-EN-3 | EN | 5 | فئة عمر BRFSS صحيحة من العمر المذكور |
| D-AR-1 / D-AR-3 | AR | 0 | غير مدعومة صراحةً |
| D-AR-2 | AR+EN | 1 | قراءة جزئية |

فئات BRFSS من العمر المذكور: 52→7، 47→6، 42→5 — كلها صحيحة.

**خطوة Apply:** تحقّق بالنقر — الحقول تُعرض كقائمة تحقّق، ولا شيء يُطبَّق حتى يضغط الطبيب Apply، وبعدها يُعاد الفحص. (`fields are NOT applied before confirmation` ثم `Apply confirms and re-runs the screening`.)

### 6 + 14. ما الذي يغادر إلى DeepSeek الآن

**حُذف بالكامل** (كان يُرسَل في كل تقرير):

- القلب 11/11: Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope
- السكري 20/21: HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education (وIncome كان يسقط أصلاً بالحدّ)

**ما يزال يُرسَل:** اسم وحدة المرض، الاحتمال، عتبة القرار، التصنيف، النطاق، وأعلى 5 ميزات SHAP **بأسمائها وقيمها** + مسرد يشرح ما تقيسه كل ميزة. لا قيمة مطلوبة: SHAP يقول أصلاً أي العوامل دفعت التقدير وفي أي اتجاه.

**"LDL":** `Cholesterol` في بيانات UCI هو الكوليسترول **الكلي**، وكانت التقارير تصفه بـ"LDL burden" — جزء دهني مختلف لا تقيسه المنصة أصلاً. المسرد ينصّ على ذلك، والـ system prompt يمنع تسمية أي تحليل خارج المسرد، و`forbidden_content()` يرفض تقريراً يذكر LDL/HDL/triglycerides/HbA1c.

**Positive مع LOW:** D-004 عند 14.4% مقابل عتبة 10.8% ونقطة قطع moderate عند 17.2% كان يُوسم إيجابياً ويُقال له «rescreen in 12 months». `band_for_report()` يضع حدّاً أدنى: من هو عند العتبة أو فوقها لا يكون LOW أبداً.

**عيب اكتُشف أثناء التحقق الحيّ:** أحد مسارات الرجوع الثلاثة (مسار رفض الحارس، وهو الذي يعمل فعلياً) كان يُسقط `decision_threshold`، فيعيد MODERATE في البيانات و«low priority» في النص. الثلاثة تمرّره الآن، ولكلٍّ اختبار.

### 9. Batch

- عمودا «Prediction» و«Diagnosis» كانا الحقيقة نفسها مرتين → عمود واحد **«Screening result»** في الجدول وفي الـ CSV.
- عنوان الاحتمال في الـ CSV كان دائماً `risk_probability_corrected`، وهو ادّعاء كاذب للقلب. `router.get_disease_info()` يُصدِر الآن `prevalence_corrected` (حقل إضافي) والتصدير يتبعه.
- **سقف 20 صفاً للسكري**، يُفحص في المتصفح قبل الرفع: «has 25 data rows — the limit for diabetes is 20 (diabetes scoring takes about 0.1 s per patient). Split the file and run it in parts.»
- النتائج تُمسح عند تبديل المرض (كانت نتائج القلب تبقى تحت عنوان «Diabetes» وعتبة 10.8%).

### 8. PDF و Print

كان `PDFReport` لا يستقبل `bestAchievable` أصلاً ولا يحتوي قسم What-If، فمريض تقول له الشاشة «Referral is recommended» يُنتج تقريراً مطبوعاً بلا سيناريوهات ولا إحالة — **النسخة التي تغادر المبنى هي التي كانت تفقد التوصية**. `frontend/src/utils/whatIfSummary.js` صار يحمل الصياغة وتصنيف الحالة مرة واحدة، وكلٌّ من البطاقة والـ PDF يعرض منه، فلا يمكن أن يفترقا.

---

## C. الإجابات المطلوبة

### (i) ما الذي يُرسَل إلى DeepSeek؟

انظر §B أعلاه. الخلاصة: **لم تعد أي قيمة مريض مقيسة تغادر الخادم.** هذا ما كان يجعل التقرير يقول "62-year-old female" و"cholesterol 340"؛ لم يعد ممكناً. معامل `features` ما زال يُقبل ليبقى العملاء الحاليون يعملون، ولا يُمرَّر.

### (ii) هل تؤثّر ملاحظات الطبيب على التنبؤ أو التقرير أو إعادة التدريب؟

| السؤال | الجواب الآن |
|---|---|
| أين يُخزَّن النص؟ | **`predictions.notes`** عبر `POST /api/v4/patients/{id}/notes` — كان لا يُخزَّن في أي مكان |
| هل يظهر لاحقاً؟ | **نعم**: في History (مع شارة «Note» على البطاقة المطوية) وفي الـ PDF |
| هل يدخل التنبؤ؟ | **لا** — لا يُدمج في `input_features` أبداً؛ اختبار يثبت ذلك |
| هل يصل إلى DeepSeek؟ | **لا** — `ClinicalReportModal` يبني حمولته من الاحتمال والعتبة والتصنيف وSHAP، ولا وصول له لهذا العمود |
| هل يصل إلى إعادة التدريب؟ | **لا** — `retrain.get_annotated_samples` يقرأ `label` و`predictions.input_features` فقط |
| ملاحظة الوسم في الطابور | **تُخزَّن** في `review_queue.notes` (كانت تُقبل ثم تُرمى) وتظهر في قائمة الطابور |

---

## D. سكربت العرض اليدوي (20 دقيقة)

**قبل البدء — شغّل التسخين:**

```bash
backend/.venv/bin/python scripts/warmup_demo_cache.py
```

يسخّن `/predict` و`/explain` و`/counterfactuals` لكل المرضى التجريبيين السبعة. التشغيل الأول ~61 ث، وبعده كل نقرة فورية لمدة 12 ساعة. **أعد تشغيله بعد أي إعادة تشغيل للـ Space** (الكاش في الذاكرة).

| الوقت | الخطوة | المتوقع | الحالة |
|---|---|---|---|
| 0:00 | `GET /` | `{"status":"Healthy"}` | ✅ |
| 0:30 | افتح الواجهة، افتح قائمة المرض | اسمان كاملان مع وصف ونسخة — **لا مفاتيح خام** | ✅ Playwright |
| 1:30 | Engineering ← heart ← Randomize | قيم معقولة سريرياً (BP ~135، كوليسترول ~232) | ✅ 500 عينة |
| 2:00 | Run Inference | نتيجة + SHAP + What-If، 0 "Invalid input" | ✅ |
| 2:30 | Scales | «UCI Heart Disease … Age is in years» — **لا BRFSS** | ✅ |
| 3:30 | بدّل للسكري ← Scales | «CDC BRFSS 2015 … Age is a BRFSS 5-year band» | ✅ |
| 4:30 | Generate AI Report | `source: llm`، بلا أدوية/تشخيص/قيم خام | ✅ 3 مرضى حيّاً |
| 6:00 | Clinical EMR Mode | يعمل فوراً، بلا أخطاء | ✅ |
| 7:00 | الصق ملاحظة إنجليزية ← Extract | قائمة تحقّق، **لا شيء يُطبَّق** حتى Apply | ✅ |
| 8:00 | اضغط Apply | «applied — screening re-run» | ✅ |
| 9:00 | الصق ملاحظة عربية ← Extract | **«Arabic notes are not supported»** صراحةً | ✅ |
| 10:00 | اكتب ملاحظة في «Clinical Note» ← Save note | «Saved to this screening» | ✅ |
| 11:00 | History | خط زمني بـ3 نقاط + شارة «Note»؛ افتح البطاقة لترى النص | ✅ |
| 12:30 | Export PDF | يحتوي What-If **ورسالة الإحالة** والملاحظة | ✅ |
| 13:30 | Print (معاينة) | **اسم المريض ظاهر** | ✅ |
| 14:30 | Sign In (doctor) ← Batch ← heart ← 3 صفوف | «Screening result» بعمود واحد | ✅ |
| 15:30 | بدّل للسكري | **النتائج تُمسح**؛ الحدّ يقول 20 | ✅ |
| 16:00 | ارفع 25 صفاً سكري | رفض بذكر عدد الصفوف والسبب | ✅ |
| 17:00 | Patient Comparison | «Patient A / Patient B»، والفرق «vs Patient A» | ✅ |
| 18:00 | Admin ← Annotation Queue | 3 صفوف مملوءة، مع خانة ملاحظة لكل صفّ | ✅ |
| 18:30 | اكتب ملاحظة واضغط + Pos | الوسم **والملاحظة** يُحفظان | ✅ من القاعدة |
| 19:30 | Dark Mode ← iPhone 12 | **لا تمرير أفقي** | ✅ 390=390 |

---

## E. الأولويات المتبقية

### P0 قبل التحكيم

| # | البند | الحالة |
|---|---|---|
| 1 | **C-1** — كلمات المرور الافتراضية + المستودع العام | **لا يزال قائماً. خارج نطاق هذه الجولة بطلب صريح.** الأكثر حرجاً بلا منازع |
| 2 | النشر على LIVE | **لم يتمّ** — انظر §G |

### ما لا يُعرض حتى لو سُئلت

- **Evidently / MLflow** — لا تَعِد بهما. Prometheus فقط.
- **BioBERT** — غير مُستدعى في النشر إطلاقاً؛ قل «قواعد regex».
- **الملاحظات العربية** — غير مدعومة، والواجهة تقولها.
- **Compare** — «مقارنة مريضين»، ليست قبل/بعد.

---

## F. ما لم يُتحقَّق منه (UNVERIFIED) ولماذا

| البند | السبب |
|---|---|
| LIVE بالكامل | لم يُنشر في هذه الجولة — §G |
| الإقلاع البارد لـ HF Space | يتطلب إيقاف الـ Space |
| تثبيت PWA الفعلي | يتطلب `beforeinstallprompt` في متصفح حقيقي؛ الأيقونات نفسها مُتحقَّق منها في البناء |
| نافذة الطباعة الفعلية | headless؛ استُخدم `emulate_media('print')` وCSS مُتحقَّق منه |
| `/admin/retrain` | خارج الحدود (يستبدل ملف الإنتاج) |
| Batch سكري 500 صف | السقف صار 20 للسكري؛ القلب مُتجَّه ولم يُختبر عند 500 هذه الجولة |
| C-1 | خارج النطاق بطلب صريح |

## G. حالة النشر — **اقرأ هذا قبل العرض**

| المكوّن | الحالة | الدليل |
|---|---|---|
| **الخلفية (HF Space)** | ✅ **منشورة ومُتحقَّق منها** | snapshot `3610ba0` (من `d8ce1bf`). `verify_live.py`: 7 مرضى × 3 endpoints = **IDENTICAL** بسماحية 1e-6، بما فيها D-003 عند 63.74%. History تعمل لكل المرضى (3 نقاط لكلٍّ، مطابقة للمحلي). الطابور فيه 3 حالات. `expires_in = 86400 s` (24 ساعة). الكاش مُسخَّن 21/21 |
| **الواجهة (Vercel)** | ❌ **لم تُنشر** | الحزمة الحيّة ما زالت `index-Ca31dcIj.js` (22 سبتمبر) والعنوان ما زال «Multi-Disease Diagnostic Platform». `/favicon.ico` و`/apple-touch-icon.png` يعيدان `text/html` |

**الدفع إلى `origin/deploy/v2-platform` لم يُطلق بناءً على Vercel** خلال 15 دقيقة من المراقبة. لا يوجد Vercel CLI ولا رمز وصول على هذا الجهاز، فالنشر يحتاج تدخّلاً يدوياً.

**نتيجة Playwright على LIVE: 9 نجاح / 21 فشل.** كل فشل بند واجهة، وسببها جميعاً واحد: الحزمة القديمة. البنود التي نجحت هي المدفوعة بالخلفية. **هذه ليست 21 مشكلة، بل مشكلة واحدة.**

**ما يجب فعله قبل التحكيم:**

1. انشر الواجهة من `deploy/v2-platform` (لوحة Vercel → Redeploy، أو `vercel --prod` من `frontend/`).
2. تأكّد أن الحزمة تغيّرت: `curl -s https://omnidiag-delta.vercel.app/ | grep -o 'index-[^"]*\.js'` — يجب ألّا تكون `index-Ca31dcIj.js`.
3. أعد تشغيل `scratch/ui_verify.py https://omnidiag-delta.vercel.app https://yahyoha-omnidiag.hf.space` — المتوقع 57/57.
4. شغّل `scripts/warmup_demo_cache.py` بعد أي إعادة تشغيل للـ Space.

**الرجوع (rollback) للخلفية:**

```bash
git push hf 2074dec5e8f0475cb59f1e39915cd2a5c28f5d37:main --force
```
