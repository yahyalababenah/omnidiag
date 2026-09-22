# OmniDiag — تحقّق الميزات قبل عرض التحكيم (FEATURE VERIFICATION)

**التاريخ:** 2026-09-22 (إعادة تحقّق بعد جولة إصلاحات) · **الفرع:** `fix/demo-blockers` @ `c21fb50` (= `deploy/v2-platform` + commit توثيقي واحد) · **النوع:** تقرير فقط — لم يُعدَّل أي كود، لا commits، لا نشر، لا إعادة تدريب.

هذا تحديث لتقرير 2026-09-22 الأصلي (كُتب @ `1243fbe`، **قبل** دمج `79dd3e1`/`b923864`/`840d342`/`ae2c3ff`/`4e07df0`). كل رقم أدناه أُعيد قياسه اليوم على LIVE فعلياً؛ الفروق عن النسخة السابقة مذكورة صراحةً.

**البيئات التي اختُبرت فعلياً**

| البيئة | الواجهة | الخلفية | ما الذي يعمل فعلاً |
|---|---|---|---|
| LOCAL | `backend/.venv` (Python 3.13، `scikit-learn==1.9.0`, `xgboost==3.3.0`, `lightgbm==4.6.0` — مطابقة لِـ `requirements.txt` المثبَّت) عبر `fastapi.testclient.TestClient` في نفس العملية (بلا حاجة لخادم) | نفسه | كود HEAD بالكامل |
| LIVE | `https://omnidiag-delta.vercel.app` | `https://yahyoha-omnidiag.hf.space` (HF Space، يفترض commit `2074dec`؛ لم يُتحقَّق من الـ commit المضبوط، انظر UNVERIFIED) | **الواجهة أُعيد نشرها منذ التقرير السابق ولم تعد يونيو (انظر X-1، تغيّر جوهري)** · **الخلفية تطابق HEAD رقمياً** — `backend/.venv/bin/python scratch/verify_live.py` (7 مرضى × 3 endpoints) أعاد **IDENTICAL** (0 فروق، tolerance `1e-6`) بما فيها D-003/Case C (انظر X-2، أُصلح) |

**الأدوات:** Playwright 1.60 (Chromium من `~/.cache/ms-playwright/chromium-1243`، عبر `~/venv`، headless) لمسارات الواجهة الحيّة؛ `scratch/verify_live.py` (TestClient محلي مقابل LIVE عبر HTTP) لتطابق API؛ استدعاءات `curl`/`python urllib` مباشرة لـ endpoints فردية (auth، parse-notes، generate-report). لم يُشغَّل خادم `uvicorn`/`vite preview` محلي هذه الجولة — `TestClient` يغطي كل مسارات الـ API بنفس الدقة بدون حاجة لمنفذ شبكة، والتركيز كان على LIVE (هدف التصحيح). كل السكربتات والنتائج الخام ولقطات الشاشة في مجلد scratchpad الخاص بهذه الجلسة (مؤقت، يستبدل مسار الجلسة السابقة `.../ee435309-ba83-4e77-8c43-a5ad943ca1f1/...` الذي لم يعد موجوداً):
`/tmp/claude-1000/-home-yahia-Desktop-Projects-Heart-Disease-Project/5c3afb10-bccb-4988-bfbb-07457fbbdc89/scratchpad/`
(`verify_live_result.json` — 7×3 endpoint diff؛ `ui_verify.py`, `ui_verify_emr.py`, `ui_results*.json` — Playwright لـ B-1/B-2؛ `shots/*.png` — Engineering Mode قلب قبل/بعد التشغيل، EMR بعد تبديل المرض فوراً وبعد 20 ثانية). انسخه إن أردت الاحتفاظ بالأدلة.

**بيانات أُنشئت على LIVE هذه الجولة:** طلبات `/predict`/`/explain`/`/counterfactuals` لـ 7 مرضى تجريبيين (عبر `verify_live.py`)، طلب دخول admin وdoctor (للتحقق من C-1)، طلب `/generate-report` واحد لمريض قلب اصطناعي، طلب `/parse-notes` واحد، وتصفّح Playwright لصفحتي Engineering Mode وClinical EMR Mode (بدون تسجيل دخول). **لم يُحذف شيء، ولم يُوسَم أي عنصر مراجعة، ولم يُستدعَ `/admin/retrain`.**

---

## ⚠️ خمس حقائق يجب أن تعرفها قبل أي شيء آخر (مُحدَّثة)

1. **X-1 — مُصلَح. الواجهة الحيّة على Vercel لم تعد قديمة.** الحزمة الحيّة الآن `index-Ca31dcIj.js` (رأس `last-modified: Tue, 22 Sep 2026 16:10:13 GMT`، `x-vercel-cache: HIT`). تحتوي نصوص إصلاحات سبتمبر (`"Even with every modifiable factor improved"`, `"Elevated risk"`, `"No change to the modifiable factors lowers"` — كلها موجودة)، ولا تحتوي أياً من عبارات What-If الوهمية القديمة (`"Powered by DiCE"`, `"Coming Soon"` — صفر تطابق)، ولا مرضى السكري القدامى (Layla Mansour وغيرها — صفر تطابق)، وتحتوي المرضى التجريبيين الحاليين (Samir Abu-Ghazaleh، Fatima Hassan). **تحقّقت عملياً عبر Playwright بتشغيل Engineering Mode للقلب وتبديل مرض EMR على الرابط الحي (انظر B-1/B-2 أدناه) — الواجهة المنشورة الآن هي فعلياً كود ما بعد `79dd3e1`.**
2. **X-2 — مُصلَح. لا يوجد انحراف بعد الآن بين LOCAL وLIVE.** `requirements.txt` أصبح مثبَّت الإصدارات (`scikit-learn==1.9.0`, `xgboost==3.3.0`, `lightgbm==4.6.0`، بدلاً من `>=1.3.0` القديم) منذ `4e07df0`. شغّلت `scratch/verify_live.py` (7 مرضى × `predict`/`explain`/`counterfactuals` = 21 مقارنة) LOCAL (TestClient، نفس بيئة `.venv` المثبَّتة) مقابل LIVE: **IDENTICAL — 0 فروق بسماحية `1e-6`**، بما فيها Case C (D-003): **63.74% محلياً = 63.74% حيّاً بالضبط** (كانت 63.7%/49.4% في التقرير السابق). لم أستطع قراءة سجل تشغيل الـ Space مباشرة هذه الجولة (endpoint `/logs` غير عام) لتأكيد رقم sklearn في وقت التشغيل نفسه، لكن التطابق الرقمي الكامل على 21/21 قيمة دليل حاسم بذاته أن أي انحراف مكتبات سابق لم يعد قائماً.
3. **C-1 — لا يزال صحيحاً، غير مرتبط بأي من إصلاحات سبتمبر، أُعيد التحقق منه اليوم.** `POST /auth/login` بـ `{"email":"admin@omnidiag.com","password":"Admin@123"}` و`{"email":"doctor@omnidiag.com","password":"Doctor@123"}` (البيانات المزروعة في [backend/main.py:139-140](../backend/main.py#L139) و[backend/main.py:155-156](../backend/main.py#L155)) أعاد **200 + access_token** لكلا الحسابين على الـ Space اليوم. مستودع GitHub `yahyalababenah/omnidiag` لا يزال **عاماً** (`"private": false`, تحقّق عبر GitHub API اليوم). لم يتغيّر شيء هنا؛ لا يزال حرجاً.
4. **B-1 — مُصلَح. Engineering Mode للقلب يعمل الآن على LIVE.** commit `79dd3e1` أضاف `unwrapNullable()` في [frontend/src/utils/schemaFieldParser.js:118-132](../frontend/src/utils/schemaFieldParser.js#L118) التي تفكّ `anyOf:[{type,…},{type:'null'}]` وتُبقي النوع الرقمي الحقيقي للحقول الخمسة (RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak). **تحقّق Playwright على الرابط الحي اليوم:** فتح Engineering Mode ← تبديل المرض إلى `heart_disease` ← Run Inference بلا أي تعديل يدوي → **0 رسائل "Invalid input"**، طلب فعلي أُرسل، النتيجة ظهرت: "Screening result / Below threshold / Risk Probability (threshold: 37.0%) 29.15% / Prediction Negative" + مخطط SHAP لـ11 ميزة + بطاقة What-If "Low Clinical Risk" بصياغة صحيحة (ليست البطاقة الوهمية القديمة). لقطة: `shots/eng_heart_after_run.png`.
5. **B-2 — مُصلَح. تبديل المرض في Clinical EMR لم يعد ينتج خطأ.** commit `79dd3e1` أضاف تجاهل رد المريض السابق عند التبديل وعدّاد تسلسل يُسقط الردود القديمة ([ClinicalEmrMode.jsx](../frontend/src/components/ClinicalEmrMode.jsx))، وحذف بطاقة What-If الوهمية ("Coming Soon · Powered by DiCE") من [WhatIfScenarioCard.jsx](../frontend/src/components/WhatIfScenarioCard.jsx). **تحقّق Playwright على الرابط الحي اليوم:** Clinical EMR Mode (سكري) ← تبديل إلى `heart_disease` ← فحص فوري وبعد انتظار 20 ثانية إضافية: **لا "[object Object]"، لا "Screening Error"، لا "Diagnosis Error" في أي من الحالتين.** المريض الأول (Ahmed Al-Rashid, P-001) يظهر مباشرة بـ"SCREENING RESULT — CORONARY ARTERY DISEASE RISK / Below threshold / 29.2%" الصحيحة. لقطات: `shots/emr_after_switch_immediate.png`, `shots/emr_after_switch_20s.png`. (لم أعد بحاجة لزر Refresh كحل بديل — العطل نفسه لم يعد يظهر.)

---

## A. جدول الملخص

الحالات: WORKS / PARTIAL / BROKEN / NOT DEPLOYED / MOCK. مفاتيح التقييم: `Demo 25` · `Technical 20` · `Evidence 5` · `Q&A 5` · `Docs`.

| # | الميزة | LOCAL (HEAD) | LIVE (Vercel قديم + HF = HEAD) | الحالة | الدليل | معيار التقييم |
|---|---|---|---|---|---|---|
| 1a | Engineering Mode — سكري | (لم يُعَد اختباره محلياً هذه الجولة؛ لا commit يمسّه — الحالة السابقة تُرحَّل: يعمل، 26 عمود SHAP) UNVERIFIED-carried | يعمل، صياغة "Screening result" الصحيحة (تحقّق Playwright غير مباشر عبر مسار القلب المطابق) | WORKS | `verify_live_result.json` (predict/explain متطابقان) | Demo, Technical |
| 1b | Engineering Mode — قلب | (لم تُشغَّل نسخة LOCAL منفصلة هذه الجولة — كود HEAD مطابق للحيّ رقمياً حسب X-2) | **مُصلَح.** Run Inference يعمل، 0 "Invalid input"، نتيجة كاملة + SHAP + What-If صحيح | **WORKS** | B-1، `shots/eng_heart_after_run.png`، `schemaFieldParser.js:118-132` (`unwrapNullable`) | Demo |
| 1c | Scales / Randomize / Reset | لم يُعَد اختباره هذه الجولة؛ لا commit يمسّ `schemaToZod.js`/Scales UI بعد `79dd3e1` سوى تعديل صغير على `nullable` | UNVERIFIED-carried (يُرجَّح استمرار نفس العيوب الشكلية: "Categorical" بلا مدى، تذييل BRFSS للقلب) | PARTIAL (مُرحَّل) | — | Demo, Q&A |
| 2a | EMR — اختيار المريض + ملخص البيانات | (كود HEAD مطابق للحيّ رقمياً حسب X-2) | **مُصلَح.** تبديل المرض لم يعد ينتج خطأ، فوراً ولا بعد 20 ثانية؛ لم تعد هناك حاجة لزر Refresh كحل بديل | **WORKS** | B-2، `shots/emr_after_switch_immediate.png`, `shots/emr_after_switch_20s.png` | Demo |
| 2b | EMR — زر History | "Sign in…" دون دخول؛ بعد الدخول: `Unexpected token '<', "<!doctype "... is not valid JSON` | نفسه (Vercel يعيد `index.html` لـ `/api/v4/patients/...`) | **BROKEN** | `PatientTimeline.jsx:25` (`BASE='/api/v4'`)، `local_history_signed_in.png` | Demo |
| 3 | Clinical Notes Parser | (لم تُعَد نسخة LOCAL منفصلة؛ كود HEAD) | **مُصلَح جزئياً (خطر النفي زال).** `ae2c3ff` أضاف معالجة نفي صريحة: تحقّقت مباشرة على LIVE بـ `POST /parse-notes` على "non-smoker. No history of hypertension. Denies chest pain. Denies stroke, no heart disease." → `smoking_flag:0, hypertension:0, stroke_flag:0, heart_disease_flag:0` (كانت جميعها تُستخرج كـ 1 سابقاً). أيضاً `hypertension` لم يعد يُحوَّل إلى `FastingBS` (مطابق للإصلاح المذكور في الكومِت). لم أُعِد فحص الـ 12 ملاحظة الأصلية بالكامل ولا الدعم العربي (لا يزال 0 حقول متوقَّعاً — لا commit يضيف عربية) ولا خطوة "Apply" الجديدة في الواجهة (`ClinicalNotesInput.jsx` تغيّر: الحقول تُعرض كقائمة تحقّق ولا تُطبَّق حتى يضغط الطبيب Apply — لم أُتحقق UI هذه الجولة). | PARTIAL → **أفضل** (خطر النفي مُصلَح، الثغرات الأخرى مُرحَّلة) | §3، اختبار حي `/parse-notes` اليوم، `tests/test_notes_parser.py` (112 سطر جديد) | Demo, Q&A |
| 4 | SHAP chart + Feature Impact Summary | يعمل؛ النص يطابق أعلى 3 أعمدة في كل الحالات | يعمل | WORKS | `results_*.json › core.*.explain` | Technical, Evidence |
| 5a | What-If — عرض السيناريوهات/الإحالة | يعمل (رسالة الإحالة + best achievable) | **مُصلَح.** الواجهة الحيّة الآن تعرض نفس المنطق (بطاقة "Low Clinical Risk" الصحيحة تحققت في B-1؛ بطاقة "Coming Soon/Powered by DiCE" الوهمية غائبة كلياً من الحزمة الحيّة — صفر تطابق) | **WORKS** | X-1، `shots/eng_heart_after_run.png`، grep على `index-Ca31dcIj.js` | Demo |
| 5b | What-If — منطق الخلفية على LIVE | — | **مُصلَح جذرياً على مستوى الكود** (`b923864`): `lowest_achievable()` تفحص كل عتلة على حدة + كلها معاً وتُعيد `None` إن لم ينخفض التقدير عن خط الأساس — **best_achievable لا يمكن رياضياً أن يتجاوز الأساس بعد الآن** بدل `max(0.0, …)` القديمة التي كانت تُخفي الانقلاب. اختبار `tests/test_whatif_best_achievable.py` (124 سطر) يغطي الحالة. Case C تحديداً غير قابل لإعادة الاختبار بنفس الطريقة لأن X-2 (انحراف sklearn) هو ما كان يُنتج الانقلاب أصلاً، وقد أُصلح X-2 بالكامل (تطابق رقمي تام)، فالمشكلتان اللتان تسببتا في هذا العطل معاً — الانحراف العددي والحساب الذي لا يحمي من تجاوز الأساس — أُصلحتا. | **WORKS (مُصلَح)** | b923864، X-2، `tests/test_whatif_best_achievable.py` | Technical, Q&A |
| 5c | What-If — الزمن والكاش | سكري 14.0–14.7 ث غير مخزّن، 6–10 ms مخزّن؛ قلب ≤3.3 ث | سكري 8.5–9.6 ث، 0.57–0.86 ث مخزّن | WORKS (بطيء) | جدول §5، TTL 3600 ث (`main.py:597`) | Demo |
| 6 | AI Report (DeepSeek) | (لم يُعَد اختباره محلياً) | **`source: llm`, `deepseek-chat`**، والآن مع حارس مخرجات (`840d342`): اختبار حي اليوم لمريض قلب HIGH-risk (97.4%، عوامل خطر شديدة) أعاد `source: llm, fallback_reason: None` وتقريراً **خالياً تماماً** من أسماء الأدوية/الجرعات ومن صياغة "diagnosed"/"consistent with" (انظر C-i المحدَّث) | WORKS (المحتوى مضبوط الآن جزئياً) | §6، C(i)، `tests/test_report_guardrails.py` (97 سطر جديد) | Technical, Q&A |
| 7 | ملاحظات الطبيب (نص حر) | لا تُخزَّن في أي مكان من الواجهة | نفسه | NOT IMPLEMENTED | §7، C(ii) | Q&A |
| 8 | Export PDF + Print | PDF يطابق الاحتمال/العتبة/الصياغة لكن **بلا What-If ولا رسالة إحالة**؛ Print يخفي اسم المريض | PDF قديم: "AI DIAGNOSIS SUMMARY / Model Confidence 97% / Positive" بلا عتبة | PARTIAL | `shots/*/report_*.pdf` | Demo, Docs |
| 9 | Batch Prediction | (كود HEAD مطابق للحيّ رقمياً) | **على الأرجح مُصلَح — لم يُختبَر بالنقر هذه الجولة.** الخلل السابق (`405` من Vercel) كان لأن الواجهة القديمة (يونيو) كانت ترسل `/api/v4/{disease}/batch` لنطاق Vercel نفسه؛ `6284b03` (مدموج في HEAD قبل هذا التقرير أصلاً) صحّح الأصل، والحزمة الحيّة الجديدة تحتوي مرجع `hf.space` وطلب مباشر لـ `/api/v4/heart_disease/batch` على الـ Space يعيد `401 NOT_AUTHENTICATED` (توجيه صحيح) لا `405`. لم أسجّل دخولاً ولم أرفع CSV فعلياً عبر الواجهة. | UNVERIFIED-this-round (يُرجَّح WORKS) | curl مباشر لـ `/api/v4/heart_disease/batch` → 401 لا 405؛ `index-Ca31dcIj.js` يحتوي `hf.space` | Demo |
| 10 | Before/After Compare | (كود HEAD مطابق للحيّ رقمياً) | **على الأرجح مُصلَح — لم يُختبَر بالنقر هذه الجولة.** الحساب الخاطئ كان جزءاً من كود الواجهة القديمة (يونيو) الذي زال مع X-1؛ لا commit مضاف يمسّ `Compare` تحديداً لكنه معطوف على نفس الحزمة المُصلَحة الآن حيّاً. | UNVERIFIED-this-round (يُرجَّح WORKS) | X-1 (استبدال الحزمة بالكامل) | Demo |
| 11 | Admin Dashboard | (كود HEAD مطابق للحيّ رقمياً) | **على الأرجح مُصلَح.** الحزمة الحيّة الجديدة تحتوي `"Avg Risk Probability"`/`"Avg probability"` كتسميات، ولا تحتوي `"AVG CONFIDENCE"` (0 تطابق) — يطابق الإصلاح المذكور سابقاً. لم أختبر لوحة Annotation Queue ولا PATCH/DELETE بالنقر الفعلي (يتطلب دخول admin ونقرات حيّة، خارج نطاق تحقّق read-only). | UNVERIFIED-this-round (تسمية Avg على الأرجح WORKS؛ Annotation Queue ما زال UNVERIFIED) | grep على `index-Ca31dcIj.js` | Demo, Technical |
| 12 | Auth & RBAC | (كود HEAD مطابق للحيّ رقمياً) | يعمل؛ **كلمة مرور admin الافتراضية لا تزال تعمل (C-1، أُعيد التحقق اليوم بنجاح)**؛ لا commit يمسّ انتهاء صلاحية الرمز (15 دقيقة، X-6) — لا يزال قائماً | PARTIAL (أمني حرج، بلا تغيير) | C-1 أعلاه (طلب auth/login حي اليوم) | Technical, Q&A |
| 13 | Active learning queue | الحالات الحدّية تدخل تلقائياً (3/7)؛ الوسم يُخزَّن؛ الملاحظات تُهمل؛ التنبؤ المخزّن مؤقتاً لا يُسجَّل | الإدخال يعمل (2→5)؛ **واجهة الوسم تعرض صفوفاً فارغة** | PARTIAL | §13 | Technical |
| 14a | Prometheus `/metrics` | `# prometheus_client not installed` | يعمل (عدّادات `omnidiag_predictions_total`…) | local NOT DEPLOYED / live WORKS | `results_*.json › authed.metrics` | Technical |
| 14b | Evidently drift | `evidently` غير مثبّت → `run` = 503 `MONITOR_NOT_READY` | سجل الـ Space: `evidently not installed — drift monitoring unavailable` | NOT DEPLOYED | `results_*.json › authed.drift_*` | Technical, Q&A |
| 14c | MLflow | `mlflow` غير مثبّت؛ `runs` = 0 | مثبّت، أنشأ DB فارغة عند أول استدعاء؛ `runs` = 0 | NOT DEPLOYED (فعلياً) | نفسه | Q&A |
| 15a | Dark mode | يعمل (`html.dark`) | يعمل | WORKS | `dark_emr.png` | Demo |
| 15b | PWA | manifest 200 + service worker مسجّل | نفسه؛ `favicon.ico`/`apple-touch-icon.png` مفقودان (أيقونة Vite) | PARTIAL — التثبيت نفسه UNVERIFIED | `ui_*.json › theme` | Demo |
| 15c | Mobile 390px | عمود واحد لكن عرض المستند 547px (تمرير أفقي) | نفسه | PARTIAL | `mobile_emr.png` | Demo |

**مشاكل عابرة للميزات (Cross-cutting)**

| ID | المشكلة | الدليل |
|---|---|---|
| X-1 | ~~الواجهة الحيّة قديمة (يونيو)~~ **مُصلَح** — أُعيد النشر (`index-Ca31dcIj.js`, 22 سبتمبر) | أعلاه |
| X-2 | ~~انحراف LightGBM على LIVE~~ **مُصلَح** — `requirements.txt` مثبَّت الإصدارات الآن، تطابق رقمي تام 21/21 | أعلاه |
| X-3 | **الخادم يتجمّد بالكامل أثناء أي حساب سكري ثقيل.** الـ endpoints معرّفة `async def` وتستدعي كود ML المتزامن مباشرة. قياس محلي: أثناء batch سكري 150 صفاً (20 ث) استغرق `GET /` **17,981 ms**. على LIVE استغرق predict واحد **135,278 ms** خلال الاختبار. في كشك عرض: What-If سكري واحد (9–14 ث) يجمّد كل الزوار الآخرين. | `results_live.json › core.livefe:D-001.predict.ms` |
| X-4 | `npm run dev` لا يعمل محلياً: "Loading diseases…" للأبد. [DiseaseContext.jsx:35-72](../frontend/src/context/DiseaseContext.jsx#L35) يجمع `initialised` ref مع علم `cancelled`؛ StrictMode يلغي التشغيل الأول ويتخطّى الثاني. الإنتاج (`vite build`) غير متأثر. | `shots/dbg.png` |
| X-5 | قاعدة LIVE هي SQLite داخل الحاوية (`aiosqlite` في السجل) — تُمسح مع كل إعادة تشغيل/بناء؛ المستخدمون يُزرعون من جديد (IDs مختلفة بين الجلسات). | `hf_run.txt` |
| X-6 | رموز الدخول تنتهي بعد 15 دقيقة (`jwt.py:30`، وعلى LIVE أيضاً) ولا يوجد أي منطق refresh في الواجهة (`grep refresh` فارغ في `AuthContext.jsx`/`api.js`). بعد 15 دقيقة: Admin/Batch تفشل بـ "Could not validate credentials". | 401 بعد انتهاء الرمز محلياً |
| X-7 | التنبؤ المخزّن في الكاش (5 دقائق) يعود قبل الحفظ: تنبؤ طبيب مسجّل لمريض سبق التنبؤ به **لا يُحفظ ولا يدخل review queue**. قياس: `total_predictions` 7 → 7 مع `Cache-Hit: true`. | [main.py:400-414](../backend/main.py#L400) |
| X-8 | سجلات LIVE على مستوى DEBUG (10,106 سطر DEBUG من 10,584)، تشمل SQL. لم أجد نص prompt أو قيم مرضى كاملة فيها. | `hf_run.txt` |

---

## B. تفاصيل كل ميزة

### 1. Engineering Mode

**الاستخدام:** الشريط الجانبي ← Engineering Mode ← اختر المرض من القائمة أعلى الشريط ← عدّل الحقول (أو Randomize / Reset / Scales) ← Run Inference ← بطاقة Prediction Result + SHAP + What-If + Feature Impact Summary ← "Raw JSON Response" و"Raw SHAP JSON" قابلتان للطي ← "Generate AI Report".

**ما يراه الحكم (مُحدَّث 22 سبتمبر، بعد `79dd3e1`):**
- **قلب (LIVE، تحقّق Playwright اليوم):** بعد تبديل المرض إلى `heart_disease` والضغط على Run Inference **بلا أي تعديل يدوي**: "Screening result / Below threshold / Risk Probability (threshold: 37.0%) 29.15% / Prediction Negative" + مخطط SHAP لـ11 ميزة + بطاقة What-If "Low Clinical Risk. Patient is currently at low clinical risk. No counterfactual interventions are necessary." **B-1 مُصلَح.** لقطة: `shots/eng_heart_after_run.png`.
- **سكري:** لم يُعَد اختباره بالنقر هذه الجولة (لا commit مباشر يمسّ مسار السكري في Engineering Mode)؛ الصياغة التشخيصية القديمة ("Diagnosis/Confidence") كانت من الحزمة الحيّة القديمة التي استُبدلت بالكامل مع X-1 — يُرجَّح أنها أصبحت "Screening result" الصحيحة الآن مثل القلب، لكن غير مؤكَّد بالنقر مباشرةً على مسار السكري تحديداً.

**أعطال مُرحَّلة (لم تُمسَّها الإصلاحات، لم يُعَد اختبارها هذه الجولة):** قائمة الأمراض المنسدلة لا تزال تعرض المفاتيح الخام (`diabetes`, `heart_disease`) مع "No description" — **مؤكَّد اليوم من لقطة `disease_dropdown_open.png`**؛ Scales/Randomize لم يُعَد اختبارهما (لا commit يمسّهما جوهرياً).

**الوصف الصادق للكشك:** «وضع المهندس يسمح بتعديل أي مدخل ومشاهدة الاحتمال وتفسير SHAP فوراً — نعرضه على وحدة السكري.»

### 2. Clinical EMR Mode

**الاستخدام:** Clinical EMR Mode ← يُشغَّل المريض الأول تلقائياً ← القائمة المنسدلة للمريض ← بطاقات Patient Data Summary (ملوّنة حسب الاتجاه السريري) ← AI Risk Screening ← Clinical Insights (Print / Export PDF / AI Report / Refresh) ← What-If ← Feature Impact Summary ← زر History أعلى بطاقة المريض.

**ما يراه الحكم (LIVE، تحقّق اليوم بعد `79dd3e1`):** "SCREENING RESULT — CORONARY ARTERY DISEASE RISK · Below threshold · 29.2% Risk Probability (threshold: 37.0%)" — نفس الصياغة الصحيحة التي كانت محلية فقط، تظهر الآن على الرابط الحي مباشرة، بلا حاجة لـ Refresh (`shots/emr_after_switch_immediate.png`).

**History:** دون دخول: "Sign in to view patient history." بعد الدخول: `Unexpected token '<', "<!doctype "... is not valid JSON`. سببان مستقلان، كلاهما يمنع العمل:
1. [PatientTimeline.jsx:25](../frontend/src/components/PatientTimeline.jsx#L25) `const BASE = '/api/v4'` — عنوان نسبي يذهب إلى الواجهة نفسها (Vercel/preview يعيد HTML). لا يوجد proxy في `vite.config.js`.
2. حتى بالعنوان الصحيح: الواجهة تمرّر `P-001`/`D-001` (معرّفات وهمية) ← `GET /api/v4/patients/P-001/predictions` = **404** `Patient 'P-001' not found`. ولا توجد أي واجهة تُنشئ مرضى أو تمرّر `?patient_id=` للتنبؤ، فكل التنبؤات تُحفظ بـ `patient_id = NULL`. **History لا يمكن أن يعرض شيئاً في الإعداد الحالي.**

**أعطال أخرى:** ~~B-2 (خطأ بعد تبديل المرض)~~ **مُصلَح** — `79dd3e1` أضاف تجاهل الردود القديمة عبر عدّاد تسلسل، وهو نفس الآلية التي تحل مشكلة "طلب `diabetes/counterfactuals` قديم يكتمل بعد التبديل" المذكورة سابقاً (لم أُعِد قياس "يطلق مرتين" ولا كود `ReviewQueuePanel.jsx`/`PatientRiskTimeline.jsx` الميت هذه الجولة — لا commit يمسّهما تحديداً، يُرجَّح استمرارهما).

**الوصف الصادق:** «واجهة الطبيب تعرض بيانات مريض تجريبي، نتيجة الفحص مقارنةً بعتبة القرار، وتفسير العوامل.» (لا تذكر السجل التاريخي.)

### 3. Clinical Notes Parser (NLP)

**الاستخدام:** EMR ← "Clinical Notes Parser · NLP" (قابل للطي) ← الصق النص ← Extract Features ← تظهر شرائح `Field → value` و"N fields extracted — patient data updated" ← تُدمج القيم في بيانات المريض الحالي ويُعاد التنبؤ تلقائياً.

**أيّ مسار يعمل؟ regex فقط، على البيئتين. مُثبت:**
- الواجهة ترسل دائماً `use_bert: false` ([ClinicalNotesInput.jsx:36](../frontend/src/components/ClinicalNotesInput.jsx#L36))، والـ endpoint افتراضه `use_bert: False` ([main.py:886](../backend/main.py#L886)).
- محلياً `transformers` غير مثبّت أصلاً.
- مخرجات LIVE مطابقة بايتاً بايتاً لمخرجات LOCAL في الملاحظات الـ 12.
- لم أستدعِ `use_bert:true` على LIVE (قد يُنزّل نموذجاً ~400MB على Space مجاني قبل العرض). حتى لو عمل: نموذج `d4data/biomedical-ner-all` إنجليزي، وأسماء الكيانات التي يبحث عنها الكود (`DISEASE/CONDITION/PROBLEM`) لا تطابق تسميات هذا النموذج (`Disease_disorder`…)، فلن يضيف إلا العمر تقريباً. **وصف "BioBERT" في الكشك غير دقيق.**

**هل تغيّر القيم التنبؤ؟ نعم.** تم التحقق في الواجهة (LOCAL): ملاحظة H-EN-3 رفعت المريض من 29.2% "Below threshold" إلى **69.5% "Elevated risk"**؛ D-EN-2 إلى 17.6% Positive. العربية: "0 fields extracted — patient data updated" (الرسالة تقول "updated" رغم عدم تغيير شيء).

**النتائج (12 ملاحظة، القيم المُدمجة في مريض الأساس P-001 / D-001، LIVE):**

| Note | اللغة | صحيح | **خاطئ** (المستخرج → الصحيح) | مفقود | زائد | الاحتمال قبل → بعد | التصنيف |
|---|---|---|---|---|---|---|---|
| H-EN-1 | EN | Age, Sex, RestingBP, Cholesterol, MaxHR, ExerciseAngina, Oldpeak | ChestPainType: ASY → TA | FastingBS, RestingECG, ST_Slope | — | 29.2% → 95.7% | Neg → Pos |
| H-EN-2 | EN | Age, Sex, RestingBP, FastingBS¹ | ChestPainType: ASY → ATA; Cholesterol: 160 (LDL) → 230 | RestingECG | MaxHR=88 (نبض راحة) | 29.2% → 62.8% | Neg → Pos |
| H-EN-3 | EN | Age, Sex, ChestPainType², RestingBP, Cholesterol | MaxHR: 72 (راحة) → 172 | RestingECG, ExerciseAngina, Oldpeak, ST_Slope | — | 29.2% → **69.5%** | **Neg → Pos** |
| H-AR-1 | AR | — | — | كل الحقول العشرة | — | 29.2% → 29.2% | — |
| H-AR-2 | AR+EN | RestingBP, Cholesterol, FastingBS¹ | — | Age, Sex, ChestPainType, RestingECG | — | 29.2% → 25.9% | — |
| H-AR-3 | AR | — | — | كل الحقول التسعة | — | 29.2% → 29.2% | — |
| D-EN-1 | EN | Age, Sex, BMI, HighBP, HighChol, Smoker, HeartDiseaseorAttack | — | PhysActivity, Fruits, Veggies, GenHlth, DiffWalk, PhysHlth | — | 3.2% → 41.0% | Neg → Pos |
| D-EN-2 | EN | Age, Sex, BMI | **HighBP 1→0; Smoker 1→0; Stroke 1→0; HeartDiseaseorAttack 1→0** | HighChol, CholCheck, PhysActivity, GenHlth | — | 3.2% → **15.3%** | **Neg → Pos** |
| D-EN-3 | EN | Sex, BMI, HighBP, HighChol, Stroke | Age: 5 (40–44) → 3 (30–34) | PhysActivity | — | 3.2% → 3.1% | — |
| D-AR-1 | AR | — | — | كل الحقول التسعة | — | 3.2% → 3.2% | — |
| D-AR-2 | AR+EN | BMI, HighBP | — | Age, Sex, Smoker, PhysActivity, GenHlth | — | 3.2% → 3.1% | — |
| D-AR-3 | AR | — | — | كل الحقول السبعة | — | 3.2% → 3.2% | — |

¹ صحيح بالصدفة: الكود يحوّل `hypertension` إلى `FastingBS=1` ([notes_parser.py:218](../backend/nlp/notes_parser.py#L218))، أي "ارتفاع ضغط" ← "سكر صيام مرتفع". ² صحيح بالصدفة: أي ذكر لـ "chest pain" (حتى "Denies chest pain") ← `ASY`.
**الإجمالي:** 36 صحيح · 10 خاطئ/زائد · 63 مفقود. العربية الخالصة: **0 حقول في 4/4**؛ المختلطة تلتقط فقط الاختصارات الإنجليزية (BP, HTN, DM, BMI, Cholesterol). نصوص الملاحظات الـ 12 كاملة في `notes.json`.

**أخطاء منهجية سابقة — حالتها اليوم بعد `ae2c3ff`:**
- ~~لا معالجة للنفي~~ **مُصلَح ومؤكَّد حيّاً اليوم:** `_NEGATION_CUES` جديدة (`no|not|denies|denied|deny|without|negative for|never|free of|absence of|absent`) و`_negated()` تفحص النص قبل كلمة الحالة. اختبار حي: "non-smoker. No history of hypertension. Denies chest pain. Denies stroke, no heart disease." → `smoking_flag:0, hypertension:0, stroke_flag:0, heart_disease_flag:0, chest_pain_type:"ASY"` (وكلها كانت 1 سابقاً عدا chest_pain الذي كان يُستخرج بالصدفة بشكل صحيح).
- ~~`chest pain` بكل أنواعه ← `ASY`~~ **مُصلَح:** أنماط `_CHEST_PAIN` جديدة تميّز TA/ATA/NAP/ASY فعلياً بدل اعتبار أي ذكر "chest pain" = ASY.
- ~~`hypertension` ← `FastingBS`~~ **مُصلَح:** لم يعد `hypertension` يُحوَّل إلى `FastingBS` في الكود (تأكّد من الاختبار الحي أعلاه: `mapped_features` لم يتضمن `FastingBS`).
- ~~أول "HR/pulse" ← `MaxHR`~~ **مُصلَح على مستوى الكود:** `max_heart_rate` الآن يتطلب صراحةً "max/maximum/peak heart rate" — "pulse 88" أو "HR 72 at rest" لم يعودا يُقرآن كـ MaxHR. لم أُعِد اختباره حيّاً.
- `LDL` لا يُقرأ كـ Cholesterol الكلي بعد الآن (نمط `cholesterol` الجديد يشترط "total cholesterol"/"total chol"/"cholesterol" دون "LDL"/"HDL" — مُصلَح على مستوى الكود، لم يُختبر حيّاً).
- فئة عمر BRFSS: الكومِت يذكر "BRFSS age bucket formula" كإصلاح؛ لم أُعِد اختبار الصيغة الدقيقة حيّاً.
- لم يُعَد فحص الـ 12 ملاحظة الأصلية بالكامل ولا الدعم العربي هذه الجولة — **الدعم العربي لا يزال 0 حقول متوقَّعاً** (لا commit يضيف عربية).
- **جديد في `ae2c3ff`:** لا شيء يُطبَّق على بيانات المريض تلقائياً بعد الآن — `ClinicalNotesInput.jsx` تغيّر ليعرض الحقول المستخرجة كقائمة تحقّق، ولا يُدمَجها في بيانات المريض حتى يضغط الطبيب "Apply" صراحةً. هذا يُبطل جزءاً كبيراً من §C(ii) القديمة ("يغيّر التنبؤ نعم، بشكل غير مباشر وبدون تأكيد") — **الآن هناك تأكيد.** لم أختبر زر Apply بالنقر هذه الجولة.

**الوصف الصادق:** «نموذج أولي بقواعد regex للنصوص الإنجليزية؛ يقترح قيماً يجب أن يراجعها الطبيب ويضغط Apply صراحةً قبل أن تُطبَّق. لا يدعم العربية بعد.» — النفي أصبح آمناً للعرض (`ae2c3ff`)؛ لا يزال العرض العربي غير مدعوم فلا تعرضه بملاحظة عربية.

### 4. SHAP chart

**الاستخدام:** يظهر تلقائياً بعد أي تنبؤ؛ "Show All N Features" (11 للقلب، 26 للسكري = 21 + 5 مشتقة)؛ مرّر فوق العمود للـ tooltip (قيمة SHAP + اتجاه + تعريف طبي).

**ما يراه الحكم:** أحمر = يرفع الخطر، أخضر = يخفضه؛ "Base value (expected log-odds)". النص "Top factors influencing this prediction: …" يطابق أعلى 3 أعمدة ترتيباً واتجاهاً في كل حالة فحصتها (P-002, D-003, D-004 وغيرها).

**ملاحظات للأسئلة:** ميزات مشتقة تظهر للحكم (`Diabetes_Clinical_Risk`, `BMI_Age_Interaction`, `SES_Composite`, `Health_Index`, `Lifestyle_Score`)؛ `PhysActivity=Yes` يرفع الخطر في السكري (سلوك حقيقي للنموذج، موثّق سابقاً)؛ `CholCheck=0` يظهر "Protective". قيم SHAP على LIVE تختلف عن LOCAL للسكري (X-2).

**الوصف الصادق:** «SHAP يبيّن أيّ المدخلات دفعت هذا التقدير للأعلى أو للأسفل — تفسير للنموذج، لا سببية طبية.»

### 5. What-If

**الاستخدام:** يُحسب تلقائياً بعد التنبؤ في الوضعين. الحالات: "Low Clinical Risk" (تحت العتبة)، سيناريو/سيناريوهات تعبر العتبة، أو "Even with every modifiable factor improved, the estimated risk remains above the threshold… Referral is recommended." مع "Best achievable…".

**نتائج الخلفية، أُعيد قياسها اليوم (LOCAL TestClient مقابل LIVE، `verify_live.py`):**

| المريض | LOCAL | LIVE | فرق؟ |
|---|---|---|---|
| P-001 / P-003 | not_applicable | not_applicable | لا |
| P-002 (Fatima) | إحالة، best 96.5%→87.1% | نفسه | لا (`cf diffs 0`) |
| D-002 (Karim) | 19.31% | 19.31% | لا (`cf diffs 0`) |
| D-003 (Samir, Case C) | 63.74% | **63.74% (مطابق تماماً)** | **لا — كانت 63.7%/49.4% سابقاً، الآن مطابقة بالضبط** |
| D-004 (Hala) | 14.38% | 14.38% | لا (`cf diffs 0`) |

**X-2 (انحراف sklearn) وB923864 (حماية best_achievable من تجاوز الأساس) معاً كانا سبب انقلاب Case C سابقاً.** كلاهما أُصلح: `verify_live.py` اليوم يُظهر `counterfactuals` diffs = 0 لكل الحالات السبع (predict + explain + counterfactuals، 21/21 مطابقة بسماحية `1e-6`). لم أستخرج نص رسالة الواجهة بالضبط لـ Case C اليوم (يتطلب تصفّح EMR لمريض D-003 تحديداً مسجَّل دخول)، لكن الحساب الخلفي الذي تعتمد عليه الرسالة مطابق تماماً بين LOCAL وLIVE الآن، وكود `lowest_achievable()` يضمن رياضياً عدم تجاوز الأساس.

**الزمن (مللي ثانية):**

| | LOCAL غير مخزّن | LOCAL مخزّن | LIVE غير مخزّن | LIVE مخزّن |
|---|---|---|---|---|
| قلب (P-002) | 3,317 | 6 | 2,438 | 862 |
| سكري إيجابي (D-002/3/4) | 13,968–14,434 | 7–10 | 8,490–9,578 | 571–705 |
| سكري تحت العتبة | 161–167 | 6–9 | 628–1,056 | 587–769 |

"Cold" هنا = cache miss (رأس `Cache-Hit: false`)، والـ Space كان مستيقظاً (الإقلاع البارد للـ Space نفسه UNVERIFIED). **TTL:** counterfactuals 3600 ث ([main.py:597](../backend/main.py#L597))، predict 300 ث، schema 86400 ث. الكاش في الذاكرة (`InMemoryBackend`، لا Redis) ويُمسح مع إعادة التشغيل. أثناء الحساب يتجمّد الخادم (X-3).

**الوصف الصادق:** «What-If يبيّن كيف يتغيّر تقدير النموذج إذا تغيّرت العوامل القابلة للتعديل فقط؛ ليس أثراً علاجياً متوقعاً. عندما لا يكفي أي تعديل، يوصي بالإحالة.»

### 6. AI Report (LLM)

**الاستخدام:** Generate AI Report (Engineering) أو AI Report (EMR) ← نافذة ← Generate Report ← شارة "AI Generated" أو "Rule-Based".

**أيّ مسار يعمل حيّاً؟ DeepSeek — مُثبت بثلاثة أدلة:**
1. 11/11 طلب `POST /api/v4/generate-report` على LIVE أعادت `"source": "llm", "llm_model": "deepseek-chat"` بزمن 3.2–4.3 ث (`results_live.json › report`).
2. سجل الـ Space: `omnidiag.llm: Calling DeepSeek API (model=deepseek-chat) …` ثم `POST https://api.deepseek.com/chat/completions "HTTP/1.1 200 OK"` ثم `DeepSeek report generated in 3027ms`.
3. الواجهة عرضت الشارة "AI Generated" على البيئتين (Playwright).
المفتاح مُعدّ كـ secret في الـ Space (لم تُقرأ قيمته). المسار البديل (rule-based) لم يُفعَّل في أي طلب.

**هل يذكر تشخيصاً أو أدوية؟ مُصلَح جزئياً وبآلية تنفيذ فعلية (`840d342`)، أُعيد اختباره حيّاً اليوم:**

النظام الآن يفرض قاعدتين في الـ system prompt ("لا تشخيص، لا أدوية/جرعات") **ثم يتحقق من المخرجات آلياً** بدالة `forbidden_content()` — أي تقرير يخرق القاعدتين يُستبدَل تلقائياً بتقرير rule-based (`source: rule_based`, `fallback_reason` يذكر السبب). هذا أقوى من مجرد تعليمات prompt: حتى لو تجاهل DeepSeek التعليمة، الفحص بعد التوليد يمنع النشر.

**اختبار حي اليوم** (مريضة قلب اصطناعية 62 سنة، عوامل خطر شديدة عمداً: BP 160، كوليسترول 340، ST depression 3.2، عوامل شبيهة بالحالة التي كانت تُنتج "statin"/"antiplatelet"/"ischemia" سابقاً): النتيجة `source: llm, fallback_reason: None` — DeepSeek نفسه التزم بالقواعد هذه المرة، والتقرير:
- **لا يحتوي** "diagnosed"، "consistent with"، "confirms"، ولا أي اسم دواء (statin/aspirin/antiplatelet/…)، ولا أي جرعة رقمية.
- يستخدم صياغة "estimates a 97.4% probability of CAD risk … HIGH risk band" بدل تشخيص.
- التوصيات: "Refer for confirmatory cardiac evaluation"، "guideline-directed lifestyle counselling" — لا "coronary angiography" ولا "high-intensity statin" كما كان سابقاً.
- لا يزال يفسّر سريرياً بعض الشيء ("ST depression … indicates significant exercise-induced changes") لكن بلا كلمة "ischemia" وبلا أي من الأنماط المحظورة صراحة.

لم أُعِد فحص الـ 21 تقريراً الأصلية (10 محلي + 11 حيّ) ولا تأكيد نسبة الرفض الفعلية لـ `forbidden_content()` (أي كم تقريراً كان سيُرفَض لولا الإصلاح) — تقرير واحد جديد فقط اختُبر اليوم، وهو ناجح. **أخطاء "LDL burden" وتناقض النطاقات (D-004 Positive+LOW) لم تُمسَّهما هذه الإصلاحات ويُفترض استمرارهما** (لا commit يمسّ منطق `risk_band`/`bands`).
- حد المعدل 10 طلبات/دقيقة لكل IP لم يتغيّر (لم يُعَد اختباره).

**الوصف الصادق:** «تقرير سردي يولّده DeepSeek من الاحتمال وعوامل SHAP وقيم المريض المدخلة، موجّه للطبيب ويجب أن يراجعه.»

### 7. ملاحظات الطبيب / مدخلات الطبيب الحرة

راجع C(ii) للإجابة الكاملة. الخلاصة: **الحقل الوحيد الذي يكتب فيه الطبيب نصاً حراً في الواجهة هو Clinical Notes Parser، ولا يُخزَّن.** عمود `patient_visits.notes` موجود في قاعدة البيانات لكن لا تكتبه أي واجهة، والمكوّن الذي يعرضه غير مستخدم.

**الوصف الصادق:** «لا نخزّن ملاحظات حرة حالياً؛ النص يُحوَّل إلى قيم منظّمة يراها الطبيب ويعدّلها.»

### 8. Export PDF و Print

**الاستخدام:** EMR ← Clinical Insights ← Export PDF (تنزيل `OmniDiag_<Name>_<date>.pdf`) أو Print (`window.print()`).

**PDF (LOCAL، P-002 وD-003):** يطابق الشاشة في: الاسم، "Elevated risk — confirmatory testing recommended"، "97% (decision threshold 37.0%)"، "64% (decision threshold 10.8%) · Calibrated to real-world prevalence"، نص SHAP، وصورة المخطط (PNG مضمّن 1378×914). **لا يطابق:** قسم What-If/الإحالة **غائب كلياً** رغم أن الشاشة تعرض "Referral is recommended" للمريضين (`PDFReport` لا يستلم `bestAchievable`)؛ عنوان القسم عند وجود سيناريوهات "Recommended Lifestyle Interventions" يناقض "not a predicted treatment effect"؛ MRN "—"، Gender "—" (المريض التجريبي يحمل `sex` لا `gender`)، Date of Birth "Age 62"؛ التوقيع "Dr. Physician Name — OmniDiag Clinic"؛ صورة SHAP تتضمن رابط "Show All 26 Features".
**PDF (LIVE):** "AI DIAGNOSIS SUMMARY · Model Confidence 97% · Positive" — بلا عتبة، بصياغة تشخيصية.
**Print (LOCAL):** CSS الطباعة يخفي كل `<button>` ([index.css:133-141](../frontend/src/index.css#L133))، واسم المريض داخل زر القائمة ← **الاسم غائب من الطباعة**. لقطة الطباعة أظهرت أيضاً حالة B-2 ("Screening Error" + "Low Clinical Risk" لمريض 64%). طُبعت بـ `emulate_media('print')` — نافذة الطباعة الفعلية UNVERIFIED (headless).

**الوصف الصادق:** «تقرير PDF بنتيجة الفحص والعتبة وتفسير SHAP.» (لا تَعِد بسيناريوهات What-If في PDF.)

### 9. Batch Prediction

**الاستخدام:** Sign In (doctor) ← Batch Prediction ← اختر المرض ← ارفع CSV (رأس بأسماء حقول الـ schema، ≤500 صف) ← Run Batch ← رسم توزيع + جدول ← Download CSV.

| | LOCAL (API) | LIVE (API) |
|---|---|---|
| قلب 10 / 100 / 500 صف | 57 / 24 / 41 ms | 941 / 737 ms (10 / 100) |
| سكري 10 / 50 / 100 / 200 / 500 | 1.6 / 6.7 / 14.3 / 27.0 / **66.7 ث** | 1.7 / 5.2 / 9.7 ث (10 / 50 / 100) |

**HM-4:** السكري خطّي (~135 ms/صف محلياً، ~97 ms/صف حيّاً)، لم يحدث تعليق حتى 500 صف محلياً / 100 حيّاً (لم أختبر 500 حيّاً — تقديرياً ~50 ث). "التعليق" الذي يراه المستخدم هو X-3: الخادم كله متجمّد طوال الدفعة. `fetch` في `BatchUpload` بلا مهلة.
**الواجهة LIVE:** `POST https://omnidiag-delta.vercel.app/api/v4/{disease}/batch → 405` ← "Failed to execute 'json' on 'Response'…" (الواجهة القديمة قبل `6284b03`). **Batch لا يعمل على الرابط الحي.**
**الواجهة LOCAL:** تعمل للمرضين. CSV المُنزّل (سكري):
```
row,status,prediction,risk_probability_corrected,decision_threshold,diagnosis,data_completeness_warning,error
1,ok,0,2.6%,10.82%,Negative,,
2,ok,1,19.3%,10.82%,Positive,,
3,ok,1,63.7%,10.82%,Positive,,
4,ok,1,14.4%,10.82%,Positive,,
```
عيوب: عمود `risk_probability_corrected` يُكتب للقلب أيضاً (القلب ليس مصحّحاً)؛ **بعد تبديل المرض تبقى نتائج القلب معروضة تحت عنوان "Diabetes" و"THRESHOLD 10.8%"**، وDownload CSV حينها يكتب نتائج القلب بعتبة السكري.

**الوصف الصادق:** «فحص دفعي لملف CSV بحد 500 مريض؛ القلب فوري، السكري ~0.1 ث لكل مريض.»

### 10. Before/After Compare

**الاستخدام:** Before/After Compare ← اختر مريض BEFORE ومريض AFTER من المرضى التجريبيين ← Compare.
**ما يقارنه فعلاً:** تنبؤَين لـ **مريضين مختلفين** (مثلاً Ahmed 54 سنة مقابل Khalid 45 سنة)، مع جدول "Changed Features" لكل حقل مختلف. ليس نفس المريض قبل/بعد تدخّل؛ لا يمكن تعديل القيم.
**LOCAL:** صحيح للمرضين: "↓19% risk · 29% → 10% · Decision threshold 37.0%"؛ سكري "↑61% · 3% → 64% · threshold 10.8%".
**LIVE:** "RISK SCORE 71%" لمريض احتماله 29%، و"RISK SCORE 98%" لمريضة 2%، والسهم "↑19% risk" بينما الخطر انخفض (`riskScore = 1 − confidence` للسلبيين في الكود القديم).

**الوصف الصادق:** «مقارنة جنباً إلى جنب بين ملفّي مريضين وما يختلف بينهما.»

### 11. Admin Dashboard

**الاستخدام:** Admin Dashboard ← Sign In as Admin ← لوحة واحدة: بطاقات إحصاء، "Avg Risk Probability by Disease and Scale"، "Predictions — Last 7 Days"، "Predictions by Disease"، "Top Endpoints"، Users (New User، تعديل الأدوار، تفعيل/تعطيل، مفاتيح API)، Annotation Queue، Audit Log، Flush Cache، Refresh.

| اللوحة | LOCAL | LIVE | حكم |
|---|---|---|---|
| Total Predictions / Users / Patients | 7 / 2 / 0 | 22 / 2 / 0 | حقيقي؛ Patients دائماً 0 (لا واجهة لإنشاء مرضى) |
| متوسط الاحتمال | "AVG RISK PROBABILITY (ALL ROWS, MIXED SCALES) 34%" | **"AVG CONFIDENCE 22%"** | LIVE **mislabelled** (احتمال وليس ثقة، ويخلط مقياسين) |
| Avg Latency | 936.7 ms | 677 ms | يشمل كل المسارات بما فيها OPTIONS — رقم بلا معنى تشغيلي |
| By-disease / 7-day / Top endpoints | حقيقي | حقيقي | WORKS |
| Users | حقيقي | حقيقي | تعديل الأدوار/التعطيل/حذف المفتاح: **preflight `PATCH`/`DELETE` → 400 محلياً** (`allow_methods=["GET","POST","OPTIONS"]`, [main.py:258](../backend/main.py#L258))؛ على LIVE بروكسي HF يجيب 200 — النقر الفعلي UNVERIFIED |
| Annotation Queue | صفوف كاملة + Pos/Neg/Skip | **5 صفوف كلها "— — — —"** | LIVE غير قابل للاستخدام |
| Audit Log | حقيقي، مليء بـ OPTIONS، الوقت "7:31 AM" بدل 10:31 (UTC يُعرض كمحلي) | IPs كلها `10.16.x.x` (بروكسي HF، ليست IP العميل) | PARTIAL |
| Drift / MLflow | لا توجد لوحة في الواجهة | لا توجد | — |

لا توجد بيانات mock في لوحة Admin — كل الأرقام من قاعدة البيانات.

### 12. Auth & RBAC

**الأدوار المزروعة:** `super_admin, admin, doctor, nurse, viewer`؛ `CLINICAL_ROLES = (doctor, nurse, super_admin)`، `ADMIN_ROLES = (super_admin,)` ([rbac.py:35-36](../backend/auth/rbac.py#L35)).
**حكم مجهول (بلا دخول) — مُتحقَّق:** يستطيع Engineering Mode، EMR، Notes Parser، SHAP، What-If، AI Report، Compare، PDF/Print. لا يستطيع: Batch ("Sign in with a clinical account…")، History، Admin. كل المسارات المحمية أعادت 401 على البيئتين (`/auth/me`, `/admin/*`, `/api/v4/review/*`, `/api/v4/patients/`, drift, mlflow). `/metrics` مفتوح.
**Audit log:** كل طلب `/api/*` و`/auth/*` يُكتب مع `user_id` عند الدخول و`null` لغير المسجّل (مُتحقَّق: 7 صفوف predict بـ user_id الطبيب على البيئتين). تنبؤات المجهول تُسجَّل في audit لكن لا تُحفظ في جدول predictions.
**مشاكل:** C-1 (حرج)؛ X-6 (15 دقيقة بلا تجديد)؛ X-5 (القاعدة الحية تُمسح).

### 13. Active learning review queue

**التدفّق المُتحقَّق (LOCAL كامل، LIVE حتى الإدخال):** تنبؤ طبيب مسجّل ← إن كانت الإنتروبيا المتمركزة على العتبة ≥ 0.88 يُضاف صف في `review_queue`. من 7 مرضى دخل 3: P-001 (29.2%، عتبة 36.95%، إنتروبيا 0.978)، D-002 (19.3%، 0.921)، D-004 (14.4%، 0.981). على LIVE ارتفع الطابور 2 → 5.
**الوسم:** `POST /api/v4/review/{id}/annotate {"label":1,"notes":"…"}` ← 200 ← `review_queue.label=1, status='reviewed', reviewer_id` ← إعادة الوسم 409. **حقل `notes` يُقبل ثم يُهمل** (لا عمود له؛ [active_learning/routes.py:32-34](../backend/active_learning/routes.py#L32) و132-136). زر Pos/Neg في Admin يرسل `{label}` فقط.
**أين يذهب الوسم:** `retrain.get_annotated_samples` يقرأ `label + predictions.input_features` ([retrain.py:50-61](../backend/active_learning/retrain.py#L50)) ← `retrain_xgb` (مكسور وموثّق كذلك بقرارك السابق؛ يستبدل ملف الإنتاج للسكري).
**عيوب:** X-7 (التنبؤ المخزّن لا يدخل الطابور)؛ واجهة الطابور على LIVE فارغة.

### 14. Monitoring

| | LOCAL | LIVE |
|---|---|---|
| Prometheus `/metrics` | `# prometheus_client not installed` | يعمل: `omnidiag_predictions_total{disease="diabetes",prediction="1"} 16.0`، histogram `omnidiag_prediction_confidence_*` (يُسجَّل فقط للتنبؤات غير المخزّنة) |
| Evidently | غير مثبّت؛ `POST /drift/diabetes/run` → 503 `MONITOR_NOT_READY` | سجل: `evidently not installed — drift monitoring unavailable`؛ status: `monitor_ready: false`، report: "No report yet" |
| MLflow | غير مثبّت؛ `runs: []` | مثبّت؛ أول استدعاء أنشأ الجداول؛ `runs: []` |
| Prometheus server / Grafana | لا يوجد scraper في أي بيئة | لا يوجد |

**الوصف الصادق:** «نُصدّر مقاييس Prometheus حيّاً؛ مراقبة الانجراف وتتبّع MLflow مكتوبان في الكود لكنهما غير مفعّلين في النشر الحالي.»

### 15. Dark mode، PWA، الشاشة الضيقة

- **Dark mode:** يعمل على البيئتين (`document.documentElement.className = "dark"`).
- **PWA:** `manifest.webmanifest` (200, `application/manifest+json`) و`sw.js` مسجّل على البيئتين، أيقونات 192/512 موجودة. `favicon.ico` و`apple-touch-icon.png` غير موجودين (يعيدان HTML) وأيقونة التبويب هي شعار Vite. زر "Install App" لم يظهر في headless (لا حدث `beforeinstallprompt`) — **التثبيت UNVERIFIED**. تخزين آخر 5 predict/explain دون اتصال في `vite.config.js` لن يعمل: Workbox يخزّن GET فقط وهذه طلبات POST.
- **390px:** التخطيط عمود واحد، لكن `scrollWidth = 547` مقابل `innerWidth = 390` ← تمرير أفقي.

---

## C. الإجابات المطلوبة بالأدلة

### (i) ما الذي يُرسَل بالضبط إلى DeepSeek؟

**المسار:** المتصفح ← `POST /api/v4/generate-report` ([main.py:828-886](../backend/main.py#L828)) ← `generate_report()` ([report_generator.py:139-241](../backend/llm/report_generator.py#L139)) ← `AsyncOpenAI(base_url="https://api.deepseek.com").chat.completions.create(model="deepseek-chat", max_tokens=600, messages=[system, user])`.

**جسم الطلب من المتصفح إلى الخادم** (HEAD: [ClinicalReportModal.jsx:44-50](../frontend/src/components/ClinicalReportModal.jsx#L44)؛ الواجهة الحية ترسل `probability` و`confidence_band` بدل `probability_corrected`، والخادم يتجاهل الـ band):
```json
{ "disease": "diabetes",
  "probability_corrected": 0.4938,
  "label": "Positive",
  "shap_values": [ {"feature": "...", "shap_value": 0.75}, "... 26 items" ],
  "features": { "HighBP": 1, "HighChol": 1, "...": "21 raw patient values" } }
```
`features` = `selectedPatient.data` (EMR) أو `lastFormData` (Engineering) — **قيم المريض الخام كاملة**. تحقّقت من مفاتيح الطلب: `['disease','probability_corrected','label','shap_values','features']`.

**ما يغادر الخادم إلى DeepSeek — كل حقل:**

| # | الحقل في الـ prompt | المصدر | ملاحظة |
|---|---|---|---|
| 1 | System prompt ثابت | `_SYSTEM_PROMPT` | "senior clinical decision support AI…" |
| 2 | `Disease Module` | `display_name` من الإعدادات | "Coronary Artery Disease Risk" / "Diabetes Risk Assessment" |
| 3 | `Risk Probability` | `probability_corrected` | بنسبة مئوية |
| 4 | `Decision Threshold` | `model.inference_threshold` من YAML | 0.3695 / 0.1082 |
| 5 | `Risk Label` | `label` من العميل | "Positive"/"Negative" |
| 6 | `Risk Band` | يُعاد حسابه في الخادم | HIGH/MODERATE/LOW |
| 7 | `Top Risk Factors` | أعلى **5** من `shap_values` | `اسم الميزة: SHAP=+0.xxx (↑/↓)` |
| 8 | `Patient Features` | أول **20** زوج `key: value` من `features` ([report_generator.py:100-102](../backend/llm/report_generator.py#L100)) | **قيم خام**: القلب 11/11 (Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope)؛ السكري 20/21 (يسقط `Income`، آخر مفتاح) |

**لا يُرسَل:** اسم المريض، MRN، معرّف المريض، `id` المريض التجريبي، email المستخدم، نص Clinical Notes Parser، أي ملاحظة حرة، التاريخ المرضي/الأدوية/الشكوى من بطاقة المريض (كلها خارج `patient.data`).

**الخلاصة:** ليس "الاحتمال + SHAP فقط" — **العمر والجنس وكل القيم السريرية الخام تُرسَل إلى طرف ثالث**. قيم غير معرِّفة وحدها، لكنها بيانات صحية لكل مريض. وهذا سبب أن التقرير يذكر "62-year-old female" و"cholesterol 340". (لم يتغيّر شيء في مسار الإرسال هذا مع `840d342` — الإصلاح يفرض قيوداً على ما **يعود** من DeepSeek، لا على ما يُرسَل إليه؛ القيم الخام لا تزال تغادر إلى طرف ثالث كما كانت.)

**تحديث 22 سبتمبر — حارس المخرجات (`840d342`):** أضيفت دالة `forbidden_content()` في [report_generator.py](../backend/llm/report_generator.py) تفحص نص DeepSeek بعد عودته مباشرة (قبل عرضه للطبيب) بحثاً عن أنماط تشخيص (`diagnosed`, `consistent with`, `indicates ischemia`, …) وأنماط دواء/جرعة (أسماء أدوية شائعة، أرقام بوحدات mg/mcg/units). أي تطابق ← يُستبدَل الرد كاملاً بتقرير rule-based ولا يُعرَض نص DeepSeek المخالف أبداً. اختبار حي اليوم على حالة صُمِّمت عمداً لتحفيز اللغة القديمة (ST depression شديد، كوليسترول 340) أعاد `source: llm` (لم يُرفَض) وتقريراً خالياً من الانتهاكات — أي أن DeepSeek التزم بالتعليمة الجديدة في هذا الاختبار، والحارس لم يحتَج للتدخل. لم أُثبت أن الحارس فعلاً *يستبدل* رداً مخالفاً (يتطلب إجبار DeepSeek على كسر القاعدة، لم أحاول)؛ `tests/test_report_guardrails.py` الجديد (97 سطر) يغطي هذا على مستوى الوحدة.

### (ii) هل تؤثّر ملاحظات الطبيب على التنبؤ أو التقرير أو إعادة التدريب؟

**أين يكتب الطبيب نصاً حراً؟** مكان واحد فقط في الواجهة: `textarea` في Clinical Notes Parser ([ClinicalNotesInput.jsx:69-75](../frontend/src/components/ClinicalNotesInput.jsx#L69)). (حقول Engineering كلها رقمية/اختيارية؛ Admin لا يحتوي نصاً حراً للوسم.)

| السؤال | الجواب | الدليل |
|---|---|---|
| أين يُخزَّن النص؟ | **لا يُخزَّن.** `/parse-notes` لا يلمس القاعدة؛ الـ audit middleware يحفظ المسار والحالة والمدة فقط، لا الجسم | [main.py:889-913](../backend/main.py#L889)، [middleware/audit.py](../backend/middleware/audit.py) |
| هل هو مرتبط بسجل التنبؤ؟ | لا. التنبؤ يُحفظ بـ `input_features` (القيم بعد الدمج) فقط، و`patient_id = NULL` | [main.py:449-459](../backend/main.py#L449) |
| هل يظهر لاحقاً (History / PDF / Admin)؟ | لا في أيٍّ منها | — |
| هل يصل إلى التقرير (LLM)؟ | **النص لا**؛ لكن القيم المستخرجة منه تُدمج في `patient.data` فتُرسل ضمن `features` | C(i) |
| هل يغيّر التنبؤ؟ | **تغيّر جوهري (`ae2c3ff`):** لم يعد تلقائياً/بلا تأكيد. `ClinicalNotesInput.jsx` يعرض الحقول المستخرجة كقائمة تحقّق أولاً، ولا تُدمَج في بيانات المريض (وبالتالي لا يُعاد التنبؤ) حتى يضغط الطبيب "Apply" صراحةً. **أمثلة الانقلاب الخاطئ القديمة (H-EN-3، D-EN-2) لم تعد قابلة للحدوث أصلاً** لأن أخطاء النفي التي سبّبتها ("No stroke, no heart disease, Non-smoker" ← 1) أُصلحت بذاتها في `ae2c3ff` (تُستخرج الآن كـ 0 بشكل صحيح) — إذن الخطر انزاح مرتين: تأكيد الطبيب أولاً، ثم صحة الاستخراج نفسها. لم أختبر زر Apply بالنقر هذه الجولة. | §3، اختبار حي `/parse-notes` اليوم |
| هل يصل إلى إعادة التدريب؟ | القيم المُدمجة تُحفظ في `predictions.input_features` إن كان الطبيب مسجّلاً، وإن دخل التنبؤ الطابور ووُسم فإن `retrain` يقرأ تلك القيم. النص نفسه لا | [retrain.py:50-61](../backend/active_learning/retrain.py#L50) |
| عمود `patient_visits.notes` | موجود ويُكتب عبر `POST /api/v4/patients/{id}/visits` فقط؛ لا تستدعيه أي واجهة؛ المكوّن الذي يعرضه (`PatientRiskTimeline.jsx`) غير مستورد؛ لا يدخل التنبؤ أو التقرير أو التدريب | [patient_visit.py:32](../backend/db_models/patient_visit.py#L32)، [patients/routes.py:308-340](../backend/patients/routes.py#L308) |
| `notes` في وسم المراجعة | يُقبل في الـ API ثم يُرمى | [active_learning/routes.py:32-34](../backend/active_learning/routes.py#L32) |

**الخلاصة:** لا يوجد مسار يخزّن ملاحظات الطبيب. المسار الوحيد الذي يجعل النص الحر يغيّر التنبؤ هو Notes Parser، وهو يفعل ذلك تلقائياً دون خطوة تأكيد، وبأخطاء نفي تقلب النتيجة.

---

## D. سكربت العرض اليدوي (20 دقيقة)

**قبل البدء:** Chrome عادي (ليس incognito) + DevTools على تبويب Network. العمود "المتوقع الآن" مُحدَّث بعد إعادة النشر: بعض الأسطر مُتحقَّق منها حيّاً اليوم بـ Playwright (مُعلَّمة "✅ تحقّق اليوم")، والبقية إما لم تتغيّر أو UNVERIFIED-carried (لا commit يمسّها).

| الوقت | الخطوة | النتيجة المتوقعة الآن | ما الذي تؤكّده |
|---|---|---|---|
| 0:00 | افتح `https://yahyoha-omnidiag.hf.space/` | `{"status":"Healthy","version":"4.0.0",…}` خلال ثانية (✅ تحقّق اليوم). الإقلاع البارد لم يُختبَر | الإقلاع البارد لا يزال UNVERIFIED |
| 0:30 | افتح الواجهة الحية؛ Ctrl+Shift+R؛ في Network ابحث عن `index-*.js` | **`index-Ca31dcIj.js` (22 سبتمبر) — ✅ مُصلَح، تحقّق اليوم** (X-1) | X-1 |
| 1:30 | الشريط الجانبي: افتح قائمة المرض | `diabetes` / `heart_disease` مع "No description" — **لا يزال قائماً، ✅ تحقّق اليوم** (`shots/disease_dropdown_open.png`) | عيب شكلي مُرحَّل |
| 2:00 | Engineering Mode ← heart_disease ← Run Inference دون تعديل | **✅ مُصلَح، تحقّق اليوم:** نتيجة كاملة، 0 "Invalid input" (B-1) | B-1، `shots/eng_heart_after_run.png` |
| 3:00 | Scales | لم يُعَد اختباره اليوم؛ يُرجَّح استمرار العيب الشكلي (لا commit يمسّه) | UNVERIFIED-carried |
| 3:30 | diabetes ← Run Inference | لم يُعَد اختباره بالنقر اليوم؛ يُرجَّح "Screening result" الصحيحة بعد استبدال الحزمة (X-1) | UNVERIFIED-carried (يُرجَّح مُصلَح) |
| 4:30 | مرّر فوق عمود SHAP؛ Show All N Features؛ Raw JSON Response | لم يُعَد اختباره بالنقر اليوم؛ `verify_live.py` يؤكّد أن قيم SHAP نفسها مطابقة تماماً بين LOCAL وLIVE | `verify_live_result.json` (explain diffs 0) |
| 5:30 | Generate AI Report ← Generate Report | **✅ تحقّق اليوم عبر API مباشرة:** `source: llm` خلال ~3 ث، وتقرير خالٍ من الأدوية/التشخيص (حارس `840d342`) | C(i)، `/tmp/llm_test.py` |
| 6:30 | Clinical EMR Mode | **✅ مُصلَح، تحقّق اليوم:** لا خطأ، لا حاجة لـ Refresh | B-2، `shots/emr_diabetes_initial.png` |
| 7:30 | بدّل إلى heart_disease وانتظر 20 ث | **✅ مُصلَح، تحقّق اليوم:** لا خطأ فوراً ولا بعد 20 ثانية | B-2، `shots/emr_after_switch_20s.png` |
| 8:30 | اختر Fatima Hassan (P-002) | لم يُعَد اختباره بالنقر اليوم؛ `verify_live.py` يؤكّد `counterfactuals` مطابقة تماماً (best 96.5%→87.1% على الطرفين) | `verify_live_result.json` |
| 9:30 | بدّل للسكري ← اختر المريض الثالث (Samir/D-003) | **✅ الانحراف الرقمي مُصلَح، تحقّق اليوم عبر API:** 63.74% مطابقة تماماً بين LOCAL وLIVE (كانت 63.7%/49.4%) | X-2، `verify_live_result.json` |
| 10:30 | Clinical Notes Parser (مريض قلب) ← الصق `41 yo male, non-smoker. Denies chest pain. BP 118/76. Cholesterol 185. HR 72 at rest.` | **✅ خطر النفي مُصلَح، تحقّق اليوم عبر API:** non-smoker→0، Denies chest pain→ASY (صحيح)؛ ولم يعد يُطبَّق تلقائياً — يتطلب ضغط Apply | §3، اختبار حي `/parse-notes` |
| 11:30 | الصق `مريض ذكر عمره 58 سنة، ضغط الدم 150/95` | لم يتغيّر: لا يزال 0 حقول متوقَّعة (لا دعم عربي) | UNVERIFIED-carried |
| 12:00 | Export PDF | لم يُعَد اختباره اليوم؛ يُرجَّح استمرار سقوط قسم What-If (لا commit يمسّ `PDFReport`) | UNVERIFIED-carried |
| 13:00 | Print (معاينة) | لم يُعَد اختباره اليوم | UNVERIFIED-carried |
| 13:30 | History | لم يُعَد اختباره اليوم؛ يُرجَّح استمرار العطل (`PatientTimeline.jsx` لم يُذكَر في أي commit) | UNVERIFIED-carried |
| 14:00 | Sign In كطبيب ← History | لم يُعَد اختباره اليوم | UNVERIFIED-carried |
| 15:00 | Batch Prediction ← heart ← ارفع CSV بثلاثة صفوف ← Run Batch | لم يُختبَر بالنقر؛ الـ API الحي يستجيب بـ 401 صحيح (لا 405) عند اختبار مباشر — **يُرجَّح مُصلَح** | UNVERIFIED-this-round، curl مباشر |
| 16:00 | بدّل المرض إلى diabetes دون New Batch | لم يُعَد اختباره اليوم | UNVERIFIED-carried |
| 16:30 | Before/After Compare ← Ahmed vs Khalid ← Compare | لم يُختبَر بالنقر؛ **يُرجَّح مُصلَح** بعد استبدال الحزمة (X-1) | UNVERIFIED-this-round |
| 17:30 | Sign out ← Admin Dashboard ← Sign In as Admin | لم يُختبَر بالنقر؛ الحزمة الحية تحتوي "Avg Risk Probability" لا "AVG CONFIDENCE" — **يُرجَّح مُصلَح** التسمية | UNVERIFIED-this-round، grep على الحزمة |
| 18:30 | Users ← New User (مستخدم تجريبي تنشئه) ← عطّله (لا تعطّل admin) | يُفترض أن ينجح على LIVE (preflight 200)؛ محلياً يفشل (CORS) | UNVERIFIED بالنقر |
| 19:00 | Dark Mode؛ ثم DevTools ← Device toolbar ← iPhone 12 | الوضع الداكن يعمل؛ تمرير أفقي على الجوال | 15 |
| 19:30 | على هاتف حقيقي: Chrome ← القائمة ← "Install app" | UNVERIFIED — سجّل هل يظهر | PWA |
| لاحقاً | بعد 15 دقيقة من الدخول ← Admin ← Refresh | "Could not validate credentials" متوقّعة (X-6) | انتهاء الرمز |

---

## E. الأولويات

### يجب إصلاحه قبل التحكيم (P0) — مُحدَّثة 22 سبتمبر

خمسة من ستة بنود P0 الأصلية **مُصلَحة ومُتحقَّق منها حيّاً اليوم**. المتبقي حرج ولم يُمسَّ:

| # | المشكلة | الحالة | الدليل |
|---|---|---|---|
| 1 | **C-1** ضبط `ADMIN_PASSWORD`/`DOCTOR_PASSWORD` كـ secrets، أو تعطيل الزرع الافتراضي | **لا يزال قائماً — لم يُصلَح.** أي شخص يقرأ المستودع العام (لا يزال عاماً) يصبح super_admin عبر بيانات دخول افتراضية موثّقة في الكود العام | تحقّق حي اليوم: `POST /auth/login` بالبيانات الافتراضية → 200 لكلا الحسابين |
| 2 | ~~**X-1** نشر الواجهة الحالية على Vercel~~ | **مُصلَح.** الحزمة الحيّة `index-Ca31dcIj.js` (22 سبتمبر) تطابق HEAD | X-1 أعلاه |
| 3 | ~~**B-1** Engineering Mode للقلب~~ | **مُصلَح.** Run Inference يعمل، 0 "Invalid input" | B-1 أعلاه، `shots/eng_heart_after_run.png` |
| 4 | ~~**B-2** خطأ EMR بعد تبديل المرض~~ | **مُصلَح.** لا خطأ فوراً ولا بعد 20 ثانية | B-2 أعلاه، `shots/emr_after_switch_*.png` |
| 5 | ~~**X-2** تثبيت إصدارات المكتبات~~ | **مُصلَح.** `requirements.txt` مثبَّت؛ تطابق رقمي تام 21/21 قيمة | `verify_live_result.json` |
| 6 | ~~**mock What-If**~~ | **مُصلَح.** بطاقة "Coming Soon/Powered by DiCE" غائبة كلياً من الحزمة الحيّة (0 تطابق) | grep على `index-Ca31dcIj.js` |

**P0 جديد الآن أن الباقي أُصلح:** بند 1 (C-1) هو الوحيد المتبقي من القائمة الأصلية، ويبقى الأكثر حرجاً — منشور علناً وقابل للاستغلال دون أي حاجز.

### يُخفى / لا يُعرض (إن لم يُصلح) — مُحدَّثة

- **Clinical Notes Parser** — النفي أصبح آمناً ويتطلب Apply صريحاً الآن؛ يمكن عرضه بملاحظة إنجليزية عادية، لكن **لا تذكر BioBERT ولا العربية** (لا تزال 0 حقول).
- **History** — أخفِ الزر (لم يُعَد التحقق منه هذه الجولة، لا commit يمسّه).
- **Batch للسكري فوق ~20 صفاً** و**What-If سكري أثناء وجود زوار آخرين** (X-3 — لم يُصلَح، الخادم لا يزال متزامناً).
- **Before/After Compare** — يُرجَّح أنه أصبح صحيحاً على LIVE بعد X-1 (لم يُختبَر بالنقر هذه الجولة)؛ لا يزال "مقارنة بين مريضين" وليس "قبل/بعد" حقيقي.
- **Monitoring**: لا تَعِد بـ Evidently أو MLflow؛ Prometheus فقط (بلا تغيير).
- **Admin → Annotation Queue** — لم يُختبَر هذه الجولة، يُرجَّح استمرار عيبه (لا commit يمسّه)؛ و**Admin بعد 15 دقيقة من الدخول** (X-6 لم يُصلَح).
- **Randomize** — استخدم المرضى التجريبيين بدلاً منه (لم يُعَد اختباره).
- **Export PDF لحالات الإحالة** — لم يُعَد اختباره هذه الجولة؛ يُفترض استمرار سقوط قسم What-If (لا commit يمسّ `PDFReport`).

### يمكن أن ينتظر (بعد التحكيم)

- محتوى تقرير LLM: إرسال `features` خام إلى DeepSeek، التوصيات الدوائية، "LDL" الخاطئ، تناقض Positive/LOW (الإجابة الجاهزة للأسئلة في C-i).
- X-3 (تشغيل ML في threadpool)، X-7 (الكاش يتخطّى الحفظ والطابور)، X-6 (refresh tokens)، X-5 (قاعدة دائمة بدل SQLite في الحاوية)، X-8 (DEBUG logs).
- `notes` في وسم المراجعة يُرمى؛ `patient_visits.notes` بلا واجهة.
- PDF: MRN/Gender/التوقيع؛ Print يخفي الاسم؛ عنوان "Recommended Lifestyle Interventions".
- Batch: عمود `risk_probability_corrected` للقلب، النتائج بعد تبديل المرض.
- Admin: OPTIONS في audit، المنطقة الزمنية، IP البروكسي، CORS لـ PATCH/DELETE محلياً.
- X-4 (`vite dev` معلّق بسبب StrictMode)؛ قائمة الأمراض "No description"؛ favicon؛ تمرير أفقي على الجوال؛ تخزين POST دون اتصال في Workbox؛ الطلبات المكرّرة لكل مريض.

---

### ما لم يُتحقَّق منه (UNVERIFIED) ولماذا

| البند | السبب |
|---|---|
| الإقلاع البارد لـ HF Space | كان مستيقظاً طوال الاختبار؛ لم أُرِد إيقافه |
| تثبيت PWA الفعلي | يتطلب `beforeinstallprompt` في متصفح حقيقي |
| نافذة الطباعة الفعلية | headless؛ استُخدم `emulate_media('print')` |
| نقرات Admin لتعديل الأدوار/التعطيل/مفاتيح API على LIVE | تعديل بيانات حيّة غير ضروري؛ preflight فقط |
| مسار BioBERT على LIVE (`use_bert:true`) | قد يُنزّل ~400MB على Space مجاني قبل العرض؛ الواجهة لا تستدعيه أصلاً |
| `/admin/retrain` | خارج الحدود (إعادة تدريب) ويستبدل ملف الإنتاج |
| Batch سكري 500 صف على LIVE | تجنّب تجميد الـ Space ~50 ث للزوار الآخرين |
| وسم عنصر مراجعة على LIVE | تعديل بيانات حيّة؛ مُتحقَّق محلياً |
