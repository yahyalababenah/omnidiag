# Criterion 01 — Problem Statement Layer Audit

**Scope:** claims about *why the problem is real*. Code, models, and metrics were **not** audited.
**Date:** 2026-09-16
**Verdict:** ❌ **FAIL** — the submitted proposal contains **zero citations**. Every epidemiological claim in it is unreferenced.

---

## 0. Scope Resolution — What Was Actually Audited

The file set named in the brief did not map cleanly onto the repo. Resolution:

| Brief says | Actual artifact | Status |
|---|---|---|
| "the submitted proposal" | `/home/yahia/Desktop/lana/OmniDiag_Proposal_AI_Expo_Jordan_2026_Updated.docx` | ⚠️ **Outside the repo.** Only artifact containing the 75.6% / 77.5% / 200-decisions claims. Treated as the submitted proposal. |
| — | `Student_Proposal_Template.docx` (repo root) | ❌ Blank IEEE template. No content. Not the submission. |
| `docs/OmniDiag_Proposal_Defense.md` | present, 444 lines, Arabic | ✅ In scope. Sole holder of the R1–R49 library. |
| `README.md` | present, 642 lines | ✅ In scope. **Contains no problem-statement claims whatsoever** (see §1.4). |
| "poster/slide/pitch drafts" | none exist | ❌ No poster, slide deck, or pitch file in the repo. |
| — | `plans/omnidiag_problem_analysis_report.md` (tracked, 550+ lines) | ✅ **Pulled into scope.** It is the repo's de-facto problem-statement document and contradicts the proposal. |
| — | `plans/ai_expo_2026_proposal.md` (deleted; recoverable at `git show 6c81963:`) | ⚠️ **Pulled into scope.** Superseded proposal draft, still public in git history, with conflicting claims. |

**Citation convention for the .docx:** it has no page or line numbers. References are given as `§section ¶N`, where N is the paragraph index produced by extracting `word/document.xml` paragraph-wise. Reproduce with:

```bash
python3 -c "
import zipfile; from xml.etree import ElementTree as ET
W='{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
z=zipfile.ZipFile('OmniDiag_Proposal_AI_Expo_Jordan_2026_Updated.docx')
r=ET.fromstring(z.read('word/document.xml'))
print('\n'.join(''.join(t.text or '' for t in p.iter(W+'t')) for p in r.iter(W+'p')))" | cat -n
```

---

## 1. Claims Table

### 1.0 The headline finding

The submitted proposal contains **0 reference IDs, 0 DOIs, 0 author-year citations, 0 "et al.", and no References section.** Verified:

```
$ grep -cE "\[R[0-9]+\]|doi|DOI|et al|References" submitted_proposal.txt
0
```

The R1–R49 library exists **only** in `docs/OmniDiag_Proposal_Defense.md` (lines 300–398), which is an internal Arabic prep document that no judge will read. Consequently **every row in the table below is flagged unreferenced-in-the-proposal** (🚩). The "Ref ID" column records what the defense doc *retroactively* maps the claim to — that mapping is not visible to a reader of the proposal.

### 1.1 Submitted proposal — §2 Problem Statement

| # | Claim text | Location | Ref ID cited *in the proposal* | Ref ID mapped *in defense doc* | Conflicting number elsewhere? |
|---|---|---|---|---|---|
| C1 | "an overall burnout rate of **75.6%** among emergency healthcare providers in Jordan" | proposal §2.1 ¶38 | 🚩 **none** | R1 Alwidyan 2026 (`Defense.md:21`) | NO |
| C2 | "rising to **77.5%** among medical residents" | proposal §2.1 ¶38 | 🚩 **none** | R2 Nimer 2021 (`Defense.md:22`) | NO |
| C3 | "**Eight out of ten residents** routinely exceed **24** consecutive working hours" | proposal §2.1 ¶38 | 🚩 **none** | ⚠️ none — `Defense.md:28-34` states this is a **secondary citation inside** Nimer, not a Nimer finding. Original source unnamed. | NO (but provenance broken) |
| C4 | "weekly workloads reaching up to **100 hours**" | proposal §2.1 ¶38 | 🚩 **none** | ⚠️ **none — the defense doc never addresses this number at all.** Wholly undefended. | NO |
| C5 | "a clinician makes **over 200 decisions per shift**" | proposal §2.1 ¶38 | 🚩 **none** | `Defense.md:38-42` — explicitly flagged "أضعف رقم في البروبوزل" (weakest number), **no peer-reviewed source** | ✅ **YES — three-way divergence, see §5.1** |
| C6 | "Clinical staff operate under sustained cognitive load — **a systemic driver of diagnostic error**" | proposal §2.1 ¶38 | 🚩 **none** | — | NO |
| C7 | "Chronic fatigue **depletes cognitive reserve** and **correlates directly with** execution-level slips" | proposal §2.1 ¶38 | 🚩 **none** | R3/R4/R5 (`Defense.md:57`) | NO |
| C8 | "fatigue becomes **a primary driver** of preventable diagnostic failure" | proposal §2.1 ¶38 | 🚩 **none** | R3 Hodkinson 2022 | NO |
| C9 | "Healthcare systems in Jordan and the wider MENA region face a **reliability crisis** in which clinical infrastructure has not kept pace with patient volume" | proposal §2.1 ¶38 | 🚩 **none** | — | NO (no number given — unfalsifiable) |
| C10 | "Definitive cardiac diagnosis depends on coronary angiography and nuclear perfusion imaging — modalities **confined to tertiary cardiac centres**" | proposal §2.2 ¶40 | 🚩 **none** | — | NO |
| C11 | "Definitive diabetes diagnosis depends on laboratory glycaemic testing. **Neither is routinely available at the primary-care level.**" | proposal §2.2 ¶40 | 🚩 **none** | — | NO. ⚠️ Overclaim: point-of-care HbA1c and capillary glucose are primary-care modalities. |
| C12 | "a patient presenting at a rural or peripheral primary healthcare centre is **not risk-stratified at all**… in practice, it does not happen" | proposal §2.2 ¶40 | 🚩 **none** | — | NO. ⚠️ Absolute claim, no Jordanian data offered. |
| C13 | "The canonical UCI Cleveland benchmark **achieves its headline accuracy using** `ca` and `thal`" | proposal §2.3 ¶42 | 🚩 **none** | ⚠️ `Defense.md:246` explicitly warns: *"لا تدّعِ أن ورقة منشورة أثبتت أن نتائج Cleveland مضخّمة"* — **do not claim a published paper established this.** The proposal states it as established fact. | NO |

### 1.2 Defense doc — claims that exist *only* there

These are the properly-referenced claims. None of them reached the proposal.

| # | Claim text | Location | Ref ID |
|---|---|---|---|
| C14 | Diagnostic error 5.08% in outpatient settings (~12M US adults/yr) | `Defense.md:81` | R8 Singh 2014 |
| C15 | 795,000 people/yr permanently disabled or dead from diagnostic error | `Defense.md:82` | R9 Newman-Toker 2024 |
| C16 | Diagnostic errors contribute to ~10% of deaths | `Defense.md:83` | R10 National Academies 2015 |
| C17 | 23% of patients who died or were transferred to ICU experienced a diagnostic error | `Defense.md:84` | R11 Auerbach 2019 |
| C18 | Radiologist accuracy **0.926 → 0.806** after night shift (12 radiologists) | `Defense.md:93` | R12 Hanna 2018 |
| C19 | +34% time-to-first-fixation on fracture post-shift | `Defense.md:93` | R12 |
| C20 | 18,488 preliminary interpretations; discordance rises toward shift end | `Defense.md:93` | ⚠️ **ambiguous** — R12–R15 cited as a block, no per-number mapping. Likely R14 Krupinski 2010. |
| C21 | 117,402 reports; end-of-shift reports 16% lower mean similarity | `Defense.md:93` | R13 Vosshenrich 2021 |
| C22 | Jordan diabetes prevalence **13.0% (1994) → 23.7% (2017)** | `Defense.md:106` | R16 Abu-Raddad 2020 |
| C23 | ~1/3 of adults with diabetes in MENA are undiagnosed | `Defense.md:106` | R17 IDF Atlas 2025 |
| C24 | CVD causes >40% of deaths in Jordan | `Defense.md:108` | R18 Al-Ajlouni 2024 |
| C25 | Burnout ≈2× odds of patient-safety incidents, **OR 2.04 (95% CI 1.69–2.45)**, 170 observational studies, 239,246 physicians; 31/39 studies self-reported | `Defense.md:51` | R3 Hodkinson 2022 |
| C26 | DDI alert override rate ~90% | `Defense.md:397` | R49 Felisberto 2024 |
| C27 | QRISK3 AUC 0.86–0.88 internal → ~0.70–0.72 on UK Biobank | `Defense.md:262` | R35 / R34 |
| C28 | FINDRISC AUC 0.85 → 0.65–0.78 externally | `Defense.md:262` | R36 |

### 1.3 `plans/omnidiag_problem_analysis_report.md` — repo-visible, all unreferenced

| # | Claim text | Location | Ref ID | Conflict? |
|---|---|---|---|---|
| C29 | "Clinicians juggle **50+ data points** per patient across **20+ patients per shift**" | `plans/omnidiag_problem_analysis_report.md:26` | 🚩 none | NO |
| C30 | "Pattern recognition **degrades after hour 8**" | `plans/omnidiag_problem_analysis_report.md:26` | 🚩 none | NO |
| C31 | "**~30% of MIs** present without chest pain" | `plans/omnidiag_problem_analysis_report.md:27` | 🚩 none | ✅ **YES — see §5.2** |
| C32 | "**25-40% of MIs in diabetic patients** present without chest pain" | `plans/omnidiag_problem_analysis_report.md:146` | Canto et al., JAMA 2000 (only citation in the file) | ✅ **YES — conflicts with C31 in the same document** |
| C33 | "A typical emergency physician processes **200+ clinical decisions per shift**" | `plans/omnidiag_problem_analysis_report.md:38` | Croskerry, 2009 (no DOI, not in R1–R49) | ✅ **YES — see §5.1** |
| C34 | "each UI friction point… adds **200-500ms** of cognitive interrupt" | `plans/omnidiag_problem_analysis_report.md:74` | 🚩 none | NO |
| C35 | "human visual system processes colour… **in under 200ms**" | `plans/omnidiag_problem_analysis_report.md:92` | 🚩 none | NO |
| C36 | "assess a patient's risk profile in **under 2 seconds**… versus **15-30 seconds** to parse a traditional numeric EMR display" | `plans/omnidiag_problem_analysis_report.md:92` | 🚩 none | NO. ⚠️ Reads as invented; no study, no measurement method. |

### 1.4 README.md — nil return

`README.md` contains **no burnout, fatigue, prevalence, diagnostic-error, decision-count, or target-setting claim of any kind.** Swept for: `burnout`, `fatigue`, `non-invasive`, `primary care`, `emergency`, `general practitioner`, `outpatient`, `prevalence`, `diagnostic error`, `triage`, `underserved`, `cardiologist`, `Jordan`, `MENA`, `tertiary`, `invasive`. All zero hits outside code identifiers (`doctor` as an RBAC role string).

> **This is itself a Criterion-1 defect.** The repository's public front door — the artifact judges will actually open — states no problem, cites no evidence, and names no clinical setting. The entire problem-statement layer lives in one Arabic prep file and one deleted planning document.

### 1.5 `plans/ai_expo_2026_proposal.md` (deleted, live in git history)

| # | Claim text | Location | Ref ID |
|---|---|---|---|
| C37 | "Jordan's healthcare system… faces a **severe shortage of specialized cardiologists** relative to population needs" | `git show 6c81963:plans/ai_expo_2026_proposal.md:24` | 🚩 none |
| C38 | "a general practitioner may spend **20-30 minutes** evaluating a patient with ambiguous cardiac symptoms" | `…:286` | 🚩 none |
| C39 | "For a clinic seeing **50 patients daily**" | `…:286` | 🚩 none |
| C40 | "global AI in healthcare market is projected to exceed **$188 billion by 2030**" | `…:301` | 🚩 none |

---

## 2. Retracted Source Sweep

Searched the full repo (excluding `node_modules/`, `.venv/`, `.git/`) plus both proposal artifacts for `Panagioti`, `10.1001/jamainternmed.2018.3713`, `178(10)`.

| File:line | Hit | Assessment |
|---|---|---|
| `docs/OmniDiag_Proposal_Defense.md:53` | "ورقة **Panagioti et al. (2018)**… **مسحوبة رسمياً (Retracted)**… **لا تستشهد بها أبداً**" | ✅ **NOT a citation.** A do-not-cite warning naming the retraction (July 2020, Manchester inquiry) and directing to R3 Hodkinson 2022. |
| `docs/OmniDiag_Proposal_Defense.md:401` | Full reference + `178(10), 1317–1331` under heading **🚫 مرجع محظور** ("Forbidden Reference"), with retraction notice `doi: 10.1001/jamainternmed.2020.1755` | ✅ **NOT a citation.** A quarantine entry, deliberately listed so it is recognised and avoided. |

**Result: ✅ CLEAN.**
- Submitted proposal: **0 hits**.
- `README.md`: **0 hits**.
- `plans/` (incl. deleted proposal): **0 hits**.
- The DOI `10.1001/jamainternmed.2018.3713` appears **nowhere in the repo**.

The two defense-doc hits are the correct handling of a retracted source and **should not be removed** — deleting them would remove the only warning that stops the paper being re-introduced. If the brief requires literal zero occurrences of the string, replace the author name with a redacted form (e.g. `[RETRACTED — JAMA Intern Med 2018]`) but keep the warning.

---

## 3. Causal Language Sweep

Sentences linking fatigue/burnout to error, classified.

### 3.1 CAUSAL — must be rewritten

| # | Sentence | Location | Trigger |
|---|---|---|---|
| **L1** | "Clinical staff operate under sustained cognitive load — **a systemic driver of** diagnostic error." | proposal §2.1 ¶38 | "driver of" |
| **L2** | "Chronic fatigue **depletes** cognitive reserve **and correlates directly with** execution-level slips during the diagnostic process." | proposal §2.1 ¶38 | "depletes" (causal mechanism asserted); "correlates **directly**" — the adverb converts an association into a directional claim |
| **L3** | "In a setting where a clinician makes over 200 decisions per shift, **fatigue becomes a primary driver of** preventable diagnostic failure." | proposal §2.1 ¶38 | "becomes a primary driver of" — strongest causal claim in the document |
| **L4** | "**Clinicians operating under documented fatigue** must triage patients in settings where…" | proposal §2.4 ¶46 | presupposes the causal chain as settled |
| **L5** | "Cognitive overload **causes** clinicians to miss atypical patterns." | `plans/omnidiag_problem_analysis_report.md:30` | "causes" |
| **L6** | "Under fatigue, cognitive heuristics **degrade into** systematic biases" | `plans/omnidiag_problem_analysis_report.md:38` | "degrade into" |
| **L7** | "### 2.3 Summary: **Smooth UI → Fewer Diagnostic Errors**" | `plans/omnidiag_problem_analysis_report.md:124` | arrow = asserted causation, in a section heading |
| **L8** | "**The causal chain is:** Smooth UI → Reduced cognitive micro-delays → Less mental fragmentation → More cognitive capacity → **Fewer anchoring/availability bias errors**" | `plans/omnidiag_problem_analysis_report.md:126-133` | explicitly labelled "causal chain"; four unsupported causal links, terminating in a diagnostic-error claim |
| **L9** | "**This is not theoretical.** The accumulation of dozens of small frictions across a shift is precisely what **causes** 'charting fatigue'" | `plans/omnidiag_problem_analysis_report.md:136` | "causes", plus an explicit denial of speculativeness with no evidence |
| **L10** | "Pattern recognition **degrades after hour 8**." | `plans/omnidiag_problem_analysis_report.md:26` | asserted dose-response with no source |

**L1–L4 are in the submitted document and are the priority.** L3 is the single most exposed sentence: it asserts fatigue as a *primary driver* of diagnostic failure while resting on a decision count (C5) that the team's own defense doc concedes has no peer-reviewed basis.

### 3.2 ASSOCIATIVE — acceptable

| Sentence | Location |
|---|---|
| "أدّعي **ارتباطاً، لا سببية**" ("I claim association, not causation") — with OR 2.04 (1.69–2.45), self-report and cross-sectional limitations both disclosed | `docs/OmniDiag_Proposal_Defense.md:51` |
| "الأدبيات في هذا الباب **متنازع عليها**… قل '**متسق مع**' لا '**يثبت**'" ("say *consistent with*, not *proves*"), citing the 2025 *Communications Psychology* null finding on decision fatigue | `docs/OmniDiag_Proposal_Defense.md:334` |

### 3.3 The defense doc targets a phrase that does not exist

`docs/OmniDiag_Proposal_Defense.md:48` states:

> **⚠️ البروبوزل يكتب "establishing a significant clinical link". هذه صياغة سببية لا تسندها المراجع.**
> (*"The proposal writes 'establishing a significant clinical link'. This is causal phrasing the references do not support."*)

**That string appears nowhere** — not in the submitted proposal, not in the deleted draft, not in the README, not in `plans/`. Verified: `grep -rn "significant clinical link"` returns only `Defense.md:48` itself.

**Consequence:** the Q&A prep is calibrated against a phrase that was already removed, and therefore **does not prepare a defence for L1, L2, or L3 — the causal sentences that are actually in the submitted text.** A judge quoting "*a primary driver of preventable diagnostic failure*" would hit an unrehearsed question. Fix the defense doc to quote the real sentences.

---

## 4. Target-User Consistency

### 4.1 Every statement of who / where

| Location | Statement | Setting |
|---|---|---|
| proposal title ¶5, ¶9 | "A Trust-Centric, Explainable AI Platform for **Non-Invasive Clinical Triage**" | unspecified |
| proposal §2.1 ¶38 | evidence base = "**emergency healthcare providers** in Jordan… **medical residents**" | 🔴 **emergency / residency** |
| proposal §2.2 ¶40 | "a patient presenting at a **rural or peripheral primary healthcare centre**" | 🟢 primary care |
| proposal §2.4 ¶46 | "Clinicians operating under documented fatigue must triage patients" | unspecified |
| proposal §3.2 ¶61 | "inputs a **general practitioner** can obtain in five minutes" | 🟢 primary care |
| proposal §3.4 ¶69 | "**General practitioner / primary-care physician**" → Clinical EMR Mode | 🟢 primary care |
| proposal §3.4 ¶72 | "Reviewing clinician (annotator)" → Review Queue | unspecified |
| proposal §3.4 ¶75 | "ML engineer / evaluator" → Engineering Mode | n/a |
| proposal §3.4 ¶78 | "Administrator / Platform Manager" → Admin Dashboard | n/a |
| proposal §3.4 ¶81 | "Epidemiologist / Clinical Researcher" → Batch Prediction | n/a |
| proposal §8.3 ¶371 | "the reason OmniDiag targets **primary care rather than tertiary cardiology**" | 🟢 primary care |
| proposal §10.1 ¶396 | "deployable precisely where diagnostic infrastructure is absent — **primary and comprehensive healthcare centres, rather than tertiary hospitals**" | 🟢 primary care |
| proposal §10.1 ¶398 | "Patients in **primary care**" | 🟢 primary care |
| proposal §10.1 ¶399 | "**General practitioners** operating under **the burnout load documented in Section 2**" | 🔴 **cross-wired** |
| proposal §10.3 ¶409 | "**Family Medicine / Outpatient department of a Jordanian university hospital** — Jordan University Hospital" | 🟡 outpatient dept of a tertiary centre |
| proposal §10.4 ¶413 | "**Ministry of Health primary healthcare network**" | 🟢 primary care |
| `plans/omnidiag_problem_analysis_report.md:22` | "**Emergency medicine** operates under a triad of compounding pressures" | 🔴 **ED** |
| `…:26` | "20+ patients **per shift**" | 🔴 ED |
| `…:38` | "A typical **emergency physician**" | 🔴 ED |
| `…:92` | "An **emergency physician** can assess a patient's risk profile in under 2 seconds" | 🔴 ED |
| `…:107` | "In a **busy ED**" | 🔴 ED |
| `…:208` | "In **emergency medicine**, knowing when to be uncertain…" | 🔴 ED |
| `…:522` | "In a **busy ED**, this could generate significant alert fatigue" | 🔴 ED |
| `git show 6c81963:plans/ai_expo_2026_proposal.md:14` | "Target Users: **Cardiologists**, general practitioners, and clinical researchers in underserved regions" | 🔴 **includes cardiology** |
| `…:24` | "initial triage decisions often fall to **general practitioners**" | 🟢 primary care |
| `…:286` | "a typical Jordanian **primary-care clinic** without on-site cardiology coverage" | 🟢 primary care |

### 4.2 Flagged disagreements

**🔴 CONFLICT 1 — the repo's problem document is set in the ED; the proposal is set in primary care.**
`plans/omnidiag_problem_analysis_report.md:22` opens *"Emergency medicine operates under a triad of compounding pressures that OmniDiag's architecture directly addresses"* and sustains the ED frame throughout (lines 26, 38, 92, 107, 208, 522). The proposal says the opposite at §8.3 ¶371 and §10.1 ¶396: primary and comprehensive health centres, **not** tertiary. An emergency department is not primary care. Both documents are in the public repo. A judge reading both sees two different products.

**🔴 CONFLICT 2 — burnout evidence is silently transferred from ED staff to GPs.**
§2.1 ¶38 measures burnout in **emergency healthcare providers** (75.6%) and **medical residents** (77.5%). §10.1 ¶399 then writes *"General practitioners operating under **the burnout load documented in Section 2**"* — but Section 2 documents no burnout load for general practitioners. The defense doc concedes this at `Defense.md:63-67`: *"لا توجد بيانات أردنية منشورة عن عبء العمل أو الاحتراق في الرعاية الأولية تحديداً، وهذه فجوة أُصرّح بها"* (*"there is no published Jordanian data on workload or burnout in primary care specifically; this is a gap I declare"*). **The concession exists only in the Arabic prep file. The proposal makes the transfer silently.** This is the load-bearing weakness of the whole problem statement: the evidence and the deployment target are in different settings.

**🔴 CONFLICT 3 — target users changed across versions and the old list is still public.**
The deleted draft names **"Cardiologists"** as the first target user (`…:14`); the submitted proposal explicitly positions against tertiary cardiology (§8.3 ¶371). The draft remains retrievable from git history at commit `6c81963`.

**🟡 CONFLICT 4 — the first-adopter site contradicts the deployment thesis.**
§2.2 ¶40 grounds the need in the **"rural or peripheral primary healthcare centre"**; §10.1 ¶396 promises deployment **"rather than tertiary hospitals"**; §10.3 ¶409 then names **Jordan University Hospital**, a tertiary academic centre, as the first adopter. Defensible (an outpatient Family Medicine clinic inside a tertiary hospital is primary-care-like), but it is not defended in the text, and it is the exact seam where the infrastructure-gap argument is weakest — a university hospital has the angiography suite the proposal says is unavailable.

### 4.3 "Non-invasive" — every occurrence and what the sentence claims

| # | Location | Sentence claims | Assessment |
|---|---|---|---|
| N1 | proposal title ¶5 / ¶9 | "Platform for **Non-Invasive** Clinical Triage" | 🔴 Asserts the whole platform is non-invasive. True of the diabetes module only. |
| N2 | proposal §2.4 ¶46 | "a risk-stratification layer that requires **no invasive test, no laboratory, and no imaging suite**" | 🔴 **False for the CAD module.** Its declared inputs include **Cholesterol** (venous lipid panel = laboratory), **RestingECG**, **ExerciseAngina**, **Oldpeak**, **ST_Slope** (exercise stress testing). |
| N3 | proposal §3.1 ¶50 | "performs **non-invasive** risk triage. A clinician enters… **cholesterol, resting ECG interpretation, exercise-related symptoms**…" | 🔴 **Self-contradictory inside one sentence.** It claims non-invasiveness and then lists a lab value and an ECG that ¶46 says are not required. |
| N4 | proposal §3.2 ¶58 | heading: "The Central Design Decision — **Non-Invasive by Construction**" | 🔴 Overclaim. The construction removed `ca`/`thal` only. |
| N5 | proposal §3.2 ¶61 | "runs on inputs a GP can obtain in **five minutes with a blood-pressure cuff, a scale, and a questionnaire**" | 🔴 **The proposal's most falsifiable sentence.** Flagged in the team's own defense doc as *"أخطر تناقض نصّي في البروبوزل"* — the most dangerous textual contradiction (`Defense.md:130`). No cuff-scale-questionnaire combination yields Oldpeak or ST_Slope. |
| N6 | proposal §3.3 ¶63 | "Once **invasive features are removed**, the remaining signal is weak…" | ✅ **Accurate.** Scoped to `ca`/`thal`. |
| N7 | proposal §4.9 ¶176 | "deployable in a setting with **no invasive diagnostic infrastructure**" | 🟡 Defensible under the narrow (angiography/nuclear) definition; inconsistent with N2's "no laboratory". |
| N8 | proposal §5.4 ¶235 | "The current **non-invasive** model achieves ROC-AUC 0.856" | 🔴 Applies the label to the CAD model specifically — the module where it is least true. |
| N9 | proposal §8.1 ¶348-350 | table row "**Invasive tests required: None / None**" | 🔴 Strongest and least qualified form. Venipuncture for a lipid panel is invasive under any ordinary clinical definition. Presented as a verified metric beside accuracy and AUC. |
| N10 | proposal §10.1 ¶396 | "engineered to be **non-invasive by construction**, it runs on inputs a GP can obtain in five minutes with **a blood-pressure cuff and a questionnaire**" | 🔴 Repeats N5 — and **silently drops "a scale"**, which ¶61 includes. Two different equipment lists for the same claim. |

**Defense-doc position (`Defense.md:132-134`):** the team already knows. The recommended repair is to replace *"Non-invasive by construction"* with ***"Deployment-feasible by design"***, and to state plainly that the diabetes module is fully non-invasive (BRFSS questionnaire, zero labs) while the CAD module requires standard **non-tertiary** cardiac workup — resting ECG, exercise test, lipid panel — with only coronary angiography and thallium scanning deliberately excluded. **That repair has not been applied to any document.** It exists in one Arabic file. The 10 occurrences above are all still live.

---

## 5. Cross-Document Number Diff

### 5.1 "Decisions per shift" — three values, three populations, three provenances

| Source | Number | Population | Unit | Citation |
|---|---|---|---|---|
| proposal §2.1 ¶38 | **over 200** | "a clinician" (generic) | per **shift** | 🚩 none |
| `plans/omnidiag_problem_analysis_report.md:38` | **200+** | "a typical **emergency physician**" | per **shift** | Croskerry, 2009 (no DOI; absent from R1–R49) |
| `docs/OmniDiag_Proposal_Defense.md:40` | **over 230** | "**ICU shift leaders**" | per **day** | unnamed ICU literature; doc states no peer-reviewed source generalises the figure |

Three separate defects in one claim: the **value** differs (200 vs 230), the **unit** differs (shift vs day), and the **population** silently widens from ICU shift leaders → emergency physicians → "a clinician" in a document whose deployment target is **primary care**. The defense doc's own instruction is *"لا تدافع عن هذا الرقم"* — do not defend this number (`Defense.md:42`). It is nonetheless load-bearing in §2.1, where it sets up the causal sentence L3.

### 5.2 "MIs without chest pain" — contradicts itself within one file

| Location | Number | Population |
|---|---|---|
| `plans/omnidiag_problem_analysis_report.md:27` | **~30%** | **all MIs** |
| `plans/omnidiag_problem_analysis_report.md:146` | **25-40%** | **MIs in diabetic patients** (Canto et al., JAMA 2000) |

Line 27 takes a diabetic-subgroup range, collapses it to a point estimate, generalises it to all myocardial infarctions, and drops the citation — 119 lines above the correctly-scoped version.

### 5.3 Equipment list for the same non-invasive claim

| Location | Equipment named |
|---|---|
| proposal §3.2 ¶61 | blood-pressure cuff, **a scale**, a questionnaire |
| proposal §10.1 ¶396 | blood-pressure cuff, a questionnaire |

### 5.4 Numbers outside the Criterion-1 remit

Reported because instruction 5 asks for *any* number differing across files. **Not audited** — models and metrics were out of scope per the brief. Listed so the divergence is on record.

| Number | Submitted proposal | README.md | `plans/ai_expo_2026_proposal.md` (git `6c81963`) | `docs/OmniDiag_Proposal_Defense.md` |
|---|---|---|---|---|
| CAD `n_estimators` | **1124** (§4.1 ¶90) | **898** (`README.md:353`) | **898** (`:81`) | — |
| CAD `learning_rate` | **0.0063** | — | **0.0136** | — |
| CAD `subsample` | **0.9637** | — | **0.964** | register: real value **0.8331** (`Defense.md:429`) |
| CAD `colsample_bytree` | **0.5455** | — | **0.545** | register: real value **0.4792** (`Defense.md:429`) |
| Optuna trials (CAD) | **300** (§4.1 ¶91) | — | **100** (`:77`) | — |
| CAD ROC-AUC | **0.856** (§8.1) | — | **0.957** full-fit / **0.919** CV (`:90-91`) | 0.856 |
| CAD accuracy | **80.17%** (§8.1) | — | **88.98%** (`:89`) | — |
| CAD sample size | **605** merged (§5.1) | — | **303 + 303**, unmerged (`:132-133`) | 605 |
| CAD feature count | **16** (§5.1) | — | **14** base / **24** engineered (`:132`, `:147`) | register: **18** per `diagnostic_v5_ab.json` (`Defense.md:433`) |

`README.md:353` still carries **898 estimators** — the superseded v1 value — while the submitted proposal states 1124. README and proposal disagree on the live model's configuration.

---

## 6. Summary of Findings

| # | Finding | Severity |
|---|---|---|
| F1 | Submitted proposal has **zero citations**. All 13 problem-statement claims (C1–C13) are unreferenced. The R1–R49 library reaches no reader-facing document. | 🔴 **Critical** |
| F2 | `README.md` — the repo's public front door — contains **no problem statement and no evidence at all**. | 🔴 **Critical** |
| F3 | Burnout evidence is measured in **ED providers and residents** but attributed to **general practitioners** (§10.1 ¶399). The gap is conceded only in an internal Arabic file. | 🔴 **Critical** |
| F4 | Three causal sentences in the submitted text (L1, L2, L3) assert fatigue → diagnostic failure. Strongest available evidence (R3, OR 2.04) is associative and cross-sectional. | 🔴 **Critical** |
| F5 | "Non-invasive" asserted **10 times**; false for the CAD module in **7** of them. §3.1 ¶50 self-contradicts within a single sentence; §8.1 presents "Invasive tests required: None" as a verified metric. | 🔴 **Critical** |
| F6 | "Over 200 decisions per shift" — three values, three populations, three unit conventions; team's own doc says do not defend it, yet it anchors the causal claim. | 🟠 **High** |
| F7 | `plans/omnidiag_problem_analysis_report.md` frames the entire problem as **emergency medicine**, contradicting the proposal's primary-care thesis. Publicly visible. | 🟠 **High** |
| F8 | Defense doc prepares a rebuttal for `"establishing a significant clinical link"` — a phrase that does not exist — and therefore prepares **no** rebuttal for the causal sentences that do. | 🟠 **High** |
| F9 | "100 hours weekly" (C4) appears in the proposal with no source and is not addressed anywhere in the defense doc. | 🟠 **High** |
| F10 | C31 vs C32: "~30% of all MIs" vs "25-40% of diabetic MIs" — 119 lines apart in one file, citation dropped from the generalised version. | 🟡 **Medium** |
| F11 | C13 states as established fact a claim the defense doc explicitly says must be presented as the team's own inference. | 🟡 **Medium** |
| F12 | Superseded proposal draft with conflicting target users and metrics remains public in git history (`6c81963`). | 🟡 **Medium** |
| F13 | `README.md:353` states 898 estimators; proposal states 1124. | 🟡 **Medium** |
| F14 | Equipment list differs between two statements of the same claim (¶61 vs ¶396). | 🟢 **Low** |
| F15 | Retracted-source sweep **clean**. The two `Panagioti` hits are do-not-cite warnings and should be retained. | ✅ **Pass** |

---

## 7. Note on Artifact Location

The submitted proposal is **not in the repository.** It lives at `/home/yahia/Desktop/lana/OmniDiag_Proposal_AI_Expo_Jordan_2026_Updated.docx`. The version that *is* recoverable from the repo (`git show 6c81963:plans/ai_expo_2026_proposal.md`) is a different, superseded document with conflicting target users, conflicting metrics, and no burnout section. Anyone auditing this project from the GitHub repository alone cannot see the submitted problem statement.

---

## تحديث لاحق — 21-09-2026

**كل رقم وجدول أعلاه (بما فيها §5's "CAD sample size 605" وF13's "898 estimators") يعكس
حالة الوثائق بتاريخ هذا التدقيق (2026-09-16) ولم يُعدَّل** — هذا سجل تاريخي لتناقض وُجد فعلاً
بين الوثائق وقتها، لا وصفاً للحالة الحالية. تحديداً: F13 تستشهد بـ`README.md:353` كما كان
حينها؛ ذلك السطر تغيّر منذ ذلك التاريخ (راجع تحديث README.md اليوم) — **التناقض الذي وثّقه
F13 لم يعد قائماً بنفس الصياغة**، لكن هذا لا يُبطل الملاحظة الأعمق (تضارب الوثائق مع بعضها
عبر الزمن)، فقط يعني أن الأرقام المحدَّدة المذكورة الآن تاريخية.

منذ 2026-09-16 استُبدل نموذج القلب بالكامل ببيانات UCI الأربعة المواقع (920 مريضاً، لا 605)
وصُحِّح انتشار السكري المنشور (0.237 بدل 0.14). لا علاقة لهذين التغييرين بمضمون هذا التدقيق
تحديداً (طبقة بيان المشكلة والاستشهادات، لا الكود أو النماذج) — مذكوران هنا فقط لأن السطر
286/287 يستشهد برقم "605" الذي صار الآن رقماً تاريخياً أيضاً على مستوى الكود، لا الوثيقة
المدقَّقة وحدها. الأرقام التقنية الحالية الصحيحة موثَّقة في `AUDIT_REPORT.md` (قسم "تحديث
لاحق — 21-09-2026") و`WEAKNESS_REGISTER.md`.

---

*Audit performed 2026-09-16. Problem-statement layer only. No code, model, or metric was evaluated. No file was modified. No commit was made.*
