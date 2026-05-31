# Ultimate Production Polish Plan

## Overview
Polish the UI/UX of Patient Parameters form (Engineering Mode) and Clinical EMR Mode for live demonstration. Fix the Sex segmented control, enhance sliders/scales with live badges, update medical tooltips with exact definitions, ensure no Arabic toggle exists, and verify zero build errors.

---

## Task 1: Fix and Beautify the Sex Field (Segmented Control)

### Files to modify:
- [`frontend/src/components/SchemaFieldFactory.jsx`](frontend/src/components/SchemaFieldFactory.jsx)
- [`frontend/src/utils/schemaToZod.js`](frontend/src/utils/schemaToZod.js)

### Changes:

**A) [`SchemaFieldFactory.jsx`](frontend/src/components/SchemaFieldFactory.jsx) — `SegmentedField` component (lines 71-115)**
- Rewrite with `bg-blue-600 text-white` active state (instead of `bg-primary-500`)
- Add smooth `duration-200` transitions and active shadow (`shadow-md`)
- Ensure proper `role="radiogroup"` ARIA attributes
- Fix `field.value ?? 0` to safely fall back to `0`
- Ensure `rounded-lg` border containment with no inner dividers that break the unified look
- Add `font-semibold` to active button text for clarity

**B) [`SchemaFieldFactory.jsx`](frontend/src/components/SchemaFieldFactory.jsx) — `useController` default value (line 312)**
- Change `defaultValue: meta.default ?? ''` to `defaultValue: meta.default ?? (meta.type === 'integer' || meta.type === 'number' ? 0 : '')`
- This ensures segmented fields get a numeric default (0) rather than an empty string

**C) [`schemaToZod.js`](frontend/src/utils/schemaToZod.js) — Add `case 'segmented'` (line 33)**
- Add a new case between `toggle` and `select`:
  ```js
  case 'segmented': {
    validator = z
      .number({ invalid_type_error: `${field.title} is required` })
      .int()
      .min(0)
      .max(1);
    break;
  }
  ```

---

## Task 2: Harmonize Sliders, Scales, and Live Value Badges

### Files to modify:
- [`frontend/src/components/SchemaFieldFactory.jsx`](frontend/src/components/SchemaFieldFactory.jsx)

### Changes:

**A) [`SliderField`](frontend/src/components/SchemaFieldFactory.jsx:149-189) — Enhance the live value badge**
- Move badge to be more prominent: render it as a styled inline `span` next to the label with `bg-blue-100 text-blue-800 font-bold px-2 py-0.5 rounded-md text-xs border border-blue-200 min-w-[2rem] text-center`
- Keep the right-side value display for slider thumb reference
- Ensure badge updates instantly when slider moves

**B) [`NumberField`](frontend/src/components/SchemaFieldFactory.jsx:197-264) — Improve synchronisation**
- Ensure `step={0.1}` for float fields (already handled at line 200 via `meta.type === 'number'`)
- Add safety: `const currentVal = field.value ?? min ?? 0` (line 204, already exists)
- Add badge for the slider value: render a small value badge next to the label when slider is shown
- Confirm two-way binding works: slider → number input (line 219) and number input → slider (line 237-250) both use `field.onChange()` with correct parsing

**C) Add a live "current value" display for slider-type fields**
- For categorical sliders (Education, Income, Age, GenHlth), the badge already exists in `SliderField` (lines 158-163)
- Enhance it with the same blue-themed style for consistency

---

## Task 3: Implement Professional Medical Tooltips (Hover Explanations)

### Files to modify:
- [`frontend/src/utils/medicalDictionary.js`](frontend/src/utils/medicalDictionary.js)

### Changes:

**Update exact definitions as specified:**

| Term | Exact Definition |
|------|-----------------|
| HighBP | "High Blood Pressure (Hypertension) — Indicator of chronic cardiovascular strain." |
| HighChol | "High Cholesterol — Elevated blood lipid levels, increasing plaque risk." |
| BMI | "Body Mass Index (BMI) — A statistical measurement of body weight relative to height." |
| RPP | "Rate Pressure Product (RPP) = Heart Rate × Systolic Blood Pressure. Measures myocardial oxygen consumption." |
| FastingBS | "Fasting Blood Sugar — Glucose level after an overnight fast; key for metabolic assessment." |
| MaxHR | "Maximum Heart Rate — Highest heart rate achieved during clinical assessment." |
| ExerciseAngina | "Exercise-Induced Angina — Chest pain brought on by physical exertion." |
| GenHlth | "General Health Rating — Self-reported or clinically assessed overall health scale." |
| MentHlth | "Mental Health Days — Number of days in the past 30 days with poor mental health or stress." |
| PhysHlth | "Physical Health Days — Number of days in the past 30 days with physical illness or injury." |
| DiffWalk | "Difficulty Walking — Indicates limited physical mobility or difficulty climbing stairs." |

Note: The current dictionary has similar definitions but the user wants exact wording — e.g., remove "and workload" from RPP, remove "stress or" from MaxHR, etc.

---

## Task 4: Remove Arabic Language Toggle

### Files to check:
- [`frontend/src/components/ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx)

### Finding:
No Arabic toggle exists in the current codebase. Search across all `frontend/src/**/*.{jsx,js}` returned zero results for Arabic/language toggle.

**Action:** No changes needed. Confirm in completion report that the interface is cleanly in English with no Arabic toggle present.

---

## Task 5: Final Quality Check

### Commands to run:
```bash
cd frontend && npm run build
```

### Verification:
1. No compile-time errors or warnings
2. All inputs render without being squished or clipped
3. Clicking 'Randomize' button fills all fields (including the Sex segmented control) with valid values
4. Slider values and badges update synchronously
5. Tooltips appear on hover with correct definitions
6. SHAP bar chart labels have working tooltips

---

## Architecture / Data Flow

```mermaid
flowchart TD
    subgraph "Schema Loading"
        A[useDiseaseSchema hook] -->|fetches JSON Schema| B[parseSchema]
        B --> C[FieldMetadata[] with component types]
    end

    subgraph "Form Rendering"
        C --> D[DynamicClinicalForm]
        D --> E[CategoryCard]
        E --> F[SchemaFieldFactory]
        F --> G{meta.component}
        G -->|segmented| H[SegmentedField - Fixed]
        G -->|slider| I[SliderField - Enhanced Badge]
        G -->|number| J[NumberField - Synced Slider+Input]
        G -->|toggle| K[ToggleField]
        G -->|select| L[SelectField]
    end

    subgraph "Validation Layer"
        C --> M[buildZodSchema]
        M --> N[Zod validation object]
        N --> O[react-hook-form resolver]
        O --> P[Form validation - NEW: segmented case]
    end

    subgraph "Tooltip System"
        Q[MedicalTooltip component] --> R[lookupMedicalTerm]
        R --> S[medicalDictionary - UPDATED definitions]
        T[ShapBarChart labels] --> R
    end

    subgraph "EMR Display"
        U[ClinicalEmrMode] --> V[Patient Data Summary]
        V --> W[MedicalTooltip on field labels]
        U --> X[SHAP Chart with tooltips]
    end
```

---

## Execution Order

1. **First**: Fix `schemaToZod.js` (add `case 'segmented'`) — this fixes the validation bug
2. **Second**: Fix `SchemaFieldFactory.jsx` SegmentedField styling + useController default
3. **Third**: Enhance SliderField and NumberField badges/Synchronisation
4. **Fourth**: Update medicalDictionary.js with exact definitions
5. **Fifth**: Verify no Arabic toggle exists (none found)
6. **Sixth**: Run build and verify zero errors
