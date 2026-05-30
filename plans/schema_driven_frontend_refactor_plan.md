# Schema-Driven Multi-Disease Frontend — Architectural Plan

## Table of Contents
1. [Global Context & Routing (The Skeleton)](#1-global-context--routing-the-skeleton)
2. [The Dynamic Form Engine (The Core)](#2-the-dynamic-form-engine-the-core)
3. [UI/UX Preservation & Adapting (The Paint)](#3-uiux-preservation--adapting-the-paint)
4. [Safety & Loading States (The Guardrails)](#4-safety--loading-states-the-guardrails)
5. [File-by-File Refactoring Roadmap](#5-file-by-file-refactoring-roadmap)
6. [Backend Schema Enhancement (Non-Breaking)](#6-backend-schema-enhancement-non-breaking)

---

## 1. Global Context & Routing (The Skeleton)

### Current State
- [`App.jsx`](frontend/src/App.jsx) has a static sidebar with two hardcoded views: *Engineering Mode* and *Clinical EMR Mode*
- Both [`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx:65) and [`EngineeringMode.jsx`](frontend/src/components/EngineeringMode.jsx:87) hardcode `'heart_disease'` in their API calls
- No global disease state exists

### Proposed Architecture

#### A. `DiseaseProvider` — React Context (`frontend/src/context/DiseaseContext.jsx`)

New file providing:

```
┌─────────────────────────────────────────────────┐
│  DiseaseProvider                                 │
│  ─────────────────                               │
│  state: {                                        │
│    selectedDisease: string,                      │
│    availableDiseases: DiseaseInfo[],             │
│    diseaseConfigs: Record<string, object>,       │
│    isLoading: boolean,                           │
│    error: Error | null                           │
│  }                                               │
│  actions: {                                      │
│    setDisease(diseaseName): void,                │
│    refreshDiseases(): Promise<void>              │
│  }                                               │
└─────────────────────────────────────────────────┘
```

**On mount**, `DiseaseProvider` calls:
1. `GET /api/v4/diseases` — list all registered diseases
2. `GET /api/v4/{disease}/schema` for each disease — pre-fetch schemas for instant form rendering

**DiseaseInfo** type:
```typescript
interface DiseaseInfo {
  name: string;           // e.g. "heart_disease"
  display_name: string;   // e.g. "Coronary Artery Disease Risk"
  description: string;
  version: string;
}
```

#### B. `DiseaseSelector` Component (`frontend/src/components/DiseaseSelector.jsx`)

A polished dropdown/tabs component placed in the **sidebar header** of [`App.jsx`](frontend/src/App.jsx:42):

```
┌──────────────────────────────────────┐
│  OmniDiag                    [▼]    │
│  Multi-Disease Platform              │
│  ────────────────────────────────    │
│  ● Heart Disease (CAD)              │
│  ○ Diabetes Risk Assessment         │
└──────────────────────────────────────┘
```

- Fetches disease list from `DiseaseProvider`
- On selection, updates context — triggers all downstream components to re-fetch
- Persists selection in `localStorage` for session continuity

#### C. Updated `App.jsx` Changes

| Current | Proposed |
|---------|----------|
| Static views array | Views persist, but components now read `selectedDisease` from context |
| Sidebar only shows mode toggle | Sidebar adds `DiseaseSelector` above mode nav |
| `ActiveComponent` renders unconditionally | `ActiveComponent` receives disease context |

```jsx
// App.jsx — key structural change
<DiseaseProvider>
  <Sidebar>
    <DiseaseSelector />  {/* NEW */}
    <ModeNav />          {/* Engineering | Clinical — unchanged */}
  </Sidebar>
  <Main>
    <ActiveComponent />  {/* reads disease from context internally */}
  </Main>
</DiseaseProvider>
```

#### D. API Layer Update

[`api.js`](frontend/src/api.js) is already disease-parameterized — `api.predict(disease, data)` and `api.explain(disease, data)`. No changes needed to the API client itself. Only the callers need to pass the dynamic disease name from context.

---

## 2. The Dynamic Form Engine (The Core)

### The Problem

| Aspect | Heart Disease | Diabetes |
|--------|---------------|----------|
| Fields | 11 (mix of string selects + numbers) | 21 (14 binary + 7 numeric/ordinal) |
| Grid layout | 2-column works perfectly | 21 fields need grouping |
| Field types | `str` selects, `int` inputs, `float` inputs | `int` binary (0/1), `float` (BMI), `int` ordinal (1-13) |

### Proposed: `DynamicClinicalForm.jsx` (`frontend/src/components/DynamicClinicalForm.jsx`)

#### A. Schema Fetching Pipeline

```
Component Mount
      │
      ▼
Fetch GET /api/v4/{disease}/schema
      │
      ▼
Parse JSON Schema → FieldMetadata[]
      │
      ▼
Render dynamic form fields
      │
      ▼
User fills form → Local state (React Hook Form)
      │
      ▼
Submit → api.predict(disease, data)
```

#### B. JSON Schema → React Component Mapping Logic

The backend's [`GET /api/v4/{disease}/schema`](backend/main.py:100) returns a Pydantic-generated JSON Schema. Here's the mapping strategy:

```typescript
interface FieldMetadata {
  name: string;
  title: string;           // Human-readable label (from description or title)
  type: 'string' | 'integer' | 'number';
  component: 'toggle' | 'select' | 'number-input' | 'slider' | 'text-input';
  validation: {
    required: boolean;
    minimum?: number;
    maximum?: number;
    enum?: string[] | number[];
    step?: number;
  };
  description: string;
  category?: string;       // Inferred from field name prefixes
}
```

**Mapping Rules (in priority order):**

| JSON Schema Pattern | Detected Via | React Component |
|---|---|---|
| `type: integer, minimum: 0, maximum: 1` | `schema.properties[X].minimum === 0 && schema.properties[X].maximum === 1` | **Toggle/Switch** — binary yes/no field |
| `type: string, enum: [...]` | `schema.properties[X].enum` exists | **Dropdown** (native `<select>`) |
| `type: integer, range ≤ 13` (ordinal) | `maximum - minimum ≤ 12` | **Slider** with labeled ticks OR number input |
| `type: number` (float) | `schema.properties[X].type === 'number'` | **Number Input** with step/min/max |
| `type: integer, range > 13` | Otherwise | **Number Input** |
| `type: string, no enum` | Fallthrough | **Text Input** (fallback — rare) |

**Critical Enhancement Required:** The current [`HeartDiseaseInput`](backend/schemas.py:24) uses `str` fields without `Literal` types, so the JSON Schema won't include `enum` arrays for categorical fields like `ChestPainType`, `ST_Slope`, etc. I propose a **non-breaking backend enhancement** (see §6 below) to add `Literal` types, which produces proper `enum` in JSON Schema without changing the API contract.

#### C. Form State Management — React Hook Form

```jsx
import { useForm, Controller } from 'react-hook-form';

function DynamicClinicalForm({ disease, schema, onSubmit }) {
  const { control, handleSubmit, formState: { errors } } = useForm({
    defaultValues: extractDefaults(schema),  // from json_schema_extra.example
    resolver: zodResolver(buildZodSchema(schema))  // client-side validation mirroring server
  });
  // ...
}
```

- `extractDefaults(schema)` — pulls `example` values from the JSON Schema's `$defs` or `properties[X].default`
- `buildZodSchema(schema)` — converts `ge`/`le`/`type`/`required` constraints into a Zod validation schema for instant client-side feedback
- Fields render using the `Controller` pattern for seamless Recharts/Toggle integration

#### D. Server-Side Validation Mirroring

The Zod schema mirrors the server's Pydantic validation:
```typescript
// Example: generated Zod for HeartDiseaseInput
const heartDiseaseSchema = z.object({
  Age: z.number().int().min(20).max(100),
  Sex: z.enum(['M', 'F']),  // after Literal enhancement
  RestingBP: z.number().int().min(80).max(220),
  // ...
});
```

This gives instant validation feedback without a round-trip.

---

## 3. UI/UX Preservation & Adapting (The Paint)

### A. Feature Categorization for Visual Hierarchy

Since the backend returns a flat list of fields, we need **client-side categorization**. Strategy:

**Heuristic Grouping Rules** (in `DynamicClinicalForm.jsx`):

| Category | Matched By | Heart Disease Fields | Diabetes Fields |
|----------|-----------|---------------------|-----------------|
| **Vitals & Signs** | `Age`, `BP`, `HR`, `BMI`, `Cholesterol`, `ChestPain`, `ST_*`, `Oldpeak`, `MaxHR` | 8 fields | BMI |
| **Lifestyle** | `Smoker`, `PhysActivity`, `Fruits`, `Veggies`, `HvyAlcoholConsump`, `Exercise*` | 1 field (ExerciseAngina) | 5 fields |
| **Demographics** | `Sex`, `Age`, `Education`, `Income` | 2 fields | 4 fields |
| **Medical History** | `HighBP`, `HighChol`, `Stroke`, `HeartDiseaseorAttack`, `RestingECG`, `FastingBS`, `Diabetes*` | 3 fields | 6 fields |
| **Healthcare Access** | `AnyHealthcare`, `NoDocbcCost`, `CholCheck` | 0 | 3 fields |
| **Mental Health** | `MentHlth`, `PhysHlth`, `GenHlth`, `DiffWalk` | 0 | 4 fields |

**Implementation:** A simple prefix/suffix keyword matcher in [`DynamicClinicalForm.jsx`](frontend/src/components/DynamicClinicalForm.jsx) maps field names to categories. Each category renders as a collapsible `<details>` card with a colored header:

```
┌─ Vitals & Signs ──────────────────┐
│ [BMI] [  30.2  ] [MaxHR] [  145  ]│
│ [Age ] [   54   ] [BP   ] [  140  ]│
│ ...                                │
└────────────────────────────────────┘
┌─ Lifestyle ────────────────────────┐
│ [Smoker] [● Yes] [○ No]           │
│ [PhysActivity] [● Yes] [○ No]     │
│ ...                                │
└────────────────────────────────────┘
```

**Grid Adaptation:** The form uses CSS grid with `auto-fill` columns:
- For small categories (≤4 fields): `grid-template-columns: repeat(2, 1fr)`
- For medium categories (5-8 fields): `grid-template-columns: repeat(3, 1fr)`
- Toggles render in compact inline rows to conserve space

This naturally handles the 6-box heart disease grid AND the 21-field diabetes form.

### B. Clinical EMR Mode Adaptation

The current [`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx) has hardcoded:

1. **Patient selector** — currently shows 3 heart-disease patients with heart-disease data only
2. **Vitals grid** — hardcoded `VitalCard` components for heart-disease-specific metrics
3. **Diagnosis badge** — hardcoded "Heart Disease Detected" / "No Heart Disease Detected"

**Refactoring Strategy:**

| Component | Current | Proposed |
|-----------|---------|----------|
| Patient selector | Static mockPatients array | Dynamic mock patients per disease (load from `mockPatients.js` structured by disease) |
| Vitals grid | 6 hardcoded `VitalCard` calls | Rendered from schema — first N numeric fields shown as vitals |
| Diagnosis badge | "Heart Disease Detected" | `{schema.display_name} Detected` — dynamic from disease info |
| Clinical Insights | Static table | Still dynamic from SHAP response — no change needed |

The **key insight** is that `ClinicalEmrMode` becomes a *consumer* of the same `DynamicClinicalForm` schema data, but in "display mode" (read-only vitals grid) rather than "edit mode."

### C. ShapBarChart — Dynamic Overflow Handling

Current [`ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx:56) computes height as `Math.max(200, data.length * 36)`. For 26 diabetes features, that's **936px** — which causes text overlap and excessive scrolling.

**Proposed Enhancement:**

```
┌────────────────────────────────────────────┐
│  SHAP Feature Impact                       │
│                                            │
│  [Bar] ST_Slope (Flat)         +0.42       │
│  [Bar] ChestPainType (ASY)     +0.38       │
│  [Bar] Oldpeak                  +0.31       │
│  [Bar] MaxHR                   -0.28       │
│  ... (Top 10 shown)                        │
│                                            │
│  [Show All 26 Features ▼]                  │
└────────────────────────────────────────────┘
```

**Implementation:**

```jsx
function ShapBarChart({ chartData, baseValue, maxVisible = 10 }) {
  const [showAll, setShowAll] = useState(false);
  const visibleData = showAll ? chartData : chartData.slice(0, maxVisible);
  
  // height caps at 400px for top-N, full height for all
  const chartHeight = showAll 
    ? Math.min(chartData.length * 36, 700) 
    : Math.min(maxVisible * 36, 400);
  
  return (
    <div>
      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart data={visibleData}>...</BarChart>
      </ResponsiveContainer>
      
      {chartData.length > maxVisible && (
        <button onClick={() => setShowAll(!showAll)}>
          {showAll ? 'Show Top 10' : `Show All ${chartData.length} Features`}
        </button>
      )}
    </div>
  );
}
```

The `maxVisible` prop defaults to `10` (configurable). This ensures heart disease (11 features) shows all by default, while diabetes (26 features) collapses to top 10.

---

## 4. Safety & Loading States (The Guardrails)

### A. Loading Skeletons

**Schema Fetch Skeleton** (`SkeletonForm.jsx`):

```
┌──────────────────────────────────────┐
│  Loading Patient Parameters...       │
│  ┌──────────────────────────────┐    │
│  │ ▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░░░░░░░  │    │  ← Shimmer bar
│  │ [━━━━━━━━━━] [━━━━━━━━━━]    │    │  ← Input placeholders
│  │ [━━━━━━━━━━] [━━━━━━━━━━]    │    │
│  │ [━━━━━━━━━━━━━━━━━━━━━━━━]   │    │
│  └──────────────────────────────┘    │
└──────────────────────────────────────┘
```

- Uses CSS `@keyframes shimmer` animation
- Renders 4-6 placeholder `<div>` elements mimicking the grid layout
- Auto-sizing based on known feature count (from `GET /api/v4/diseases` response)
- Only shown **while schema is loading** (typically <500ms for cached schemas)

**Prediction Loading Skeleton** (updates current [`ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx:278)):

The existing spinner + "Waking up..." text is preserved. We add a **pulsing metric card** animation:
```
┌──────────────────────────────────────┐
│  AI Diagnostic Panel                 │
│  ┌──────────┐ ┌──────────┐ ┌──────┐ │
│  │ ░░░░░░░  │ │ ░░░░░░░  │ │ ░░░░ │ │  ← Pulsing cards
│  │ ░░░░░░░  │ │ ░░░░░░░  │ │ ░░░░ │ │
│  └──────────┘ └──────────┘ └──────┘ │
└──────────────────────────────────────┘
```

### B. Error Boundaries

**Schema Error Boundary** (`frontend/src/components/ErrorBoundary.jsx`):

```jsx
class SchemaErrorBoundary extends React.Component {
  state = { hasError: false, error: null };
  
  static getDerivedStateFromError(error) {
    return { hasError: true, error };
  }
  
  render() {
    if (this.state.hasError) {
      return <SchemaErrorFallback 
        disease={this.props.disease}
        error={this.state.error}
        onRetry={() => this.setState({ hasError: false })}
      />;
    }
    return this.props.children;
  }
}
```

**Fallback UI:**

```
┌─────────────────────────────────────────┐
│  ⚠️ Schema Load Error                   │
│                                         │
│  Failed to load form schema for         │
│  "Diabetes Risk Assessment".            │
│                                         │
│  This could mean:                       │
│  • The backend is still waking up       │
│  • This disease is not registered       │
│                                         │
│  [Try Again] [Switch Disease]           │
└─────────────────────────────────────────┘
```

**Disease-Specific Error Boundaries:**
- `SchemaErrorBoundary` — wraps `DynamicClinicalForm`
- `PredictionErrorBoundary` — wraps the prediction/SHAP results panel
- `GlobalErrorBoundary` — wraps the entire app (already handled via try/catch)

### C. Graceful Degradation Chain

```
Schema Fetch Fails?
  ├── Show SchemaErrorFallback
  ├── Offer "Retry" button (re-fetches schema)
  └── Offer "Switch Disease" (navigates to another disease)

Prediction Fails?
  ├── Current error banner preserved (already in both modes)
  └── Input form state is preserved — user can edit and retry

SHAP Explanation Fails?
  ├── Show prediction result without SHAP chart
  └── Show informational message: "Feature importance unavailable"
```

---

## 5. File-by-File Refactoring Roadmap

### New Files to Create

| File | Purpose |
|------|---------|
| [`frontend/src/context/DiseaseContext.jsx`](frontend/src/context/DiseaseContext.jsx) | Global disease state provider |
| [`frontend/src/components/DiseaseSelector.jsx`](frontend/src/components/DiseaseSelector.jsx) | Dropdown/tabs for disease switching |
| [`frontend/src/components/DynamicClinicalForm.jsx`](frontend/src/components/DynamicClinicalForm.jsx) | Schema-driven dynamic form engine |
| [`frontend/src/components/SchemaFieldFactory.jsx`](frontend/src/components/SchemaFieldFactory.jsx) | Maps JSON Schema field → React component |
| [`frontend/src/components/FormSkeleton.jsx`](frontend/src/components/FormSkeleton.jsx) | Loading skeleton for schema fetch |
| [`frontend/src/components/ErrorBoundary.jsx`](frontend/src/components/ErrorBoundary.jsx) | Error boundary wrapper |
| [`frontend/src/components/SchemaErrorFallback.jsx`](frontend/src/components/SchemaErrorFallback.jsx) | Error UI for schema failures |
| [`frontend/src/hooks/useDiseaseSchema.js`](frontend/src/hooks/useDiseaseSchema.js) | Custom hook: fetches + caches schema |
| [`frontend/src/hooks/useDiseaseForm.js`](frontend/src/hooks/useDiseaseForm.js) | Custom hook: form state + validation |
| [`frontend/src/utils/schemaToZod.js`](frontend/src/utils/schemaToZod.js) | JSON Schema → Zod schema converter |
| [`frontend/src/utils/schemaFieldParser.js`](frontend/src/utils/schemaFieldParser.js) | Extracts FieldMetadata from JSON Schema |
| [`frontend/src/utils/featureCategorizer.js`](frontend/src/utils/featureCategorizer.js) | Categorizes fields into groups |

### Files to Modify

| File | Changes |
|------|---------|
| [`frontend/src/App.jsx`](frontend/src/App.jsx) | Wrap with `DiseaseProvider`, add `DiseaseSelector` to sidebar |
| [`frontend/src/components/EngineeringMode.jsx`](frontend/src/components/EngineeringMode.jsx) | Replace hardcoded `FIELD_META` + manual form with `DynamicClinicalForm`; remove hardcoded `'heart_disease'` |
| [`frontend/src/components/ClinicalEmrMode.jsx`](frontend/src/components/ClinicalEmrMode.jsx) | Make patient selector disease-aware; dynamic vitals grid from schema; dynamic diagnosis text |
| [`frontend/src/components/ShapBarChart.jsx`](frontend/src/components/ShapBarChart.jsx) | Add `maxVisible` prop + "Show All" toggle |
| [`frontend/src/mockPatients.js`](frontend/src/mockPatients.js) | Restructure as `{ heart_disease: [...], diabetes: [...] }` |
| [`frontend/src/api.js`](frontend/src/api.js) | Add `getSchema(disease)` method |

### Files to Delete

None — we refactor in place, preserving all existing UI/UX.

---

## 6. Backend Schema Enhancement (Non-Breaking)

The current [`HeartDiseaseInput`](backend/schemas.py:24) uses plain `str` for categorical fields, which means `model_json_schema()` won't include `enum` arrays. This is the **only backend change needed**.

**Proposed change** — add `Literal` types to categorical fields:

```python
# Current (no enum in JSON Schema):
Sex: str = Field(..., description="Sex: 'M' or 'F' (or encoded 0/1)")

# Proposed (enum appears in JSON Schema automatically):
Sex: Literal['M', 'F'] = Field(..., description="Sex")
```

This is **non-breaking** because:
- The API still accepts the same string values (`'M'`, `'F'`)
- Pydantic v2 handles `Literal` natively
- The existing validation behavior is identical (values must match one of the literals)
- The `/schema` endpoint now returns richer metadata including `enum` arrays

**Complete list of fields to enhance:**

| File | Field | Current Type | Proposed Literal |
|------|-------|-------------|-----------------|
| [`backend/schemas.py:46`](backend/schemas.py:46) | `Sex` (heart) | `str` | `Literal['M', 'F']` |
| [`backend/schemas.py:47`](backend/schemas.py:47) | `ChestPainType` | `str` | `Literal['TA', 'ATA', 'NAP', 'ASY']` |
| [`backend/schemas.py:50`](backend/schemas.py:50) | `RestingECG` | `str` | `Literal['Normal', 'ST', 'LVH']` |
| [`backend/schemas.py:52`](backend/schemas.py:52) | `ExerciseAngina` | `str` | `Literal['Y', 'N']` |
| [`backend/schemas.py:54`](backend/schemas.py:54) | `ST_Slope` | `str` | `Literal['Up', 'Flat', 'Down']` |
| [`backend/schemas.py:199-212`](backend/schemas.py:199-212) | Diabetes binary fields | `int` (ge=0, le=1) | No change needed — already have valid constraints |

---

## Summary: Data Flow Diagram

```
                    ┌─────────────────┐
                    │  DiseaseProvider │  ← GET /api/v4/diseases
                    │  (React Context) │  ← GET /api/v4/{d}/schema (x N)
                    └────────┬────────┘
                             │ selectedDisease
              ┌──────────────┼──────────────┐
              │              │              │
              ▼              ▼              ▼
     ┌──────────────┐ ┌──────────┐ ┌──────────────┐
     │ Disease      │ │ Engineer │ │ Clinical EMR │
     │ Selector     │ │ ing Mode │ │ Mode         │
     │ (Dropdown)   │ │          │ │              │
     └──────────────┘ │ ┌──────┐ │ │ ┌──────────┐ │
                      │ │Dynamic│ │ │ │Dynamic   │ │
                      │ │Clinica│ │ │ │Clinician │ │
                      │ │lForm  │ │ │ │View      │ │
                      │ │(Edit) │ │ │ │(Display) │ │
                      │ └──────┘ │ │ └──────────┘ │
                      │ ┌──────┐ │ │ ┌──────────┐ │
                      │ │Shap  │ │ │ │Shap      │ │
                      │ │Bar   │ │ │ │Bar       │ │
                      │ │Chart │ │ │ │Chart     │ │
                      │ └──────┘ │ │ └──────────┘ │
                      └──────────┘ └──────────────┘
                              │              │
                              ▼              ▼
                    ┌──────────────────────────┐
                    │  api.predict(disease,    │
                    │    patientData)          │
                    │  api.explain(disease,    │
                    │    patientData)          │
                    └─────────────┬────────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │  POST /api/v4/{disease}/ │
                    │   predict & explain      │
                    └──────────────────────────┘
```

---

**Plan complete.** All 4 pillars are addressed. I have not modified any files. Please review and approve the architecture so I can proceed with implementation.
