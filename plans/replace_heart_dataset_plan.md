# Plan: Replace Heart Dataset with heart11.csv

## 1. Current State

### Existing Pipeline

```
Raw Data                          Processed
─────────────────────────────────────────────────────────────────
heart.csv (tab-sep, 12 cols) ──┐
                                ├── merge_data.py ──► merged_heart_data.csv
Z-Alizadeh sani dataset.xlsx ──┘        │
                                         └── clean_data.py ──► final_ready_data.csv
                                                                      │
                                                 └── Label Encoding + StandardScaler
                                                 └── MissForest Imputation
```

### Config ([`configs/heart_disease.yaml`](configs/heart_disease.yaml:24))
```yaml
raw_files:
  - "heart.csv"
  - "Z-Alizadeh sani dataset.xlsx"
merged_file: "merged_heart_data.csv"
final_clean_file: "final_ready_data.csv"
```

## 2. New Dataset Analysis: [`heart11.csv`](heart11.csv)

| # | Column | Description | Example Values | Target Column |
|---|--------|-------------|----------------|---------------|
| 1 | `age` | Age in years | 34–71 | `Age` |
| 2 | `sex` | 1=male, 0=female | 0, 1 | `Sex` → 'M'/'F' |
| 3 | `cp` | Chest pain type (0-indexed) | 0,1,2,3 | `ChestPainType` |
| 4 | `trestbps` | Resting blood pressure | 100–180 | `RestingBP` |
| 5 | `chol` | Serum cholesterol | 131–360 | `Cholesterol` |
| 6 | `fbs` | Fasting blood sugar >120 | 0, 1 | `FastingBS` |
| 7 | `restecg` | Resting ECG (0-indexed) | 0,1,2 | `RestingECG` |
| 8 | `thalach` | Max heart rate | 105–192 | `MaxHR` |
| 9 | `exang` | Exercise angina | 0, 1 | `ExerciseAngina` → 'Y'/'N' |
| 10 | `oldpeak` | ST depression | 0–5.6 | `Oldpeak` |
| 11 | `slope` | ST slope (0-indexed) | 0,1,2 | `ST_Slope` |
| 12 | `ca` | Major vessels colored | 0–4 ⚠️ | **Dropped** (extra) |
| 13 | `thal` | Thalassemia | 0,1,2,3 | **Dropped** (extra) |
| 14 | `target` | Heart disease (0=no, 1=yes) | 0, 1 | `HeartDisease` |

**Shape:** ~1026 rows × 14 columns

## 3. Column Mapping Specification

### Categorical Encodings

| New Column | Values | Maps To | Logic |
|------------|--------|---------|-------|
| `sex` | 0, 1 | `Sex` | 0 → 'F', 1 → 'M' |
| `cp` | 0,1,2,3 | `ChestPainType` | 0→'TA', 1→'ATA', 2→'NAP', 3→'ASY' |
| `restecg` | 0,1,2 | `RestingECG` | 0→'Normal', 1→'ST', 2→'LVH' |
| `exang` | 0, 1 | `ExerciseAngina` | 0→'N', 1→'Y' |
| `slope` | 0,1,2 | `ST_Slope` | 0→'Up', 1→'Flat', 2→'Down' |
| `target` | 0, 1 | `HeartDisease` | Direct copy |
| `ca` | 0–4 | n/a | **Drop** |
| `thal` | 0,1,2,3 | n/a | **Drop** |

### Numerical Columns (direct copy, renamed)
| New Column | Target Column |
|------------|---------------|
| `age` | `Age` |
| `trestbps` | `RestingBP` |
| `chol` | `Cholesterol` |
| `fbs` | `FastingBS` |
| `thalach` | `MaxHR` |
| `oldpeak` | `Oldpeak` |

## 4. Data Quality Issues to Handle

### Issue 1: `ca` column has value `4` (out of expected 0–3 range)
- **Action:** Cap at 3 OR flag as NaN for imputation. Since `ca` is dropped anyway, this is automatically handled.

### Issue 2: Duplicate rows
- Several identical rows found (e.g., lines 17-18 duplicate lines 15-16, lines 56-57 are identical).
- **Action:** Drop duplicate rows after transformation.

### Issue 3: `thal` column has value `0` (out of expected 1–3 range)
- **Action:** Since `thal` is dropped, this is automatically handled.

### Issue 4: File format change
- Current `heart.csv` is tab-separated (`sep='\t'`), new file is comma-separated.
- **Action:** Update separator in merge script.

## 5. Implementation Steps

### Step 1: Create [`experiment_files/data_pipeline/transform_heart11.py`](experiment_files/data_pipeline/transform_heart11.py)
New script that:
1. Reads `heart11.csv` from project root
2. Renames columns per mapping table above
3. Maps categorical values to the 12-column standard format
4. Drops `ca` and `thal` columns
5. Removes duplicate rows
6. Copies the transformed file to `data/heart_disease/raw/heart.csv` (comma-separated)
7. Archives the original `heart.csv` as `heart.csv.bak` for safety

### Step 2: Update [`experiment_files/data_pipeline/merge_data.py`](experiment_files/data_pipeline/merge_data.py)
Change line 24 from:
```python
old_data = pd.read_csv(old_data_path, sep='\t')
```
to:
```python
old_data = pd.read_csv(old_data_path, sep=',')
```

### Step 3: Update config if needed
The config already references `raw_files[0]` as `"heart.csv"` — no change needed since we write the transformed data to the same filename. But we should verify the config is correct.

### Step 4: Run pipeline end-to-end
```bash
python experiment_files/data_pipeline/transform_heart11.py
python experiment_files/data_pipeline/merge_data.py
python experiment_files/data_pipeline/clean_data.py
```

### Step 5: Verification
Check that:
- [`data/heart_disease/raw/heart.csv`](data/heart_disease/raw/heart.csv) has 12 columns in the correct format
- [`data/heart_disease/processed/merged_heart_data.csv`](data/heart_disease/processed/merged_heart_data.csv) contains merged data from both sources
- [`data/heart_disease/processed/final_ready_data.csv`](data/heart_disease/processed/final_ready_data.csv) contains cleaned, encoded, and scaled data
- [`models/heart_disease/preprocessors/`](models/heart_disease/preprocessors/) has `label_encoders.pkl` and `standard_scaler.pkl`

## 6. Pipeline Data Flow (After Changes)

```
                            ┌─────────────────────────┐
                            │   heart11.csv (14 cols)  │
                            │   /workspaces/Heart_...  │
                            └──────────┬──────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────┐
                            │  transform_heart11.py   │
                            │  • Map columns 14→12    │
                            │  • Map cats to strings  │
                            │  • Drop ca, thal        │
                            │  • Remove duplicates    │
                            └──────────┬──────────────┘
                                       │
                                       ▼
                            ┌─────────────────────────┐
                            │  heart.csv (12 cols)    │
                            │  data/heart_disease/raw │
                            └──────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  │
          ┌─────────────────┐  ┌──────────────────┐      │
          │  merge_data.py  │  │ Z-Alizadeh sani  │      │
          │  (comma sep)    │  │ dataset.xlsx     │      │
          └────────┬────────┘  └──────────────────┘      │
                   │                                     │
                   ▼                                     │
          ┌─────────────────┐                            │
          │ merged_heart_   │                            │
          │ data.csv        │                            │
          └────────┬────────┘                            │
                   │                                     │
                   ▼                                     │
          ┌─────────────────┐                            │
          │  clean_data.py  │◄───────────────────────────┘
          │  • Imputation   │
          │  • LabelEncoder │
          │  • StandardScal │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ final_ready_    │
          │ data.csv        │
          │ (ready for ML)  │
          └─────────────────┘
```
