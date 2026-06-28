# OmniDiag — Entity Relationship Diagram

## Overview

The OmniDiag database consists of **7 tables** supporting multi-disease clinical decision support with role-based access control, patient records, prediction history, expert review workflow, and immutable audit logging.

---

## Entity Relationship Diagram

```
┌──────────────────────┐       ┌──────────────────────┐       ┌──────────────────────────┐
│        roles         │       │      user_roles      │       │         users            │
├──────────────────────┤       ├──────────────────────┤       ├──────────────────────────┤
│ PK │ id: Integer     │◄──────│ FK │ role_id: Integer│       │ PK │ id: UUID             │
│    │ name: String(50)│       │ FK │ user_id: UUID   │──────►│    │ email: String(255)    │
│    │ description     │       │    │ assigned_at     │       │    │ hashed_password: Str  │
└──────────────────────┘       └──────────────────────┘       │    │ full_name: String(255)│
                                                                │    │ is_active: Boolean    │
                                                                │    │ created_at: DateTime  │
                                                                │    │ updated_at: DateTime  │
                                                                └───────────┬──────────────┘
                                                                            │
                                    ┌───────────────────────────────────────┼───────────────────────────┐
                                    │                                       │                           │
                                    ▼                                       ▼                           ▼
        ┌──────────────────────┐                      ┌──────────────────────────┐      ┌──────────────────────────┐
        │      patients        │                      │      predictions         │      │      audit_logs           │
        ├──────────────────────┤                      ├──────────────────────────┤      ├──────────────────────────┤
        │ PK │ id: UUID        │                      │ PK │ id: UUID            │      │ PK │ id: BigInteger       │
        │    │ mrn: String(50) │◄─────────────────────│ FK │ patient_id: UUID    │      │ FK │ user_id: UUID        │
        │    │ full_name: Str  │                      │    │ disease: Str(100)   │      │    │ endpoint: Str(255)   │
        │    │ date_of_birth   │                      │    │ input_features: JSON│      │    │ method: String(10)   │
        │    │ gender: Str(10) │                      │    │ prediction: Integer │      │    │ status_code: Integer │
        │    │ contact_email   │                      │    │ confidence: Float   │      │    │ ip_address: Str(45)  │
        │ FK │ created_by: UUID│                      │    │ diagnosis: Str(100) │      │    │ duration_ms: Float   │
        │    │ created_at      │                      │    │ shap_chart: JSON    │      │    │ created_at: DateTime │
        │    │ deleted_at      │                      │ FK │ created_by: UUID    │      └──────────────────────────┘
        └──────────────────────┘                      │    │ created_at: DateTime│
                                                      └───────────┬──────────────┘
                                                                  │
                                                                  │  (1:1)
                                                                  ▼
                                                      ┌──────────────────────────┐
                                                      │      review_queue        │
                                                      ├──────────────────────────┤
                                                      │ PK │ id: UUID            │
                                                      │ FK │ prediction_id: UUID │
                                                      │    │ uncertainty_score   │
                                                      │ FK │ reviewer_id: UUID   │
                                                      │    │ label: Integer      │
                                                      │    │ reviewed_at         │
                                                      │    │ status: String(20)  │
                                                      │    │ created_at: DateTime│
                                                      └──────────────────────────┘
```

---

## Table Definitions

### 1. `roles`

Defines role-based access control levels.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `Integer` | `PK`, `autoincrement` | Auto-incrementing ID |
| `name` | `String(50)` | `UNIQUE`, `NOT NULL` | Role name: `super_admin`, `doctor`, `nurse`, `viewer` |
| `description` | `String(255)` | `nullable` | Human-readable description |

**Seed data:**
```sql
INSERT INTO roles (name, description) VALUES
  ('super_admin', 'Full system access — user management, audit review, model administration'),
  ('doctor',     'Clinical access — predict, explain, counterfactuals, patient records'),
  ('nurse',      'Limited clinical access — predict, view patient records'),
  ('viewer',     'Read-only access — view predictions and patient data');
```

---

### 2. `users`

System users (doctors, nurses, admins).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | `PK`, `default=uuid4` | UUID primary key |
| `email` | `String(255)` | `UNIQUE`, `NOT NULL`, `INDEX` | Login email |
| `hashed_password` | `String(255)` | `NOT NULL` | bcrypt-hashed password |
| `full_name` | `String(255)` | `NOT NULL` | Display name |
| `is_active` | `Boolean` | `default=True` | Soft disable account |
| `created_at` | `DateTime(tz)` | `server_default=now()` | Account creation timestamp |
| `updated_at` | `DateTime(tz)` | `onupdate=now()` | Last profile update |

**Relationships:**
- `roles` (many-to-many via `user_roles` join table)
- `predictions` (one-to-many, as `created_by`)
- `patients` (one-to-many, as `created_by`)
- `audit_logs` (one-to-many)
- `review_queue` (one-to-many, as `reviewer_id`)

---

### 3. `user_roles`

Association (join) table linking users to roles (many-to-many).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `user_id` | `UUID` | `FK → users.id`, `PK (composite)` | Reference to user |
| `role_id` | `Integer` | `FK → roles.id`, `PK (composite)` | Reference to role |
| `assigned_at` | `DateTime` | `server_default=now()` | When the role was assigned |

**Foreign Keys:**
- `user_id` → `users.id` (`ON DELETE CASCADE`)
- `role_id` → `roles.id` (`ON DELETE CASCADE`)

---

### 4. `patients`

Patient demographic records with soft-delete support.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | `PK`, `default=uuid4` | UUID primary key |
| `mrn` | `String(50)` | `UNIQUE`, `NOT NULL`, `INDEX` | Medical Record Number |
| `full_name` | `String(255)` | `NOT NULL` | Patient's full name |
| `date_of_birth` | `Date` | `nullable` | Date of birth |
| `gender` | `String(10)` | `nullable` | Gender identity |
| `contact_email` | `String(255)` | `nullable` | Contact email |
| `created_by` | `UUID` | `FK → users.id`, `nullable` | Who registered this patient |
| `created_at` | `DateTime(tz)` | `server_default=now()` | Registration timestamp |
| `deleted_at` | `DateTime` | `nullable` | Soft-delete timestamp (GDPR) |

**Foreign Keys:**
- `created_by` → `users.id` (`SET NULL`)

**Relationships:**
- `predictions` (one-to-many)

---

### 5. `predictions`

Every prediction made by the system, linked to a patient (or anonymous).

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | `PK`, `default=uuid4` | UUID primary key |
| `patient_id` | `UUID` | `FK → patients.id`, `nullable` | Patient (nullable for anonymous) |
| `disease` | `String(100)` | `NOT NULL`, `INDEX` | Disease key (e.g., `heart_disease`) |
| `input_features` | `JSON` | `NOT NULL` | Raw patient data dict sent to model |
| `prediction` | `Integer` | `NOT NULL` | Model output: `0` (negative) or `1` (positive) |
| `confidence` | `Float` | `NOT NULL` | Probability score `0.0`–`1.0` |
| `diagnosis` | `String(100)` | `nullable` | Human label: `"Positive"` / `"Negative"` |
| `shap_chart_data` | `JSON` | `nullable` | Stored SHAP values (if explain was called) |
| `created_by` | `UUID` | `FK → users.id`, `nullable` | Who initiated the prediction |
| `created_at` | `DateTime(tz)` | `server_default=now()` | Prediction timestamp |

**Foreign Keys:**
- `patient_id` → `patients.id` (`SET NULL`)
- `created_by` → `users.id` (`SET NULL`)

**Relationships:**
- `patient` (many-to-one)
- `created_by` user (many-to-one)
- `review_queue` (one-to-one)

---

### 6. `review_queue`

Uncertain predictions (confidence 40–60%) flagged for expert review.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `UUID` | `PK`, `default=uuid4` | UUID primary key |
| `prediction_id` | `UUID` | `FK → predictions.id`, `UNIQUE` | The uncertain prediction |
| `uncertainty_score` | `Float` | `nullable` | Entropy value |
| `reviewer_id` | `UUID` | `FK → users.id`, `nullable` | Assigned reviewer |
| `label` | `Integer` | `nullable` | Expert annotation: `0` or `1` |
| `reviewed_at` | `DateTime` | `nullable` | When the review occurred |
| `status` | `String(20)` | `default="pending"` | `"pending"` \| `"reviewed"` \| `"skipped"` |
| `created_at` | `DateTime(tz)` | `server_default=now()` | When added to queue |

**Foreign Keys:**
- `prediction_id` → `predictions.id` (`CASCADE`)
- `reviewer_id` → `users.id` (`SET NULL`)

---

### 7. `audit_logs`

Immutable log of every authenticated API request for compliance.

| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| `id` | `BigInteger` | `PK`, `autoincrement` | Auto-incrementing ID |
| `user_id` | `UUID` | `FK → users.id`, `nullable` | Who made the request |
| `endpoint` | `String(255)` | `NOT NULL` | API path (e.g., `/api/v4/heart_disease/predict`) |
| `method` | `String(10)` | `NOT NULL` | HTTP method: `GET`, `POST`, etc. |
| `status_code` | `Integer` | `nullable` | HTTP response status code |
| `ip_address` | `String(45)` | `nullable` | Client IP (supports IPv6) |
| `duration_ms` | `Float` | `nullable` | Request duration in milliseconds |
| `created_at` | `DateTime(tz)` | `server_default=now()`, `INDEX` | Request timestamp |

**Foreign Keys:**
- `user_id` → `users.id` (`SET NULL`)

---

## Referential Integrity Summary

| FK Constraint | From | To | On Delete |
|---------------|------|----|-----------|
| `user_roles.user_id` → `users.id` | `user_roles` | `users` | `CASCADE` |
| `user_roles.role_id` → `roles.id` | `user_roles` | `roles` | `CASCADE` |
| `patients.created_by` → `users.id` | `patients` | `users` | `SET NULL` |
| `predictions.patient_id` → `patients.id` | `predictions` | `patients` | `SET NULL` |
| `predictions.created_by` → `users.id` | `predictions` | `users` | `SET NULL` |
| `review_queue.prediction_id` → `predictions.id` | `review_queue` | `predictions` | `CASCADE` |
| `review_queue.reviewer_id` → `users.id` | `review_queue` | `users` | `SET NULL` |
| `audit_logs.user_id` → `users.id` | `audit_logs` | `users` | `SET NULL` |

---

## Indexes

| Table | Column(s) | Type | Purpose |
|-------|-----------|------|---------|
| `users` | `email` | Unique + B-tree | Fast login lookup |
| `patients` | `mrn` | Unique + B-tree | Fast MRN search |
| `predictions` | `disease` | B-tree | Filter predictions by disease |
| `predictions` | `patient_id` | B-tree | Lookup predictions by patient |
| `audit_logs` | `created_at` | B-tree | Time-range audit queries |
| `review_queue` | `prediction_id` | Unique + B-tree | One-to-one enforcement |
| `review_queue` | `status` | B-tree | Filter by review status |
