/**
 * DynamicClinicalForm.jsx
 * =======================
 * The core Schema-Driven Dynamic Form Engine.
 *
 * Fetches the JSON Schema for the selected disease via useDiseaseForm hook,
 * renders categorized accordion cards with dynamic field components,
 * and provides submit/reset controls.
 *
 * Features:
 *   - Schema fetching with loading skeleton
 *   - Error boundary + retry
 *   - Categorized collapsible cards (Vitals, Lifestyle, etc.)
 *   - Toggle rows rendered inline in compact grids
 *   - React Hook Form integration with Zod validation
 *   - Submit button that triggers parent callback
 */

import { useState } from 'react';
import {
  Play,
  RotateCcw,
  Loader2,
  ChevronDown,
  Activity,
  Heart,
  User,
  Stethoscope,
  Pill,
  Wind,
  ClipboardList,
  AlertCircle,
  Shuffle,
} from 'lucide-react';
import { useDisease } from '../context/DiseaseContext';
import { useDiseaseForm } from '../hooks/useDiseaseForm';
import { SchemaErrorBoundary } from './ErrorBoundary';
import SchemaFieldFactory from './SchemaFieldFactory';
import FormSkeleton from './FormSkeleton';
import { getCategoryIcon } from '../utils/featureCategorizer';

// ── Icon resolver for categories ──
const CATEGORY_ICONS = {
  Activity, Heart, User, Stethoscope, Pill, Wind, ClipboardList,
};

function resolveIcon(iconName) {
  return CATEGORY_ICONS[iconName] || ClipboardList;
}

/**
 * Collapsible category card — styled accordion for field groups.
 */
function CategoryCard({ category, fields, control, errors, defaultOpen }) {
  const [isOpen, setIsOpen] = useState(defaultOpen);
  const Icon = resolveIcon(getCategoryIcon(category));

  // Separate toggle fields from others for compact layout
  const toggleFields = fields.filter((f) => f.component === 'toggle');
  const otherFields = fields.filter((f) => f.component !== 'toggle');

  return (
    <div className="card overflow-hidden">
      <button
        type="button"
        onClick={() => setIsOpen((prev) => !prev)}
        className="card-header w-full flex items-center justify-between cursor-pointer hover:bg-gray-50 transition-colors"
      >
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
          <Icon className="w-4 h-4 text-primary-500" />
          {category}
          <span className="text-[10px] font-normal text-gray-400 ml-1">
            ({fields.length})
          </span>
        </h3>
        <ChevronDown
          className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${
            isOpen ? 'rotate-180' : ''
          }`}
        />
      </button>

      {isOpen && (
        <div className="card-body border-t border-clinical-border">
          {/* Toggle fields — compact inline grid */}
          {toggleFields.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-6 gap-y-3 mb-4">
              {toggleFields.map((field) => (
                <div key={field.name} className="flex items-center justify-between py-1">
                  <SchemaFieldFactory
                    meta={field}
                    control={control}
                    errors={errors}
                  />
                </div>
              ))}
            </div>
          )}

          {/* Other fields — standard grid */}
          {otherFields.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {otherFields.map((field) => (
                <SchemaFieldFactory
                  key={field.name}
                  meta={field}
                  control={control}
                  errors={errors}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

/**
 * DynamicClinicalForm — the main form component.
 *
 * @param {{
 *   diseaseName: string,
 *   onRunInference: (formData: object) => Promise<void>,
 *   loading?: boolean,
 *   defaultOpenCategories?: string[],
 * }} props
 */
export default function DynamicClinicalForm({
  diseaseName,
  onRunInference,
  loading = false,
  defaultOpenCategories = ['Vitals & Signs', 'Medical History'],
}) {
  const [hasRun, setHasRun] = useState(false);

  // ── Randomize all fields with valid data ──
  const randomizeFields = () => {
    const randomized = {};
    fields.forEach((f) => {
      if (f.component === 'toggle') {
        randomized[f.name] = Math.random() > 0.5 ? 1 : 0;
      } else if (f.component === 'select' && f.validation.enum?.length) {
        const opts = f.validation.enum;
        randomized[f.name] = opts[Math.floor(Math.random() * opts.length)];
      } else if (f.type === 'number' || f.type === 'integer') {
        const min = f.validation.minimum ?? 0;
        const max = f.validation.maximum ?? 100;
        const step = f.validation.step ?? (f.type === 'number' ? 0.1 : 1);
        const val = min + Math.random() * (max - min);
        randomized[f.name] = step < 1 ? parseFloat(val.toFixed(1)) : Math.round(val);
      } else {
        randomized[f.name] = f.default ?? '';
      }
    });
    form.reset(randomized);
  };

  // ── Form hook ──
  const {
    form,
    categorizedFields,
    fields,
    schemaLoading,
    schemaError,
    refetchSchema,
    submitForm,
    resetForm,
    hasErrors,
  } = useDiseaseForm(diseaseName, {
    onSubmit: async (data) => {
      setHasRun(true);
      if (onRunInference) {
        await onRunInference(data);
      }
    },
  });

  const { control, formState: { errors } } = form;

  // ── Loading skeleton ──
  if (schemaLoading) {
    return (
      <FormSkeleton
        fieldCount={fields.length || 12}
        diseaseName={diseaseName}
      />
    );
  }

  // ── Schema error ──
  if (schemaError) {
    return (
      <div className="card border-red-200">
        <div className="card-body text-center py-8">
          <AlertCircle className="w-8 h-8 text-red-400 mx-auto mb-3" />
          <p className="text-sm font-medium text-red-700 mb-1">
            Failed to load form schema
          </p>
          <p className="text-xs text-red-500 mb-4">{schemaError}</p>
          <button onClick={refetchSchema} className="btn-secondary text-xs">
            Retry
          </button>
        </div>
      </div>
    );
  }

  // ── Empty schema ──
  if (!fields || fields.length === 0) {
    return (
      <div className="card">
        <div className="card-body text-center py-8 text-gray-400 text-sm">
          No form fields available for this disease.
        </div>
      </div>
    );
  }

  // ── Render form ──
  return (
    <form onSubmit={submitForm} className="space-y-6">
      {/* Form header with controls */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <p className="text-xs text-gray-500">
            {fields.length} feature{fields.length !== 1 ? 's' : ''} ·{' '}
            {categorizedFields.size} categor{categorizedFields.size !== 1 ? 'ies' : 'y'}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={randomizeFields}
            className="btn-secondary text-xs"
            title="Fill all fields with random valid data"
          >
            <Shuffle className="w-3.5 h-3.5" />
            Randomize
          </button>
          <button
            type="button"
            onClick={resetForm}
            className="btn-secondary text-xs"
            title="Reset to defaults"
          >
            <RotateCcw className="w-3.5 h-3.5" />
            Reset
          </button>
          <button
            type="submit"
            disabled={loading || hasErrors}
            className="btn-primary text-sm"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Running Inference...
              </>
            ) : (
              <>
                <Play className="w-4 h-4" />
                Run Inference
              </>
            )}
          </button>
        </div>
      </div>

      {/* Validation error summary */}
      {hasErrors && (
        <div className="flex items-start gap-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
          <AlertCircle className="w-4 h-4 text-amber-500 mt-0.5 shrink-0" />
          <p className="text-xs text-amber-700">
            Please fix the highlighted fields before running inference.
          </p>
        </div>
      )}

      {/* Categorized accordion cards */}
      <div className="space-y-4">
        {Array.from(categorizedFields.entries()).map(([category, categoryFields]) => (
          <CategoryCard
            key={category}
            category={category}
            fields={categoryFields}
            control={control}
            errors={errors}
            defaultOpen={defaultOpenCategories.includes(category)}
          />
        ))}
      </div>
    </form>
  );
}
