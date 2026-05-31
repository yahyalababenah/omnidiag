/**
 * SchemaFieldFactory.jsx
 * =======================
 * Renders a single form field based on its FieldMetadata.
 * Maps component types to actual React input elements.
 *
 * Supported components:
 *   segmented → Segmented radio-group (Male=1 / Female=0)
 *   toggle    → Styled checkbox/switch (binary 0/1)
 *   select    → Native <select> dropdown (enum)
 *   slider    → Range slider with live value badge (small-range integer)
 *   number    → Twin-bound slider + <input type="number"> with min/max/step
 *   text      → <input type="text"> (fallback)
 */

import { useController } from 'react-hook-form';

// ── Icons per component type ──
const componentIcons = {
  segmented: '⊙',
  toggle: '⊡',
  select: '▼',
  slider: '═',
  number: '#',
  text: 'Aa',
};

/**
 * Toggle/Switch — binary 0/1 field rendered as a styled checkbox.
 */
function ToggleField({ field, meta, error }) {
  const isOn = field.value === 1 || field.value === true;

  return (
    <div className="flex items-center justify-between">
      <label
        htmlFor={meta.name}
        className="text-xs font-medium text-gray-700 cursor-pointer select-none"
      >
        {meta.title}
      </label>
      <button
        id={meta.name}
        type="button"
        role="switch"
        aria-checked={isOn}
        onClick={() => field.onChange(isOn ? 0 : 1)}
        className={`
          relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-200
          ${isOn ? 'bg-primary-500' : 'bg-gray-200'}
          focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-1
        `}
      >
        <span
          className={`
            inline-block h-4 w-4 transform rounded-full bg-white shadow-sm transition-transform duration-200
            ${isOn ? 'translate-x-6' : 'translate-x-1'}
          `}
        />
      </button>
      {error && <p className="text-xs text-red-500 mt-0.5">{error}</p>}
    </div>
  );
}

/**
 * Segmented Button Group — styled radio-group for distinct binary states (e.g., Sex: Male/Female).
 * Renders two buttons side by side; active state uses primary color.
 */
function SegmentedField({ field, meta, error }) {
  const val = field.value ?? 0;

  return (
    <div>
      <label className="block text-xs font-medium text-gray-600 mb-2">
        {meta.title}
      </label>
      <div className="flex rounded-lg border border-gray-300 overflow-hidden" role="radiogroup">
        <button
          type="button"
          role="radio"
          aria-checked={val === 1}
          onClick={() => field.onChange(1)}
          className={`
            flex-1 px-4 py-2 text-sm font-medium transition-colors duration-150
            ${val === 1
              ? 'bg-primary-500 text-white shadow-sm'
              : 'bg-white text-gray-700 hover:bg-gray-50'
            }
          `}
        >
          Male
        </button>
        <div className="w-px bg-gray-300" />
        <button
          type="button"
          role="radio"
          aria-checked={val === 0}
          onClick={() => field.onChange(0)}
          className={`
            flex-1 px-4 py-2 text-sm font-medium transition-colors duration-150
            ${val === 0
              ? 'bg-primary-500 text-white shadow-sm'
              : 'bg-white text-gray-700 hover:bg-gray-50'
            }
          `}
        >
          Female
        </button>
      </div>
      {error && <p className="text-xs text-red-500 mt-0.5">{error}</p>}
    </div>
  );
}

/**
 * Dropdown — renders a native <select> for enum string/number fields.
 */
function SelectField({ field, meta, error }) {
  const options = meta.validation.enum || [];

  return (
    <div>
      <label htmlFor={meta.name} className="block text-xs font-medium text-gray-600 mb-1">
        {meta.title}
      </label>
      <select
        id={meta.name}
        {...field}
        className={`select-field ${error ? 'border-red-400 ring-1 ring-red-400' : ''}`}
        onChange={(e) => field.onChange(e.target.value)}
      >
        {options.map((opt) => (
          <option key={String(opt)} value={opt}>
            {String(opt)}
          </option>
        ))}
      </select>
      {error && <p className="text-xs text-red-500 mt-0.5">{error}</p>}
    </div>
  );
}

/**
 * Slider — range input for small-range integer fields.
 * Renders the current value as a bold badge next to the label.
 */
function SliderField({ field, meta, error }) {
  const min = meta.validation.minimum ?? 0;
  const max = meta.validation.maximum ?? 10;
  const val = field.value ?? min;

  return (
    <div>
      <label htmlFor={meta.name} className="block text-xs font-medium text-gray-600 mb-1">
        {meta.title}
        {' '}
        <span className="inline-flex items-center justify-center min-w-[1.5rem] h-5 px-1.5
                         text-xs font-bold text-primary-700 bg-primary-50
                         border border-primary-200 rounded-md">
          {val}
        </span>
      </label>
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-400 w-6 text-right">{min}</span>
        <input
          id={meta.name}
          type="range"
          min={min}
          max={max}
          step={1}
          value={val}
          onChange={(e) => field.onChange(parseInt(e.target.value, 10))}
          className="flex-1 h-2 rounded-full appearance-none cursor-pointer
                     bg-gray-200 accent-primary-500
                     [&::-webkit-slider-thumb]:appearance-none
                     [&::-webkit-slider-thumb]:w-4
                     [&::-webkit-slider-thumb]:h-4
                     [&::-webkit-slider-thumb]:rounded-full
                     [&::-webkit-slider-thumb]:bg-primary-500
                     [&::-webkit-slider-thumb]:shadow-sm"
        />
        <span className="text-xs font-mono font-semibold text-gray-700 w-6">{val}</span>
      </div>
      {error && <p className="text-xs text-red-500 mt-0.5">{error}</p>}
    </div>
  );
}

/**
 * Number Input — for float or wide-range integer fields.
 * Shows a twin-bound slider alongside the number input when min/max bounds exist.
 * Both controls share the exact same field.value — moving the slider updates the
 * number box instantly, and typing in the box moves the slider instantly.
 */
function NumberField({ field, meta, error }) {
  const min = meta.validation.minimum;
  const max = meta.validation.maximum;
  const step = meta.validation.step ?? (meta.type === 'number' ? 0.1 : 1);

  // Show a slider when we have finite numeric bounds and the range isn't excessive
  const showSlider = min !== undefined && max !== undefined && (max - min) <= 500;
  const currentVal = field.value ?? min ?? 0;

  return (
    <div>
      <label htmlFor={meta.name} className="block text-xs font-medium text-gray-600 mb-1">
        {meta.title}
      </label>
      <div className="flex items-center gap-3">
        {showSlider && (
          <input
            type="range"
            min={min}
            max={max}
            step={step}
            value={currentVal}
            onChange={(e) => field.onChange(parseFloat(e.target.value))}
            className="flex-1 h-2 rounded-full appearance-none cursor-pointer
                       bg-gray-200 accent-primary-500
                       [&::-webkit-slider-thumb]:appearance-none
                       [&::-webkit-slider-thumb]:w-4
                       [&::-webkit-slider-thumb]:h-4
                       [&::-webkit-slider-thumb]:rounded-full
                       [&::-webkit-slider-thumb]:bg-primary-500
                       [&::-webkit-slider-thumb]:shadow-sm"
          />
        )}
        <input
          id={meta.name}
          type="number"
          min={min}
          max={max}
          step={step}
          value={currentVal}
          onChange={(e) => {
            const raw = e.target.value;
            if (raw === '') {
              field.onChange('');
              return;
            }
            const parsed = Number(raw);
            if (isNaN(parsed)) return;
            // Clamp within bounds to match slider behaviour
            let clamped = parsed;
            if (min !== undefined) clamped = Math.max(min, clamped);
            if (max !== undefined) clamped = Math.min(max, clamped);
            field.onChange(clamped);
          }}
          className={`input-field w-24 ${error ? 'border-red-400 ring-1 ring-red-400' : ''}`}
        />
      </div>
      {error && <p className="text-xs text-red-500 mt-0.5">{error}</p>}
      {(min !== undefined || max !== undefined) && (
        <p className="text-[10px] text-gray-400 mt-0.5">
          {min !== undefined && `Min: ${min}`}
          {min !== undefined && max !== undefined && ' | '}
          {max !== undefined && `Max: ${max}`}
        </p>
      )}
    </div>
  );
}

/**
 * Text Input — fallback for string fields without enum.
 */
function TextField({ field, meta, error }) {
  return (
    <div>
      <label htmlFor={meta.name} className="block text-xs font-medium text-gray-600 mb-1">
        {meta.title}
      </label>
      <input
        id={meta.name}
        type="text"
        {...field}
        className={`input-field ${error ? 'border-red-400 ring-1 ring-red-400' : ''}`}
      />
      {error && <p className="text-xs text-red-500 mt-0.5">{error}</p>}
    </div>
  );
}

// ── Component Registry ──
const COMPONENT_MAP = {
  segmented: SegmentedField,
  toggle: ToggleField,
  select: SelectField,
  slider: SliderField,
  number: NumberField,
  text: TextField,
};

/**
 * SchemaFieldFactory — renders a single dynamic form field.
 *
 * @param {{
 *   meta: import('../utils/schemaFieldParser').FieldMetadata,
 *   control: import('react-hook-form').Control,
 *   errors: object
 * }} props
 */
export default function SchemaFieldFactory({ meta, control, errors }) {
  const {
    field,
    fieldState: { error },
  } = useController({
    name: meta.name,
    control,
    defaultValue: meta.default ?? '',
    rules: {
      required: meta.validation.required ? `${meta.title} is required` : false,
      min: meta.validation.minimum,
      max: meta.validation.maximum,
    },
  });

  const FieldComponent = COMPONENT_MAP[meta.component] || TextField;
  const errMsg = error?.message || errors?.[meta.name];

  return (
    <FieldComponent
      field={field}
      meta={meta}
      error={errMsg}
    />
  );
}
