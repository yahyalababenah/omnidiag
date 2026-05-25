import { useState } from 'react';
import {
  Activity,
  Heart,
  Thermometer,
  Wind,
  Droplets,
  Zap,
  Play,
  RotateCcw,
  Loader2,
  AlertCircle,
  CheckCircle2,
} from 'lucide-react';
import { api } from '../api';
import ShapBarChart from './ShapBarChart';

const FIELD_META = {
  Age:            { label: 'Age', type: 'number', min: 20, max: 100, icon: Heart },
  Sex:            { label: 'Sex', type: 'select', options: ['M', 'F'], icon: Heart },
  ChestPainType:  { label: 'Chest Pain Type', type: 'select', options: ['TA', 'ATA', 'NAP', 'ASY'], icon: Activity },
  RestingBP:      { label: 'Resting BP (mm Hg)', type: 'number', min: 80, max: 220, icon: Heart },
  Cholesterol:    { label: 'Cholesterol (mg/dl)', type: 'number', min: 100, max: 600, icon: Droplets },
  FastingBS:      { label: 'Fasting Blood Sugar > 120', type: 'select', options: [0, 1], icon: Thermometer },
  RestingECG:     { label: 'Resting ECG', type: 'select', options: ['Normal', 'ST', 'LVH'], icon: Activity },
  MaxHR:          { label: 'Max Heart Rate', type: 'number', min: 60, max: 220, icon: Heart },
  ExerciseAngina: { label: 'Exercise Angina', type: 'select', options: ['Y', 'N'], icon: Wind },
  Oldpeak:        { label: 'Oldpeak (ST depression)', type: 'number', step: 0.1, icon: Zap },
  ST_Slope:       { label: 'ST Slope', type: 'select', options: ['Up', 'Flat', 'Down'], icon: Activity },
};

const DEFAULT_PATIENT = {
  Age: 54,
  Sex: 'M',
  ChestPainType: 'ATA',
  RestingBP: 140,
  Cholesterol: 289,
  FastingBS: 0,
  RestingECG: 'Normal',
  MaxHR: 122,
  ExerciseAngina: 'N',
  Oldpeak: 0.0,
  ST_Slope: 'Flat',
};

export default function EngineeringMode() {
  const [form, setForm] = useState({ ...DEFAULT_PATIENT });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [shapData, setShapData] = useState(null);

  const updateField = (field, value) => {
    setForm((prev) => ({ ...prev, [field]: value }));
  };

  const resetForm = () => {
    setForm({ ...DEFAULT_PATIENT });
    setResult(null);
    setShapData(null);
    setError(null);
  };

  const runInference = async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    setShapData(null);

    try {
      const [pred, expl] = await Promise.all([
        api.predict('heart_disease', form),
        api.explain('heart_disease', form),
      ]);
      setResult(pred);
      setShapData(expl);
    } catch (err) {
      setError(err.message || 'Inference failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Engineering Mode</h1>
        <p className="text-sm text-gray-500 mt-1">
          Manually test the OmniDiag inference pipeline. Adjust patient parameters and run inference to see predictions and SHAP explanations.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* ── Input Form ── */}
        <div className="card">
          <div className="card-header flex items-center justify-between">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-600" />
              Patient Parameters
            </h2>
            <button onClick={resetForm} className="btn-secondary text-xs" title="Reset to defaults">
              <RotateCcw className="w-3.5 h-3.5" />
              Reset
            </button>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {Object.entries(FIELD_META).map(([key, meta]) => {
                const Icon = meta.icon;
                return (
                  <div key={key}>
                    <label className="block text-xs font-medium text-gray-600 mb-1 flex items-center gap-1.5">
                      <Icon className="w-3.5 h-3.5 text-gray-400" />
                      {meta.label}
                    </label>
                    {meta.type === 'select' ? (
                      <select
                        className="select-field"
                        value={form[key]}
                        onChange={(e) => updateField(key, e.target.value)}
                      >
                        {meta.options.map((opt) => (
                          <option key={opt} value={opt}>
                            {opt}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="number"
                        className="input-field"
                        min={meta.min}
                        max={meta.max}
                        step={meta.step || 1}
                        value={form[key]}
                        onChange={(e) => updateField(key, e.target.value)}
                      />
                    )}
                  </div>
                );
              })}
            </div>

            <div className="mt-6">
              <button
                onClick={runInference}
                disabled={loading}
                className="btn-primary w-full"
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
        </div>

        {/* ── Results Panel ── */}
        <div className="space-y-6">
          {/* Prediction Result */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-primary-600" />
                Prediction Result
              </h2>
            </div>
            <div className="card-body">
              {error && (
                <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-red-800">Error</p>
                    <p className="text-sm text-red-600 mt-0.5">{error}</p>
                  </div>
                </div>
              )}

              {!result && !error && (
                <div className="text-center text-gray-400 py-8 text-sm">
                  Adjust patient parameters and click "Run Inference" to see results.
                </div>
              )}

              {result && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Diagnosis</span>
                    <span className={result.diagnosis === 'Positive' ? 'badge-positive' : 'badge-negative'}>
                      {result.diagnosis === 'Positive' ? (
                        <><AlertCircle className="w-3.5 h-3.5" /> Positive</>
                      ) : (
                        <><CheckCircle2 className="w-3.5 h-3.5" /> Negative</>
                      )}
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Confidence</span>
                    <span className="text-sm font-mono font-semibold">
                      {(result.confidence * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600">Prediction</span>
                    <span className="text-sm font-mono">{result.prediction}</span>
                  </div>

                  {/* Raw JSON */}
                  <details className="mt-4">
                    <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                      Raw JSON Response
                    </summary>
                    <pre className="mt-2 p-3 bg-gray-50 rounded-lg text-xs font-mono text-gray-700 overflow-x-auto max-h-48 overflow-y-auto">
                      {JSON.stringify(result, null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          </div>

          {/* SHAP Explanation */}
          {shapData && (
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Zap className="w-5 h-5 text-primary-600" />
                  SHAP Explanation
                </h2>
              </div>
              <div className="card-body">
                <ShapBarChart
                  shapValues={shapData.shap_values}
                  featureNames={shapData.feature_names}
                  baseValue={shapData.base_value}
                />

                {/* Clinical Summary */}
                {shapData.clinical_summary && (
                  <div className="mt-4 space-y-2">
                    <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg">
                      <p className="text-xs font-medium text-blue-800 mb-1">Clinical Summary (EN)</p>
                      <p className="text-sm text-blue-900">{shapData.clinical_summary.en}</p>
                    </div>
                    <div className="p-3 bg-emerald-50 border border-emerald-100 rounded-lg" dir="rtl">
                      <p className="text-xs font-medium text-emerald-800 mb-1 font-arabic">الملخص السريري (AR)</p>
                      <p className="text-sm text-emerald-900 font-arabic">{shapData.clinical_summary.ar}</p>
                    </div>

                    <details>
                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700">
                        Raw SHAP JSON
                      </summary>
                      <pre className="mt-2 p-3 bg-gray-50 rounded-lg text-xs font-mono text-gray-700 overflow-x-auto max-h-48 overflow-y-auto">
                        {JSON.stringify(shapData, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
