import { useState, useEffect, useCallback } from 'react';
import {
  Activity,
  Heart,
  Thermometer,
  Wind,
  Droplets,
  Zap,
  AlertCircle,
  CheckCircle2,
  Stethoscope,
  Pill,
  FileText,
  User,
  RefreshCw,
  Loader2,
  ChevronDown,
  Languages,
} from 'lucide-react';
import { api } from '../api';
import mockPatients from '../mockPatients';
import ShapBarChart from './ShapBarChart';

/**
 * Clinical EMR Mode — Doctor's View
 *
 * Features:
 * - Mock patient selector (3 pre-defined patients)
 * - Vitals monitor with Tailwind grid cards and Lucide icons
 * - AI Diagnostic Panel with badge (Green/Red)
 * - Clinical Insights rendering bilingual clinical_summary
 */
export default function ClinicalEmrMode() {
  const [selectedPatient, setSelectedPatient] = useState(mockPatients[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [showLang, setShowLang] = useState('en');
  const [patientSelectOpen, setPatientSelectOpen] = useState(false);

  const runDiagnosis = useCallback(async (patient) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setShapData(null);

    try {
      const [pred, expl] = await Promise.all([
        api.predict('heart_disease', patient.data),
        api.explain('heart_disease', patient.data),
      ]);
      setResult(pred);
      setShapData(expl);
    } catch (err) {
      setError(err.message || 'Diagnosis failed. Is the backend running?');
    } finally {
      setLoading(false);
    }
  }, []);

  // Auto-run diagnosis when patient changes
  useEffect(() => {
    if (selectedPatient) {
      runDiagnosis(selectedPatient);
    }
  }, [selectedPatient, runDiagnosis]);

  const selectPatient = (patient) => {
    setSelectedPatient(patient);
    setPatientSelectOpen(false);
  };

  const isPositive = result?.diagnosis === 'Positive';
  const confidencePct = result ? (result.confidence * 100).toFixed(1) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Clinical EMR Mode</h1>
          <p className="text-sm text-gray-500 mt-1">
            Doctor's view &mdash; AI-assisted cardiovascular risk assessment
          </p>
        </div>

        {/* Language toggle */}
        <button
          onClick={() => setShowLang((prev) => (prev === 'en' ? 'ar' : 'en'))}
          className="btn-secondary text-xs"
          title="Toggle language"
        >
          <Languages className="w-3.5 h-3.5" />
          {showLang === 'en' ? 'العربية' : 'English'}
        </button>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        {/* ── Left Column: Patient Info + Vitals ── */}
        <div className="xl:col-span-1 space-y-6">
          {/* Patient Selector */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <User className="w-5 h-5 text-primary-600" />
                Patient
              </h2>
            </div>
            <div className="card-body">
              {/* Dropdown selector */}
              <div className="relative">
                <button
                  onClick={() => setPatientSelectOpen((prev) => !prev)}
                  className="w-full flex items-center justify-between gap-3 p-3 bg-gray-50 rounded-lg border border-clinical-border hover:bg-gray-100 transition-colors"
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-full bg-primary-100 text-primary-700 flex items-center justify-center text-sm font-bold">
                      {selectedPatient.avatar}
                    </div>
                    <div className="text-left">
                      <p className="text-sm font-semibold text-gray-900">{selectedPatient.name}</p>
                      <p className="text-xs text-gray-500">
                        {selectedPatient.id} &middot; {selectedPatient.age}yrs &middot; {selectedPatient.sex}
                      </p>
                    </div>
                  </div>
                  <ChevronDown className="w-4 h-4 text-gray-400" />
                </button>

                {patientSelectOpen && (
                  <>
                    <div className="fixed inset-0 z-10" onClick={() => setPatientSelectOpen(false)} />
                    <div className="absolute top-full left-0 right-0 mt-1 bg-white border border-clinical-border rounded-lg shadow-lg z-20 overflow-hidden">
                      {mockPatients.map((p) => (
                        <button
                          key={p.id}
                          onClick={() => selectPatient(p)}
                          className={`w-full flex items-center gap-3 px-4 py-3 text-left hover:bg-gray-50 transition-colors ${
                            p.id === selectedPatient.id ? 'bg-primary-50' : ''
                          }`}
                        >
                          <div className="w-9 h-9 rounded-full bg-gray-100 text-gray-600 flex items-center justify-center text-xs font-bold">
                            {p.avatar}
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-900">{p.name}</p>
                            <p className="text-xs text-gray-500">{p.id}</p>
                          </div>
                        </button>
                      ))}
                    </div>
                  </>
                )}
              </div>

              {/* Patient details */}
              <div className="mt-4 space-y-3">
                <div className="flex items-start gap-2">
                  <FileText className="w-4 h-4 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs font-medium text-gray-500">History</p>
                    <p className="text-sm text-gray-800">{selectedPatient.history}</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Pill className="w-4 h-4 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs font-medium text-gray-500">Medications</p>
                    <p className="text-sm text-gray-800">{selectedPatient.medications}</p>
                  </div>
                </div>
                <div className="flex items-start gap-2">
                  <Stethoscope className="w-4 h-4 text-gray-400 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-xs font-medium text-gray-500">Admitting Complaint</p>
                    <p className="text-sm text-gray-800">{selectedPatient.admittingComplaint}</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Vitals Monitor */}
          <div className="card">
            <div className="card-header">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary-600" />
                Vitals Monitor
              </h2>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-2 gap-3">
                <VitalCard
                  icon={<Heart className="w-5 h-5 text-red-500" />}
                  label="Heart Rate"
                  value={`${selectedPatient.data.MaxHR} bpm`}
                  status={selectedPatient.data.MaxHR < 100 ? 'normal' : 'warning'}
                />
                <VitalCard
                  icon={<Thermometer className="w-5 h-5 text-orange-500" />}
                  label="Resting BP"
                  value={`${selectedPatient.data.RestingBP} mmHg`}
                  status={selectedPatient.data.RestingBP > 140 ? 'warning' : 'normal'}
                />
                <VitalCard
                  icon={<Droplets className="w-5 h-5 text-blue-500" />}
                  label="Cholesterol"
                  value={`${selectedPatient.data.Cholesterol} mg/dl`}
                  status={selectedPatient.data.Cholesterol > 240 ? 'warning' : 'normal'}
                />
                <VitalCard
                  icon={<Wind className="w-5 h-5 text-teal-500" />}
                  label="Exercise Angina"
                  value={selectedPatient.data.ExerciseAngina === 'Y' ? 'Present' : 'Absent'}
                  status={selectedPatient.data.ExerciseAngina === 'Y' ? 'warning' : 'normal'}
                />
                <VitalCard
                  icon={<Zap className="w-5 h-5 text-purple-500" />}
                  label="ST Slope"
                  value={selectedPatient.data.ST_Slope}
                  status={selectedPatient.data.ST_Slope === 'Down' ? 'warning' : selectedPatient.data.ST_Slope === 'Flat' ? 'elevated' : 'normal'}
                />
                <VitalCard
                  icon={<Activity className="w-5 h-5 text-gray-500" />}
                  label="Chest Pain"
                  value={selectedPatient.data.ChestPainType}
                  status={selectedPatient.data.ChestPainType === 'ASY' ? 'warning' : 'normal'}
                />
              </div>
            </div>
          </div>
        </div>

        {/* ── Right Column: AI Diagnosis + SHAP ── */}
        <div className="xl:col-span-2 space-y-6">
          {/* AI Diagnostic Panel */}
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <Zap className="w-5 h-5 text-primary-600" />
                AI Diagnostic Panel
              </h2>
              {loading && (
                <span className="flex items-center gap-1.5 text-xs text-gray-500">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Analyzing...
                </span>
              )}
            </div>
            <div className="card-body">
              {error && (
                <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-red-800">Diagnosis Error</p>
                    <p className="text-sm text-red-600 mt-0.5">{error}</p>
                  </div>
                </div>
              )}

              {loading && !result && (
                <div className="flex flex-col items-center justify-center py-12 text-gray-400">
                  <Loader2 className="w-8 h-8 animate-spin mb-3" />
                  <p className="text-sm">Running AI diagnosis...</p>
                </div>
              )}

              {result && !error && (
                <div className="space-y-6">
                  {/* Diagnosis badge + confidence */}
                  <div className="flex items-center justify-between p-4 rounded-lg border" style={{
                    backgroundColor: isPositive ? '#fef2f2' : '#f0fdf4',
                    borderColor: isPositive ? '#fecaca' : '#bbf7d0',
                  }}>
                    <div className="flex items-center gap-3">
                      {isPositive ? (
                        <AlertCircle className="w-8 h-8 text-red-500" />
                      ) : (
                        <CheckCircle2 className="w-8 h-8 text-green-500" />
                      )}
                      <div>
                        <p className="text-lg font-bold" style={{ color: isPositive ? '#dc2626' : '#16a34a' }}>
                          {isPositive ? 'Heart Disease Detected' : 'No Heart Disease Detected'}
                        </p>
                        <p className="text-sm text-gray-600">
                          {isPositive
                            ? 'AI analysis indicates elevated risk. Clinical correlation recommended.'
                            : 'AI analysis indicates low risk. Continue routine monitoring.'}
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <span className={isPositive ? 'badge-positive text-base' : 'badge-negative text-base'}>
                        {confidencePct}%
                      </span>
                      <p className="text-xs text-gray-500 mt-1">Confidence</p>
                    </div>
                  </div>

                  {/* Key metrics row */}
                  <div className="grid grid-cols-3 gap-4">
                    <MetricBox label="Prediction" value={result.prediction === 1 ? 'Positive' : 'Negative'} color={isPositive ? 'red' : 'green'} />
                    <MetricBox label="Confidence" value={`${confidencePct}%`} color="blue" />
                    <MetricBox label="SHAP Base Value" value={shapData?.base_value?.toFixed(4) || '\u2014'} color="gray" />
                  </div>
                </div>
              )}

              {!result && !loading && !error && (
                <div className="text-center text-gray-400 py-8 text-sm">
                  Select a patient to begin AI diagnosis.
                </div>
              )}
            </div>
          </div>

          {/* SHAP Explanation + Clinical Insights */}
          {shapData && (
            <div className="card">
              <div className="card-header flex items-center justify-between">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-primary-600" />
                  Clinical Insights
                </h2>
                <button
                  onClick={() => runDiagnosis(selectedPatient)}
                  disabled={loading}
                  className="btn-secondary text-xs"
                  title="Refresh diagnosis"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              </div>
              <div className="card-body">
                {/* SHAP Bar Chart */}
                <ShapBarChart
                  shapValues={shapData.shap_values}
                  featureNames={shapData.feature_names}
                  baseValue={shapData.base_value}
                />

                {/* Clinical Summary */}
                {shapData.clinical_summary && (
                  <div className="mt-6 space-y-3 border-t border-clinical-border pt-4">
                    <div className="flex items-center gap-2">
                      <FileText className="w-4 h-4 text-gray-500" />
                      <span className="text-sm font-semibold text-gray-700">
                        {showLang === 'en' ? 'Clinical Note' : '\u0645\u0644\u0627\u062d\u0638\u0629 \u0633\u0631\u064a\u0631\u064a\u0629'}
                      </span>
                    </div>

                    <div className="p-4 bg-blue-50 border border-blue-100 rounded-lg">
                      <p className="text-sm text-blue-900 leading-relaxed">
                        {showLang === 'en' ? shapData.clinical_summary.en : shapData.clinical_summary.ar}
                      </p>
                    </div>

                    {/* Feature impact table */}
                    <details className="mt-4">
                      <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 flex items-center gap-1">
                        <ChevronDown className="w-3 h-3" />
                        {showLang === 'en' ? 'Detailed Feature Impact' : '\u062a\u0623\u062b\u064a\u0631 \u0627\u0644\u0645\u064a\u0632\u0627\u062a \u0627\u0644\u062a\u0641\u0635\u064a\u0644\u064a'}
                      </summary>
                      <div className="mt-3 overflow-x-auto">
                        <table className="w-full text-xs">
                          <thead>
                            <tr className="border-b border-clinical-border">
                              <th className="text-left py-2 px-2 font-medium text-gray-500">
                                {showLang === 'en' ? 'Feature' : '\u0627\u0644\u0645\u064a\u0632\u0629'}
                              </th>
                              <th className="text-right py-2 px-2 font-medium text-gray-500">
                                {showLang === 'en' ? 'SHAP Value' : '\u0642\u064a\u0645\u0629 SHAP'}
                              </th>
                              <th className="text-right py-2 px-2 font-medium text-gray-500">
                                {showLang === 'en' ? 'Impact' : '\u0627\u0644\u062a\u0623\u062b\u064a\u0631'}
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {shapData.feature_names.map((name, i) => {
                              const val = shapData.shap_values[i];
                              return (
                                <tr key={name} className="border-b border-gray-100">
                                  <td className="py-2 px-2 font-medium text-gray-800">{name}</td>
                                  <td className="py-2 px-2 text-right font-mono text-gray-600">{val.toFixed(4)}</td>
                                  <td className="py-2 px-2 text-right">
                                    {val > 0 ? (
                                      <span className="text-red-600 font-medium">{'\u2191'} Risk</span>
                                    ) : val < 0 ? (
                                      <span className="text-green-600 font-medium">{'\u2193'} Protective</span>
                                    ) : (
                                      <span className="text-gray-400">{'\u2014'}</span>
                                    )}
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
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

/* ── Sub-components ── */

function VitalCard({ icon, label, value, status }) {
  const statusColors = {
    normal: 'border-gray-200 bg-white',
    warning: 'border-red-200 bg-red-50',
    elevated: 'border-yellow-200 bg-yellow-50',
  };

  return (
    <div className={`flex items-center gap-3 p-3 rounded-lg border ${statusColors[status] || statusColors.normal}`}>
      <div className="shrink-0">{icon}</div>
      <div className="min-w-0">
        <p className="text-xs text-gray-500 truncate">{label}</p>
        <p className="text-sm font-semibold text-gray-900 truncate">{value}</p>
      </div>
    </div>
  );
}

function MetricBox({ label, value, color }) {
  const colorMap = {
    red: 'bg-red-50 border-red-200 text-red-700',
    green: 'bg-green-50 border-green-200 text-green-700',
    blue: 'bg-blue-50 border-blue-200 text-blue-700',
    gray: 'bg-gray-50 border-gray-200 text-gray-700',
  };

  return (
    <div className={`p-3 rounded-lg border text-center ${colorMap[color] || colorMap.gray}`}>
      <p className="text-xs font-medium opacity-75">{label}</p>
      <p className="text-sm font-bold mt-0.5">{value}</p>
    </div>
  );
}
