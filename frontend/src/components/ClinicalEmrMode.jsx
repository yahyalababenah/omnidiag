import { useState, useEffect, useCallback, useRef } from 'react';
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
} from 'lucide-react';
import MedicalTooltip from './MedicalTooltip';
import { api } from '../api';
import { useDisease } from '../context/DiseaseContext';
import { useDiseaseSchema } from '../hooks/useDiseaseSchema';
import { getPatientsForDisease } from '../mockPatients';
import ShapBarChart from './ShapBarChart';
import WhatIfScenarioCard from './WhatIfScenarioCard';

/**
 * Clinical EMR Mode — Doctor's View
 *
 * Features:
 * - Dynamic mock patient selector per disease
 * - Dynamic patient data grid (schema-driven vital cards)
 * - AI Diagnostic Panel with badge (Green/Red)
 * - Clinical Insights rendering bilingual clinical_summary
 * - Disease-aware API calls via DiseaseContext
 */
export default function ClinicalEmrMode() {
  const { selectedDisease, currentDiseaseInfo } = useDisease();
  const titleCase = (str) => str.charAt(0).toUpperCase() + str.slice(1);
  const diseaseLabel = titleCase(currentDiseaseInfo?.display_name || selectedDisease || 'Heart Disease');

  const { fields, loading: schemaLoading, error: schemaError } = useDiseaseSchema(selectedDisease);

  // Build a lookup map: field name → FieldMetadata
  const fieldMap = new Map((fields || []).map((f) => [f.name, f]));

  // Get mock patients for the selected disease
  const patients = getPatientsForDisease(selectedDisease);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [loading, setLoading] = useState(false);
  const [coldStart, setColdStart] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [counterfactualsData, setCounterfactualsData] = useState(null);
  const [counterfactualsLoading, setCounterfactualsLoading] = useState(false);
  const [patientSelectOpen, setPatientSelectOpen] = useState(false);
  const coldStartTimer = useRef(null);

  // Reset patient and clear results when disease changes
  useEffect(() => {
    const newPatients = getPatientsForDisease(selectedDisease);
    setSelectedPatient(newPatients.length > 0 ? newPatients[0] : null);
    setResult(null);
    setShapData(null);
    setCounterfactualsData(null);
    setError(null);
  }, [selectedDisease]);

  // Show "Waking up..." message if request takes > 8s (HF Spaces cold start)
  useEffect(() => {
    if (loading) {
      coldStartTimer.current = setTimeout(() => setColdStart(true), 8_000);
    } else {
      setColdStart(false);
    }
    return () => {
      if (coldStartTimer.current) clearTimeout(coldStartTimer.current);
    };
  }, [loading]);

  const runDiagnosis = useCallback(async (patient) => {
    if (!selectedDisease || !patient) return;

    setLoading(true);
    setCounterfactualsLoading(true);
    setColdStart(false);
    setError(null);
    setResult(null);
    setShapData(null);
    setCounterfactualsData(null);

    try {
      const [pred, expl] = await Promise.all([
        api.predict(selectedDisease, patient.data),
        api.explain(selectedDisease, patient.data),
      ]);
      setResult(pred);
      setShapData(expl);
    } catch (err) {
      setError(err.message || 'Diagnosis failed. Is the backend running?');
    }

    // Counterfactuals are optional — failure renders mock data with Coming Soon badge
    try {
      const cfResponse = await api.counterfactuals(selectedDisease, patient.data);
      setCounterfactualsData(cfResponse?.counterfactuals ?? null);
    } catch {
      setCounterfactualsData(null);
    } finally {
      setLoading(false);
      setCounterfactualsLoading(false);
    }
  }, [selectedDisease]);

  // Auto-run diagnosis when patient changes
  useEffect(() => {
    if (selectedPatient && selectedDisease) {
      runDiagnosis(selectedPatient);
    }
  }, [selectedPatient, selectedDisease, runDiagnosis]);

  const selectPatient = (patient) => {
    setSelectedPatient(patient);
    setPatientSelectOpen(false);
  };

  const isPositive = result?.diagnosis === 'Positive';
  const confidencePct = result ? (result.confidence * 100).toFixed(1) : null;

  // ── Derive a status for a patient data field ──
  const getFieldStatus = (fieldName, value) => {
    const meta = fieldMap.get(fieldName);
    if (!meta) return 'normal';

    // Toggle/binary: 1 = risk factor present
    if (meta.component === 'toggle') {
      return value === 1 ? 'warning' : 'normal';
    }

    // Numeric: compare to min/max range
    if ((meta.type === 'number' || meta.type === 'integer') &&
        meta.validation.minimum !== undefined && meta.validation.maximum !== undefined) {
      const range = meta.validation.maximum - meta.validation.minimum;
      if (range > 0) {
        const ratio = (value - meta.validation.minimum) / range;
        // Upper quartile = elevated risk
        if (ratio > 0.75) return 'warning';
        if (ratio > 0.5) return 'elevated';
      }
    }

    // Select/enum: check if value is a concerning option
    if (meta.component === 'select' && meta.validation.enum) {
      const enumVals = meta.validation.enum;
      // If the value is the last enum entry (usually worst), flag it
      if (enumVals.length > 1 && value === enumVals[enumVals.length - 1]) return 'warning';
      if (enumVals.length > 2 && value === enumVals[enumVals.length - 2]) return 'elevated';
    }

    return 'normal';
  };

  // ── Format a field value for display ──
  const formatFieldValue = (fieldName, value) => {
    const meta = fieldMap.get(fieldName);
    if (meta?.component === 'toggle') {
      return value === 1 ? 'Yes' : 'No';
    }
    if (typeof value === 'number') {
      return value % 1 === 0 ? value.toString() : value.toFixed(1);
    }
    return String(value ?? '—');
  };

  // ── Pick an icon for a field based on its name ──
  const getFieldIcon = (fieldName) => {
    const name = fieldName.toLowerCase();
    if (name.includes('heart') || name.includes('cardiac') || name.includes('maxhr')) return Heart;
    if (name.includes('temp') || name.includes('fever')) return Thermometer;
    if (name.includes('bp') || name.includes('blood')) return Droplets;
    if (name.includes('chol') || name.includes('lipid')) return Droplets;
    if (name.includes('breath') || name.includes('lung') || name.includes('wind') || name.includes('st')) return Wind;
    if (name.includes('pain') || name.includes('angina') || name.includes('chest')) return AlertCircle;
    if (name.includes('bmi') || name.includes('weight')) return Activity;
    if (name.includes('smoke') || name.includes('alcohol')) return Wind;
    if (name.includes('age')) return User;
    return Activity;
  };

  const noDiseaseMsg = !selectedDisease ? (
    <div className="col-span-full text-center text-gray-400 py-12 text-sm">
      No disease selected. Please select a disease from the sidebar.
    </div>
  ) : null;

  const noPatientsMsg = selectedDisease && patients.length === 0 && !schemaLoading ? (
    <div className="col-span-full text-center text-gray-400 py-12 text-sm">
      No mock patients defined for <strong>{diseaseLabel}</strong>.
    </div>
  ) : null;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Clinical EMR Mode</h1>
          <p className="text-sm text-gray-500 mt-1">
            Doctor's view &mdash; AI-assisted <strong>{diseaseLabel}</strong> risk assessment
          </p>
        </div>
      </div>

      {noDiseaseMsg}
      {noPatientsMsg}

      {selectedDisease && patients.length > 0 && selectedPatient && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* ── Left Column: Patient Info + Data Grid ── */}
          <div className="md:col-span-1 xl:col-span-1 space-y-6">
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
                        {patients.map((p) => (
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

            {/* Patient Data Grid — schema-driven vital cards */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-primary-600" />
                  Patient Data Summary
                </h2>
              </div>
              <div className="card-body">
                {schemaLoading ? (
                  <div className="flex items-center justify-center py-8 text-gray-400">
                    <Loader2 className="w-5 h-5 animate-spin mr-2" />
                    <span className="text-sm">Loading schema...</span>
                  </div>
                ) : schemaError ? (
                  <div className="text-center text-gray-400 py-6 text-sm">
                    Unable to load field definitions.
                  </div>
                ) : (
                  <div className="grid grid-cols-2 gap-3">
                    {Object.entries(selectedPatient.data).map(([fieldName, value]) => {
                      const meta = fieldMap.get(fieldName);
                      const status = getFieldStatus(fieldName, value);
                      const Icon = getFieldIcon(fieldName);

                      return (
                        <div
                          key={fieldName}
                          className={`flex items-center gap-3 p-3 rounded-lg border ${
                            status === 'warning'
                              ? 'border-red-200 bg-red-50'
                              : status === 'elevated'
                              ? 'border-yellow-200 bg-yellow-50'
                              : 'border-gray-200 bg-white'
                          }`}
                        >
                          <div className="shrink-0">
                            <Icon className={`w-5 h-5 ${
                              status === 'warning' ? 'text-red-500' :
                              status === 'elevated' ? 'text-yellow-500' :
                              'text-gray-500'
                            }`} />
                          </div>
                          <div className="min-w-0 flex-1">
                            <p className="text-[11px] text-gray-500 truncate" title={meta?.description || fieldName}>
                              <MedicalTooltip term={meta?.title || fieldName}>
                                {meta?.title || fieldName}
                              </MedicalTooltip>
                            </p>
                            <p className="text-sm font-semibold text-gray-900 truncate">
                              {formatFieldValue(fieldName, value)}
                            </p>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* ── Right Column: AI Diagnosis + SHAP ── */}
          <div className="md:col-span-1 xl:col-span-2 space-y-6">
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
                    <p className="text-sm">
                      {coldStart ? 'Waking up the diagnostic engine...' : 'Running AI diagnosis...'}
                    </p>
                    {coldStart && (
                      <p className="text-xs text-amber-600 mt-2">
                        This might take a few moments if it's the first scan of the day.
                      </p>
                    )}
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
                            {isPositive
                              ? `${diseaseLabel} Detected`
                              : `No ${diseaseLabel} Detected`}
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
                  {/* SHAP Bar Chart — maxVisible=10 by default, with "Show All" toggle for large feature sets */}
                  <ShapBarChart
                    chartData={shapData.chart_data}
                    baseValue={shapData.base_value}
                    maxVisible={10}
                  />

                  {/* DiCE Counterfactuals — What-If Scenarios (same layout as Engineering Mode) */}
                  <div className="mt-6">
                    <WhatIfScenarioCard
                      counterfactuals={counterfactualsData}
                      loading={counterfactualsLoading}
                      prediction={result?.prediction}
                    />
                  </div>

                  {/* Textual Explanation */}
                  {shapData.text_explanation && (
                    <div className="mt-6 space-y-3 border-t border-clinical-border pt-4">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-gray-500" />
                        <span className="text-sm font-semibold text-gray-700">
                          Feature Impact Summary
                        </span>
                      </div>

                      <div className="p-4 bg-blue-50 border border-blue-100 rounded-lg">
                        <p className="text-sm text-blue-900 leading-relaxed">
                          {shapData.text_explanation}
                        </p>
                      </div>

                      {/* ═══ Feature Spotlight: Diabetes_Clinical_Risk ═══ */}
                      {(() => {
                        const topFeature = shapData.chart_data?.length > 0
                          ? shapData.chart_data.reduce((a, b) =>
                              Math.abs(a.shap_value) > Math.abs(b.shap_value) ? a : b
                            )
                          : null;
                        const isClinicalRiskTop = topFeature?.feature === 'Diabetes_Clinical_Risk';
                        return (
                          <details className="mt-4" open>
                            <summary className={`text-xs cursor-pointer flex items-center gap-1 ${isClinicalRiskTop ? 'text-indigo-600 hover:text-indigo-800' : 'text-purple-600 hover:text-purple-800'}`}>
                              <ChevronDown className="w-3 h-3" />
                              {isClinicalRiskTop
                                ? 'Top Feature Spotlight: Diabetes_Clinical_Risk'
                                : 'Engineered Feature: Diabetes_Clinical_Risk'}
                            </summary>
                            <div className={`mt-3 p-4 rounded-lg border ${isClinicalRiskTop ? 'bg-indigo-50 border-indigo-200' : 'bg-purple-50 border-purple-200'}`}>
                              <div className="flex items-start gap-3">
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold shrink-0 ${isClinicalRiskTop ? 'bg-indigo-200 text-indigo-700' : 'bg-purple-200 text-purple-700'}`}>
                                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                                  </svg>
                                </div>
                                <div className="min-w-0">
                                  <p className="text-sm font-semibold text-gray-900">
                                    Diabetes_Clinical_Risk — Engineered Risk Index
                                  </p>
                                  <p className="text-xs text-gray-600 mt-1 leading-relaxed">
                                    A logarithmic risk index computed as:
                                    <code className="mx-1 px-1 py-0.5 bg-gray-100 rounded text-[11px] font-mono">
                                      exp(BMI×0.05 + Age×0.03 + GenHlth×0.2 + HighBP×0.5)
                                    </code>
                                  </p>
                                  <p className="text-xs text-gray-500 mt-1">
                                    HighBP carries 50% weight — the single strongest modifiable risk factor.
                                    BMI contributes 5% per unit; effect compounds exponentially. This
                                    engineered feature often dominates the SHAP explanation for diabetes.
                                  </p>
                                  {topFeature && (
                                    <p className="text-xs text-gray-500 mt-1">
                                      Current top feature: <span className="font-semibold">{topFeature.feature}</span>
                                      {' '}(SHAP = {topFeature.shap_value.toFixed(4)})
                                    </p>
                                  )}
                                </div>
                              </div>
                            </div>
                          </details>
                        );
                      })()}

                      {/* Feature impact table */}
                      <details className="mt-4">
                        <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-700 flex items-center gap-1">
                          <ChevronDown className="w-3 h-3" />
                          Detailed Feature Impact
                        </summary>
                        <div className="mt-3 overflow-x-auto">
                          <table className="w-full text-xs">
                            <thead>
                              <tr className="border-b border-clinical-border">
                                <th className="text-left py-2 px-2 font-medium text-gray-500">
                                  Feature
                                </th>
                                <th className="text-right py-2 px-2 font-medium text-gray-500">
                                  SHAP Value
                                </th>
                                <th className="text-right py-2 px-2 font-medium text-gray-500">
                                  Impact
                                </th>
                              </tr>
                            </thead>
                            <tbody>
                              {shapData.chart_data.map((item) => {
                                const val = item.shap_value;
                                const isRisk = item.feature === 'Diabetes_Clinical_Risk';
                                return (
                                  <tr key={item.feature} className={`border-b border-gray-100 ${isRisk ? 'bg-indigo-50/50' : ''}`}>
                                    <td className="py-2 px-2 font-medium text-gray-800">
                                      <MedicalTooltip term={item.feature}>
                                        {item.feature}
                                      </MedicalTooltip>
                                      {isRisk && (
                                        <span className="ml-1.5 inline-flex items-center px-1.5 py-0.5 rounded text-[10px] font-bold bg-indigo-100 text-indigo-700">
                                          Engineered
                                        </span>
                                      )}
                                    </td>
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
      )}
    </div>
  );
}

/* ── Sub-components ── */

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
