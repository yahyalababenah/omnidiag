import { useState, useEffect, useRef } from 'react';
import {
  Activity,
  Zap,
  Loader2,
  AlertCircle,
  CheckCircle2,
  RefreshCw,
  FileText,
} from 'lucide-react';
import { api } from '../api';
import { useDisease } from '../context/DiseaseContext';
import DynamicClinicalForm from './DynamicClinicalForm';
import ShapBarChart from './ShapBarChart';
import WhatIfScenarioCard from './WhatIfScenarioCard';
import ClinicalReportModal from './ClinicalReportModal';

export default function EngineeringMode() {
  const { selectedDisease, currentDiseaseInfo } = useDisease();
  const [loading, setLoading] = useState(false);
  const [coldStart, setColdStart] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [shapData, setShapData] = useState(null);
  const [counterfactualsData, setCounterfactualsData] = useState(null);
  const [counterfactualsLoading, setCounterfactualsLoading] = useState(false);
  const [lastFormData, setLastFormData] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const coldStartTimer = useRef(null);

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

  const runInference = async (formData) => {
    if (!selectedDisease) return;

    setLoading(true);
    setCounterfactualsLoading(true);
    setColdStart(false);
    setError(null);
    setResult(null);
    setShapData(null);
    setCounterfactualsData(null);
    setLastFormData(formData);

    try {
      const [pred, expl] = await Promise.all([
        api.predict(selectedDisease, formData),
        api.explain(selectedDisease, formData),
      ]);
      setResult(pred);
      setShapData(expl);
    } catch (err) {
      setError(err.message || 'Inference failed. Is the backend running?');
    }

    // Counterfactuals are optional — failure renders mock data with Coming Soon badge
    try {
      const cfResponse = await api.counterfactuals(selectedDisease, formData);
      setCounterfactualsData(cfResponse?.counterfactuals ?? null);
    } catch {
      setCounterfactualsData(null);
    } finally {
      setLoading(false);
      setCounterfactualsLoading(false);
    }
  };

  const handleRefresh = () => {
    if (lastFormData && selectedDisease) {
      runInference(lastFormData);
    }
  };

  const diseaseLabel = currentDiseaseInfo?.display_name || selectedDisease || 'Heart Disease';

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Engineering Mode</h1>
        <p className="text-sm text-gray-500 mt-1">
          Manually test the OmniDiag inference pipeline for <strong>{diseaseLabel}</strong>.
          Adjust patient parameters and run inference to see predictions and SHAP explanations.
        </p>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* ── Dynamic Input Form ── */}
        <div className="card">
          <div className="card-header">
            <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
              <Activity className="w-5 h-5 text-primary-600" />
              Patient Parameters
            </h2>
          </div>
          <div className="card-body">
            {selectedDisease ? (
              <DynamicClinicalForm
                diseaseName={selectedDisease}
                onRunInference={runInference}
                loading={loading}
              />
            ) : (
              <div className="text-center text-gray-400 py-8 text-sm">
                No disease selected. Please select a disease from the sidebar.
              </div>
            )}
          </div>
          {coldStart && (
            <p className="text-xs text-amber-600 text-center pb-4 px-4">
              This might take a few moments if it's the first scan of the day.
            </p>
          )}
        </div>

        {/* ── Results Panel ── */}
        <div className="space-y-6">
          {/* Prediction Result */}
          <div className="card">
            <div className="card-header flex items-center justify-between">
              <h2 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                <CheckCircle2 className="w-5 h-5 text-primary-600" />
                Prediction Result
              </h2>
              {result && (
                <button
                  onClick={handleRefresh}
                  disabled={loading}
                  className="btn-secondary text-xs"
                  title="Re-run inference with same parameters"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
                  Refresh
                </button>
              )}
            </div>
            <div className="card-body">
              {loading && !result && (
                <div className="flex flex-col items-center justify-center py-8 text-gray-400">
                  <Loader2 className="w-8 h-8 animate-spin mb-3" />
                  <p className="text-sm">
                    {coldStart ? 'Waking up the diagnostic engine...' : 'Running AI inference...'}
                  </p>
                </div>
              )}

              {error && (
                <div className="flex items-start gap-3 p-4 bg-red-50 border border-red-200 rounded-lg">
                  <AlertCircle className="w-5 h-5 text-red-500 mt-0.5 shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-red-800">Error</p>
                    <p className="text-sm text-red-600 mt-0.5">{error}</p>
                  </div>
                </div>
              )}

              {!result && !error && !loading && (
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
                    <span className="text-sm font-mono font-semibold">
                      {result.prediction === 1 ? 'Positive' : 'Negative'}
                    </span>
                  </div>

                  {/* Generate AI Report */}
                  {shapData && (
                    <button
                      onClick={() => setShowReportModal(true)}
                      className="btn-secondary w-full text-sm flex items-center justify-center gap-2 mt-2"
                    >
                      <FileText className="w-4 h-4" />
                      Generate AI Report
                    </button>
                  )}

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
                  chartData={shapData.chart_data}
                  baseValue={shapData.base_value}
                />

                {/* DiCE Counterfactuals — What-If Scenarios */}
                <div className="mt-6">
                  <WhatIfScenarioCard
                    counterfactuals={counterfactualsData}
                    loading={counterfactualsLoading}
                    prediction={result?.prediction}
                  />
                </div>

                {/* Textual Explanation */}
                {shapData.text_explanation && (
                  <div className="mt-4 space-y-2">
                    <div className="p-3 bg-blue-50 border border-blue-100 rounded-lg">
                      <p className="text-xs font-medium text-blue-800 mb-1">Feature Impact Summary</p>
                      <p className="text-sm text-blue-900">{shapData.text_explanation}</p>
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

      {/* AI Report Modal */}
      {showReportModal && result && shapData && (
        <ClinicalReportModal
          disease={selectedDisease}
          probability={result.confidence}
          label={result.diagnosis}
          confidenceBand={result.confidence >= 0.7 ? 'HIGH' : result.confidence >= 0.4 ? 'MODERATE' : 'LOW'}
          shapValues={shapData.chart_data}
          features={lastFormData}
          onClose={() => setShowReportModal(false)}
        />
      )}
    </div>
  );
}
