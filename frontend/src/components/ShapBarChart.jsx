import { useState } from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
  CartesianGrid,
} from 'recharts';
import { lookupMedicalTerm } from '../utils/medicalDictionary';

const COLOR_POSITIVE = '#1a9105';
const COLOR_NEGATIVE = '#a31616';

/**
 * Horizontal bar chart of SHAP values.
 * Positive (risk-increasing) bars are red, negative (protective) are green.
 *
 * Props:
 *   chartData  — Array of { feature: string, shap_value: number } sorted by
 *                |shap_value| descending (as returned by the backend).
 *   baseValue  — Optional base (expected) value from the SHAP explainer.
 *   maxVisible — Maximum number of features to show before "Show All" (default 10).
 */
export default function ShapBarChart({ chartData, baseValue, maxVisible = 10 }) {
  const [showAll, setShowAll] = useState(false);

  if (!chartData || chartData.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8 text-sm">
        No SHAP values to display. Run inference first.
      </div>
    );
  }

  // chartData is already sorted by |shap_value| descending from the backend
  const data = chartData.map((d) => ({
    name: d.feature,
    value: d.shap_value,
    absValue: Math.abs(d.shap_value),
  }));

  const truncated = !showAll && data.length > maxVisible;
  const displayData = truncated ? data.slice(0, maxVisible) : data;

  const maxAbs = Math.max(...data.map((d) => d.absValue), 0.01);
  const domainMax = maxAbs * 1.15;

  const chartHeight = Math.max(200, displayData.length * 36);

  /**
   * Custom Y-axis tick renderer.
   * Uses SVG foreignObject to embed HTML with the MedicalTooltip hover effect.
   */
  const renderCustomYAxisTick = ({ x, y, payload }) => {
    const featureName = payload.value;
    const desc = lookupMedicalTerm(featureName);

    return (
      <g transform={`translate(${x},${y})`}>
        <foreignObject
          x={-140}
          y={-10}
          width={140}
          height={22}
          style={{ overflow: 'visible' }}
        >
          <div
            className="group/tooltip relative inline-flex justify-end"
            style={{ width: '100%' }}
          >
            <span
              className="text-xs font-medium truncate"
              style={{
                color: '#334155',
                maxWidth: '130px',
                display: 'inline-block',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis',
                cursor: desc ? 'help' : 'default',
                borderBottom: desc ? '1px dashed #94a3b8' : 'none',
              }}
            >
              {featureName}
            </span>
            {desc && (
              <span
                role="tooltip"
                className="
                  invisible group-hover/tooltip:visible
                  opacity-0 group-hover/tooltip:opacity-100
                  transition-all duration-200 delay-[400ms]
                  absolute bottom-full left-1/2 -translate-x-1/2 mb-1
                  z-50
                  max-w-[260px] w-max
                  px-3 py-2
                  rounded-lg
                  bg-gray-900 text-white
                  text-xs leading-relaxed
                  shadow-lg
                  pointer-events-none
                  text-left
                  font-normal
                "
              >
                {desc}
                <span
                  className="
                    absolute top-full left-1/2 -translate-x-1/2
                    w-0 h-0
                    border-4 border-transparent border-t-gray-900
                  "
                />
              </span>
            )}
          </div>
        </foreignObject>
      </g>
    );
  };

  /**
   * Custom Recharts Tooltip content with medical descriptions.
   */
  const CustomRechartsTooltip = ({ active, payload, label }) => {
    if (!active || !payload || payload.length === 0) return null;
    const item = payload[0];
    const desc = lookupMedicalTerm(label);
    const val = item.value;
    const isRiskInc = val >= 0;

    return (
      <div
        style={{
          fontSize: 12,
          borderRadius: 8,
          border: '1px solid #e2e8f0',
          boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
          background: 'white',
          padding: '8px 12px',
          maxWidth: 280,
        }}
      >
        <p style={{ fontWeight: 600, margin: '0 0 4px', color: '#1e293b' }}>
          {label}
        </p>
        <p style={{ margin: '0 0 2px', color: isRiskInc ? '#dc2626' : '#16a34a' }}>
          SHAP: {val.toFixed(4)}
          {' '}
          <span>{isRiskInc ? '↑ Risk-increasing' : '↓ Protective'}</span>
        </p>
        {desc && (
          <p style={{ margin: '4px 0 0', color: '#64748b', lineHeight: 1.4, fontSize: 11 }}>
            {desc}
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="space-y-3">
      {baseValue !== undefined && (
        <div className="text-xs text-gray-500 text-center">
          Base value (expected log-odds): <span className="font-mono font-medium">{baseValue.toFixed(4)}</span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={displayData}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 150, bottom: 4 }}
          barSize={20}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
          <XAxis
            type="number"
            domain={[-domainMax, domainMax]}
            tick={{ fontSize: 11, fill: '#64748b' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(tick) => tick.toFixed(2)}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={renderCustomYAxisTick}
            tickLine={false}
            axisLine={false}
            width={150}
          />
          <Tooltip
            content={<CustomRechartsTooltip />}
            cursor={{ fill: '#f8fafc' }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {displayData.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.value >= 0 ? COLOR_POSITIVE : COLOR_NEGATIVE}
                fillOpacity={0.75 + Math.abs(entry.value) / (maxAbs * 2) * 0.25}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Show All / Show Less toggle */}
      {data.length > maxVisible && (
        <button
          type="button"
          onClick={() => setShowAll((prev) => !prev)}
          className="w-full text-xs text-primary-600 hover:text-primary-700 font-medium py-1.5 transition-colors"
        >
          {truncated
            ? `Show All ${data.length} Features`
            : `Show Less (${maxVisible} features)`}
        </button>
      )}

      <div className="flex justify-center gap-6 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLOR_POSITIVE }} />
          Risk-increasing
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-sm" style={{ backgroundColor: COLOR_NEGATIVE }} />
          Protective
        </span>
      </div>
    </div>
  );
}
