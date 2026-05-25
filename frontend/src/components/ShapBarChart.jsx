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

const COLOR_POSITIVE = '#dc2626';
const COLOR_NEGATIVE = '#16a34a';

/**
 * Horizontal bar chart of SHAP values.
 * Positive (risk-increasing) bars are red, negative (protective) are green.
 */
export default function ShapBarChart({ shapValues, featureNames, baseValue }) {
  if (!shapValues || !featureNames || shapValues.length === 0) {
    return (
      <div className="text-center text-gray-400 py-8 text-sm">
        No SHAP values to display. Run inference first.
      </div>
    );
  }

  // Build data sorted by |SHAP| descending
  const data = featureNames
    .map((name, i) => ({
      name,
      value: shapValues[i],
      absValue: Math.abs(shapValues[i]),
    }))
    .sort((a, b) => b.absValue - a.absValue);

  const maxAbs = Math.max(...data.map((d) => d.absValue), 0.01);
  const domainMax = maxAbs * 1.15;

  const formatTooltip = (value) => [
    `SHAP: ${value.toFixed(4)}`,
    value >= 0 ? '↑ Increases risk' : '↓ Decreases risk',
  ];

  return (
    <div className="space-y-3">
      {baseValue !== undefined && (
        <div className="text-xs text-gray-500 text-center">
          Base value (expected log-odds): <span className="font-mono font-medium">{baseValue.toFixed(4)}</span>
        </div>
      )}

      <ResponsiveContainer width="100%" height={Math.max(200, data.length * 36)}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 4, right: 16, left: 8, bottom: 4 }}
          barSize={20}
        >
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
          <XAxis
            type="number"
            domain={[-domainMax, domainMax]}
            tick={{ fontSize: 11, fill: '#64748b' }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            tick={{ fontSize: 11, fill: '#334155', fontWeight: 500 }}
            tickLine={false}
            axisLine={false}
            width={100}
          />
          <Tooltip
            formatter={formatTooltip}
            contentStyle={{
              fontSize: 12,
              borderRadius: 8,
              border: '1px solid #e2e8f0',
              boxShadow: '0 2px 8px rgba(0,0,0,0.08)',
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.value >= 0 ? COLOR_POSITIVE : COLOR_NEGATIVE}
                fillOpacity={0.75 + Math.abs(entry.value) / (maxAbs * 2) * 0.25}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

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
