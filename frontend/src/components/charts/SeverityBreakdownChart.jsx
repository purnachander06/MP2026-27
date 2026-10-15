import React from 'react';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';

export function SeverityBreakdownChart({ severityData = {} }) {
  const severities = ['critical', 'high', 'medium', 'low', 'informational'];
  const colors = {
    critical: '#EF4444',
    high: '#F97316',
    medium: '#F59E0B',
    low: '#3B82F6',
    informational: '#10B981',
  };

  const chartData = severities
    .filter((sev) => severityData[sev] !== undefined && severityData[sev] > 0)
    .map((sev) => ({
      name: sev.charAt(0).toUpperCase() + sev.slice(1),
      key: sev,
      value: severityData[sev] || 0,
    }));

  if (chartData.length === 0) {
    return (
      <div className="h-56 flex items-center justify-center text-xs text-slate-500">
        No explanation severity distribution records available.
      </div>
    );
  }

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <PieChart>
          <Tooltip
            contentStyle={{
              backgroundColor: '#FFFFFF',
              borderColor: '#E5E7EB',
              borderRadius: '8px',
              fontSize: '12px',
              color: '#202124',
              boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
            }}
          />
          <Legend
            layout="horizontal"
            verticalAlign="bottom"
            align="center"
            wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
          />
          <Pie
            data={chartData}
            cx="50%"
            cy="45%"
            innerRadius={50}
            outerRadius={75}
            paddingAngle={4}
            dataKey="value"
          >
            {chartData.map((entry) => (
              <Cell key={`cell-${entry.key}`} fill={colors[entry.key] || '#64748B'} />
            ))}
          </Pie>
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}

export default SeverityBreakdownChart;
