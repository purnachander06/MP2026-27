import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { DETECTOR_NAMES } from '../../utils/constants';

export function ThreatDistributionChart({ data = [] }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-500">
        No threat breakdown records available.
      </div>
    );
  }

  // Aggregate by detector: { detector_name, malicious: X, benign: Y }
  const detectorMap = {};
  data.forEach((item) => {
    const det = item.detector_name;
    if (!detectorMap[det]) {
      detectorMap[det] = {
        name: DETECTOR_NAMES[det] || det,
        detector: det,
        Malicious: 0,
        Benign: 0,
      };
    }
    if (item.predicted_class === 'malicious') {
      detectorMap[det].Malicious += item.count || 0;
    } else {
      detectorMap[det].Benign += item.count || 0;
    }
  });

  const chartData = Object.values(detectorMap);

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
          <XAxis
            dataKey="name"
            stroke="#667085"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            interval={0}
            angle={-15}
            textAnchor="end"
          />
          <YAxis
            stroke="#667085"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            allowDecimals={false}
          />
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
          <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '15px' }} />
          <Bar dataKey="Malicious" fill="#EF4444" radius={[4, 4, 0, 0]} maxBarSize={30} />
          <Bar dataKey="Benign" fill="#22C55E" radius={[4, 4, 0, 0]} maxBarSize={30} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default ThreatDistributionChart;
