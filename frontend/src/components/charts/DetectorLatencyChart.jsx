import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { DETECTOR_NAMES } from '../../utils/constants';

export function DetectorLatencyChart({ data = [] }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-56 flex items-center justify-center text-xs text-slate-500">
        No detector latency statistics available.
      </div>
    );
  }

  const chartData = data.map((d) => ({
    name: DETECTOR_NAMES[d.detector_name] || d.detector_name,
    rawName: d.detector_name,
    LatencyMs: d.mean_inference_time_ms || 0,
    Total: d.total_predictions || 0,
  }));

  const colors = ['#F26522', '#38BDF8', '#818CF8', '#A78BFA', '#34D399'];

  return (
    <div className="h-56 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={chartData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 40, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" horizontal={false} />
          <XAxis
            type="number"
            stroke="#667085"
            fontSize={11}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            unit=" ms"
          />
          <YAxis
            type="category"
            dataKey="name"
            stroke="#667085"
            fontSize={10}
            tickLine={false}
            axisLine={{ stroke: '#E5E7EB' }}
            width={120}
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
            formatter={(value) => [`${value} ms`, 'Mean Latency']}
          />
          <Bar dataKey="LatencyMs" radius={[0, 4, 4, 0]} maxBarSize={20}>
            {chartData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default DetectorLatencyChart;
