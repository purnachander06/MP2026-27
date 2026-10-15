import React from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { formatDate } from '../../utils/formatters';

export function AlertTrendChart({ data = [], bucketSize = 'hour', onBucketChange = null }) {
  if (!data || data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-xs text-slate-500">
        No time-series prediction records available for this observation window.
      </div>
    );
  }

  // Format data for chart display
  const chartData = data.map((d) => ({
    time: formatDate(d.bucket_start),
    rawTime: d.bucket_start,
    Total: d.total_predictions || 0,
    Malicious: d.malicious_count || 0,
    Benign: d.benign_count || 0,
  }));

  return (
    <div className="w-full">
      {onBucketChange && (
        <div className="flex items-center justify-end gap-2 mb-3">
          <span className="text-[11px] text-[#667085]">Time Bucket:</span>
          {['hour', 'day', 'week'].map((b) => (
            <button
              key={b}
              onClick={() => onBucketChange(b)}
              className={`px-2 py-0.5 rounded text-xs font-medium uppercase tracking-wider transition-colors ${
                bucketSize === b
                  ? 'bg-[#F26522] text-white'
                  : 'bg-white border border-[#E5E7EB] text-[#667085] hover:text-[#202124]'
              }`}
            >
              {b}
            </button>
          ))}
        </div>
      )}

      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#38BDF8" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#38BDF8" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorMalicious" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorBenign" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#22C55E" stopOpacity={0.2} />
                <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
              </linearGradient>
            </defs>

            <CartesianGrid strokeDasharray="3 3" stroke="#E5E7EB" vertical={false} />
            <XAxis
              dataKey="time"
              stroke="#667085"
              fontSize={11}
              tickLine={false}
              axisLine={{ stroke: '#E5E7EB' }}
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
              labelStyle={{ color: '#667085', marginBottom: '4px' }}
            />
            <Legend
              wrapperStyle={{ fontSize: '11px', paddingTop: '10px' }}
              iconType="circle"
            />
            <Area
              type="monotone"
              dataKey="Total"
              stroke="#38BDF8"
              fillOpacity={1}
              fill="url(#colorTotal)"
              strokeWidth={1.5}
            />
            <Area
              type="monotone"
              dataKey="Malicious"
              stroke="#EF4444"
              fillOpacity={1}
              fill="url(#colorMalicious)"
              strokeWidth={2}
            />
            <Area
              type="monotone"
              dataKey="Benign"
              stroke="#22C55E"
              fillOpacity={1}
              fill="url(#colorBenign)"
              strokeWidth={1.5}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default AlertTrendChart;
