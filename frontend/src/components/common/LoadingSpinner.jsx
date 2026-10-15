import React from 'react';

export function LoadingSpinner({ text = 'Loading SOC telemetry...', size = 'md' }) {
  const sizes = {
    sm: 'w-4 h-4 border-2',
    md: 'w-8 h-8 border-3',
    lg: 'w-12 h-12 border-4',
  };

  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 text-center">
      <div
        className={`${sizes[size] || sizes.md} rounded-full border-slate-700 border-t-[#F26522] animate-spin`}
      />
      {text && <p className="mt-3 text-xs text-slate-400 font-medium tracking-wide">{text}</p>}
    </div>
  );
}

export default LoadingSpinner;
