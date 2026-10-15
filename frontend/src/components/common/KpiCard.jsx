import React from 'react';

export function KpiCard({
  title,
  value,
  description,
  icon: Icon,
  variant = 'default',
  badgeText = null,
}) {
  const borderColors = {
    default: 'border-[#E5E7EB] hover:border-slate-300',
    accent: 'border-[#F26522]/30 hover:border-[#F26522]/60',
    critical: 'border-red-200 hover:border-red-400',
    warning: 'border-amber-200 hover:border-amber-400',
    success: 'border-emerald-200 hover:border-emerald-400',
  };

  const iconColors = {
    default: 'text-[#667085] bg-[#F8F9FA]',
    accent: 'text-[#F26522] bg-[#F26522]/10',
    critical: 'text-red-600 bg-red-50',
    warning: 'text-amber-600 bg-amber-50',
    success: 'text-emerald-600 bg-emerald-50',
  };

  return (
    <div
      className={`bg-white rounded-xl p-5 border ${borderColors[variant] || borderColors.default} transition-all duration-200 flex flex-col justify-between shadow-sm`}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-[#667085]">{title}</p>
          <div className="mt-2 flex items-baseline gap-2">
            <span className="text-2xl md:text-3xl font-bold tracking-tight text-[#202124] font-mono">
              {value}
            </span>
            {badgeText && (
              <span className="text-xs px-2 py-0.5 rounded-full bg-[#F8F9FA] text-[#202124] font-sans border border-[#E5E7EB]">
                {badgeText}
              </span>
            )}
          </div>
        </div>
        {Icon && (
          <div className={`p-2.5 rounded-lg ${iconColors[variant] || iconColors.default}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>

      {description && (
        <p className="mt-3 text-xs text-[#667085] line-clamp-2">{description}</p>
      )}
    </div>
  );
}

export default KpiCard;
