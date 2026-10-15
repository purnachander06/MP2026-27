import React from 'react';
import { Database } from 'lucide-react';

export function EmptyState({
  title = 'No records found',
  description = 'No persisted alerts or analytics match the specified filters.',
  icon: Icon = Database,
  actionButton = null,
}) {
  return (
    <div className="flex flex-col items-center justify-center p-8 text-center rounded-xl border border-dashed border-[#E5E7EB] bg-white my-4">
      <div className="p-3 rounded-full bg-[#F8F9FA] text-[#667085] mb-3 border border-[#E5E7EB]">
        <Icon className="w-6 h-6 text-[#667085]" />
      </div>
      <h3 className="text-sm font-semibold text-[#202124]">{title}</h3>
      <p className="mt-1 text-xs text-[#667085] max-w-sm">{description}</p>
      {actionButton && <div className="mt-4">{actionButton}</div>}
    </div>
  );
}

export default EmptyState;
