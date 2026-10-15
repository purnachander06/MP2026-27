import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Activity,
  ShieldAlert,
  Cpu,
  Brain,
  BarChart3,
  Settings,
  X,
  ShieldCheck,
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/monitoring', label: 'Live Monitoring', icon: Activity },
  { to: '/alerts', label: 'Threat Alerts', icon: ShieldAlert },
  { to: '/models', label: 'Detection Models', icon: Cpu },
  { to: '/explanations', label: 'AI Explanations', icon: Brain },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
  { to: '/settings', label: 'System Status', icon: Settings },
];

export function Sidebar({ isOpen, onClose }) {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm lg:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      {/* Sidebar container */}
      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 w-64 bg-white border-r border-[#E5E7EB] flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Brand header */}
        <div className="h-16 px-5 border-b border-[#E5E7EB] flex items-center justify-between">
          <NavLink to="/" className="flex items-center gap-3 group" onClick={onClose}>
            <div className="p-2 rounded-lg bg-[#F26522]/10 border border-[#F26522]/20 text-[#F26522] group-hover:scale-105 transition-transform">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <span className="font-bold text-sm text-[#202124] tracking-wide block">
                IoT THREAT SOC
              </span>
              <span className="text-[10px] text-[#667085] font-mono tracking-wider block">
                FEDERATED IDS &bull; FL+LLM
              </span>
            </div>
          </NavLink>

          <button
            onClick={onClose}
            className="p-1 rounded-lg text-[#667085] hover:text-[#202124] hover:bg-slate-100 lg:hidden"
            aria-label="Close menu"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation links */}
        <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/'}
                onClick={onClose}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-xs font-medium transition-all ${
                    isActive
                      ? 'bg-[#F26522]/10 text-[#F26522] border border-[#F26522]/25 font-semibold'
                      : 'text-[#667085] hover:text-[#202124] hover:bg-[#F8F9FA] border border-transparent'
                  }`
                }
              >
                <Icon className="w-4 h-4 shrink-0" />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Project footnote */}
        <div className="p-4 border-t border-[#E5E7EB] bg-[#F8F9FA]">
          <div className="text-[11px] text-[#667085]">
            <span className="font-semibold text-[#202124] block">IoT-23 Defense Fleet</span>
            <span className="text-[#667085] block mt-0.5 font-mono text-[10px]">
              RF &bull; LR &bull; FedAvg &bull; FedOpt &bull; GSFS
            </span>
          </div>
        </div>
      </aside>
    </>
  );
}

export default Sidebar;
