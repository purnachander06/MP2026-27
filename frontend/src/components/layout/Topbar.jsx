import React, { useEffect, useState } from 'react';
import { Menu, RefreshCw, Server, Database, AlertCircle } from 'lucide-react';
import { healthService } from '../../services/healthService';
import { StatusIndicator } from '../common/StatusIndicator';

export function Topbar({ onMenuClick, onRefresh, isRefreshing = false, title = 'SOC Threat Console' }) {
  const [healthStatus, setHealthStatus] = useState({ online: false, checking: true });
  const [dbStatus, setDbStatus] = useState({ connected: false, checking: true });

  const checkStatus = async () => {
    try {
      const h = await healthService.getHealth();
      setHealthStatus({ online: h.status === 'healthy', checking: false });
    } catch {
      setHealthStatus({ online: false, checking: false });
    }

    try {
      const r = await healthService.getReadiness();
      setDbStatus({
        connected: r.database_connected === true,
        ready: r.status === 'ready',
        checking: false,
      });
    } catch {
      setDbStatus({ connected: false, ready: false, checking: false });
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 20000); // Check every 20s
    return () => clearInterval(interval);
  }, []);

  const handleManualRefresh = () => {
    checkStatus();
    if (onRefresh) onRefresh();
  };

  return (
    <header className="h-16 px-4 sm:px-6 bg-white/95 backdrop-blur-md border-b border-[#E5E7EB] sticky top-0 z-30 flex items-center justify-between">
      <div className="flex items-center gap-3">
        <button
          onClick={onMenuClick}
          className="p-2 rounded-lg text-[#667085] hover:text-[#202124] hover:bg-slate-100 lg:hidden"
          aria-label="Open menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div>
          <h1 className="text-sm sm:text-base font-bold text-[#202124] tracking-tight">{title}</h1>
          <p className="text-[11px] text-[#667085] hidden sm:block">
            LLMs meet Federated Learning for Scalable and Secure IoT Management
          </p>
        </div>
      </div>

      <div className="flex items-center gap-2 sm:gap-4">
        {/* API Backend indicator */}
        <div className="hidden md:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB]">
          <Server className="w-3.5 h-3.5 text-[#667085]" />
          <span className="text-xs text-[#202124]">API:</span>
          {healthStatus.checking ? (
            <span className="text-xs text-[#667085]">Checking...</span>
          ) : (
            <StatusIndicator
              status={healthStatus.online}
              label={healthStatus.online ? 'Online' : 'Offline'}
            />
          )}
        </div>

        {/* PostgreSQL Database indicator */}
        <div className="hidden md:flex items-center gap-2 px-2.5 py-1 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB]">
          <Database className="w-3.5 h-3.5 text-[#667085]" />
          <span className="text-xs text-[#202124]">DB:</span>
          {dbStatus.checking ? (
            <span className="text-xs text-[#667085]">Checking...</span>
          ) : (
            <StatusIndicator
              status={dbStatus.connected}
              label={dbStatus.connected ? 'Connected' : 'Unavailable'}
            />
          )}
        </div>

        {/* Security / Mode pill */}
        <div className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-amber-50 border border-amber-200 text-amber-800 text-[11px] font-medium">
          <AlertCircle className="w-3.5 h-3.5" />
          <span>Research Mode &bull; Trusted Network Only</span>
        </div>

        {/* Refresh button */}
        <button
          onClick={handleManualRefresh}
          disabled={isRefreshing}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[#E5E7EB] bg-white hover:bg-[#F8F9FA] text-[#202124] text-xs font-medium transition-colors shadow-sm disabled:opacity-50"
          title="Refresh live telemetry and analytics"
        >
          <RefreshCw className={`w-3.5 h-3.5 text-[#F26522] ${isRefreshing ? 'animate-spin' : ''}`} />
          <span className="hidden sm:inline">Refresh</span>
        </button>
      </div>
    </header>
  );
}

export default Topbar;
