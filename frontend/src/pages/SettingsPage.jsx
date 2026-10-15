import React, { useEffect, useState } from 'react';
import {
  Settings,
  Server,
  Database,
  ShieldAlert,
  ExternalLink,
  RefreshCw,
  Cpu,
  Layers,
  Lock,
} from 'lucide-react';
import { healthService } from '../services/healthService';
import { LoadingSpinner } from '../components/common/LoadingSpinner';
import { StatusIndicator } from '../components/common/StatusIndicator';
import { DETECTOR_NAMES } from '../utils/constants';

export function SettingsPage() {
  const [health, setHealth] = useState(null);
  const [readiness, setReadiness] = useState(null);
  const [isLoading, setIsLoading] = useState(true);

  const fetchStatus = async () => {
    setIsLoading(true);
    try {
      const [h, r] = await Promise.all([
        healthService.getHealth(),
        healthService.getReadiness(),
      ]);
      setHealth(h);
      setReadiness(r);
    } catch {
      // Handled via state checks
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
  }, []);

  const apiBaseUrl = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000';

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-base font-bold text-[#202124] flex items-center gap-2">
            <Settings className="w-5 h-5 text-[#F26522]" />
            <span>System Infrastructure & Diagnostic Status</span>
          </h2>
          <p className="text-xs text-[#667085]">
            Runtime architecture, database connection diagnostics, and API endpoint verification
          </p>
        </div>

        <button
          onClick={fetchStatus}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-[#F8F9FA] text-[#202124] text-xs font-medium border border-[#E5E7EB] transition-colors shadow-sm"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Re-check Connectivity</span>
        </button>
      </div>

      {/* Connectivity Diagnostics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* API Backend Card */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-[#E5E7EB]">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-sky-50 text-sky-700 border border-sky-200">
                <Server className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-[#202124]">FastAPI Backend Service</h3>
            </div>
            <StatusIndicator
              status={health?.status === 'healthy'}
              label={health?.status === 'healthy' ? 'Operational' : 'Unavailable'}
            />
          </div>

          <div className="space-y-2 text-xs font-mono text-[#202124]">
            <div className="flex justify-between py-1 border-b border-[#E5E7EB]/60">
              <span className="text-[#667085]">Configured Base URL:</span>
              <span>{apiBaseUrl}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#E5E7EB]/60">
              <span className="text-[#667085]">Service Name:</span>
              <span>{health?.service || 'iot-threat-explanation-api'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#E5E7EB]/60">
              <span className="text-[#667085]">Version:</span>
              <span>v{health?.version || '1.0.0'}</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#667085]">Liveness Timestamp:</span>
              <span>{health?.timestamp || '—'}</span>
            </div>
          </div>
        </div>

        {/* Database Persistence Card */}
        <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-[#E5E7EB]">
            <div className="flex items-center gap-2.5">
              <div className="p-2 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200">
                <Database className="w-4 h-4" />
              </div>
              <h3 className="text-sm font-bold text-[#202124]">PostgreSQL Persistence</h3>
            </div>
            <StatusIndicator
              status={readiness?.database_connected === true}
              label={readiness?.database_connected ? 'Connected' : 'Disconnected'}
            />
          </div>

          <div className="space-y-2 text-xs font-mono text-[#202124]">
            <div className="flex justify-between py-1 border-b border-[#E5E7EB]/60">
              <span className="text-[#667085]">Database Driver:</span>
              <span>SQLAlchemy 2.x + asyncpg</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#E5E7EB]/60">
              <span className="text-[#667085]">Connection Health:</span>
              <span>{readiness?.database_connected ? 'SELECT 1 Ping Succeeded' : 'Ping Failed'}</span>
            </div>
            <div className="flex justify-between py-1 border-b border-[#E5E7EB]/60">
              <span className="text-[#667085]">Tables Configured:</span>
              <span>predictions, explanations</span>
            </div>
            <div className="flex justify-between py-1">
              <span className="text-[#667085]">Migrations Harness:</span>
              <span>Alembic Async Migrations</span>
            </div>
          </div>
        </div>
      </div>

      {/* Model Checkpoint Inspection Strip */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm space-y-4">
        <div className="flex items-center gap-2">
          <Cpu className="w-5 h-5 text-[#F26522]" />
          <h3 className="text-sm font-bold text-[#202124]">Detector Artifact Readiness on Disk</h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
          {Object.entries(readiness?.details || {}).map(([det, isReady]) => (
            <div
              key={det}
              className="p-3 rounded-lg bg-[#F8F9FA] border border-[#E5E7EB] flex items-center justify-between"
            >
              <div>
                <span className="text-xs font-semibold text-[#202124] block truncate" title={det}>
                  {DETECTOR_NAMES[det] || det}
                </span>
                <span className="text-[10px] text-[#667085] font-mono">{det}</span>
              </div>
              <StatusIndicator status={isReady} label={isReady ? 'Ready' : 'Missing'} />
            </div>
          ))}
        </div>
      </div>

      {/* Security Limitations Disclosure */}
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-5 space-y-3">
        <div className="flex items-center gap-2 text-amber-800 font-bold text-xs uppercase tracking-wider">
          <Lock className="w-4 h-4" />
          <span>Security & Environment Constraints Notice</span>
        </div>
        <p className="text-xs text-amber-800 leading-relaxed">
          <strong>Authentication & Authorization Status:</strong> In accordance with project phase constraints, this dashboard runs in research/development mode. No authentication or role-based access control (RBAC) is implemented yet. Do not expose this service to untrusted public networks without an external authentication proxy or reverse proxy (e.g. Nginx with OAuth2/TLS).
        </p>
      </div>

      {/* API Documentation Quick Links */}
      <div className="bg-white border border-[#E5E7EB] rounded-xl p-5 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h4 className="text-sm font-bold text-[#202124]">Interactive API Specifications</h4>
          <p className="text-xs text-[#667085]">Explore interactive Swagger UI and ReDoc schema documentation</p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href={`${apiBaseUrl}/docs`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-[#F8F9FA] text-[#202124] text-xs font-medium border border-[#E5E7EB] transition-colors shadow-sm"
          >
            <span>Swagger UI (/docs)</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
          <a
            href={`${apiBaseUrl}/redoc`}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-white hover:bg-[#F8F9FA] text-[#202124] text-xs font-medium border border-[#E5E7EB] transition-colors shadow-sm"
          >
            <span>ReDoc (/redoc)</span>
            <ExternalLink className="w-3.5 h-3.5" />
          </a>
        </div>
      </div>
    </div>
  );
}

export default SettingsPage;
