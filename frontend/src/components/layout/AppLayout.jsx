import React, { useState } from 'react';
import { Outlet, useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';

const ROUTE_TITLES = {
  '/': 'Security Operations Center — Overview',
  '/monitoring': 'Real-Time Telemetry & Event Stream',
  '/alerts': 'Threat Detections & Historical Alerts',
  '/models': 'Intrusion Detection Models & Offline Benchmark',
  '/explanations': 'Advisory Threat Explanations & LLM Auditing',
  '/analytics': 'Historical Threat Analytics & SOC Metrics',
  '/settings': 'System Architecture & Infrastructure Status',
};

export function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [refreshTrigger, setRefreshTrigger] = useState(0);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const location = useLocation();

  const currentTitle = ROUTE_TITLES[location.pathname] || 'SOC Threat Management Console';

  const triggerRefresh = () => {
    setIsRefreshing(true);
    setRefreshTrigger((prev) => prev + 1);
    setTimeout(() => setIsRefreshing(false), 800);
  };

  return (
    <div className="min-h-screen bg-[#F8F9FA] flex text-[#202124]">
      {/* Sidebar */}
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 lg:pl-64">
        <Topbar
          onMenuClick={() => setSidebarOpen(true)}
          onRefresh={triggerRefresh}
          isRefreshing={isRefreshing}
          title={currentTitle}
        />

        <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
          {/* Provide refreshTrigger and triggerRefresh to all child route pages */}
          <Outlet context={{ refreshTrigger, triggerRefresh }} />
        </main>

        <footer className="py-4 px-6 border-t border-[#E5E7EB] bg-white text-center text-[11px] text-[#667085]">
          <span>
            Research Prototype &bull; LLMs meet Federated Learning for Scalable and Secure IoT Management &bull; IoT-23 Dataset
          </span>
        </footer>
      </div>
    </div>
  );
}

export default AppLayout;
