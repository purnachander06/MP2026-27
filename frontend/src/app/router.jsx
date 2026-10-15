import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { AppLayout } from '../components/layout/AppLayout';
import { OverviewPage } from '../pages/OverviewPage';
import { MonitoringPage } from '../pages/MonitoringPage';
import { AlertsPage } from '../pages/AlertsPage';
import { AlertDetailPage } from '../pages/AlertDetailPage';
import { ModelsPage } from '../pages/ModelsPage';
import { ExplanationsPage } from '../pages/ExplanationsPage';
import { AnalyticsPage } from '../pages/AnalyticsPage';
import { SettingsPage } from '../pages/SettingsPage';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: <OverviewPage />,
      },
      {
        path: 'monitoring',
        element: <MonitoringPage />,
      },
      {
        path: 'alerts',
        element: <AlertsPage />,
      },
      {
        path: 'alerts/:predictionId',
        element: <AlertDetailPage />,
      },
      {
        path: 'models',
        element: <ModelsPage />,
      },
      {
        path: 'explanations',
        element: <ExplanationsPage />,
      },
      {
        path: 'analytics',
        element: <AnalyticsPage />,
      },
      {
        path: 'settings',
        element: <SettingsPage />,
      },
      {
        path: '*',
        element: <Navigate to="/" replace />,
      },
    ],
  },
]);

export default router;
