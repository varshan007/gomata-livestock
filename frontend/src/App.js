import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { LiveTelemetryProvider } from './context/LiveTelemetryContext';
import PrivateRoute from './components/PrivateRoute';
import Landing from './pages/Landing';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import LivestockDetail from './pages/LivestockDetail';
import MovementAnalytics from './pages/MovementAnalytics';
import UserProfile from './pages/UserProfile';
import FarmDetails from './pages/FarmDetails';
import HealthAnalytics from './pages/HealthAnalytics';
import MapIntelligence from './pages/MapIntelligence';
import AlertsCenter from './pages/AlertsCenter';
import LivestockManagement from './pages/LivestockManagement';
import Devices from './pages/Devices';
import Breeds from './pages/Breeds';
import Staff from './pages/Staff';
import Predictions from './pages/Predictions';
import PredictionData from './pages/PredictionData';
import BehaviourAnalysis from './pages/BehaviourAnalysis';
import DiseaseRisk from './pages/DiseaseRisk';
import AIOrchestrator from './pages/AIOrchestrator';
import './App.css';

function App() {
  return (
    <Router>
      <AuthProvider>
        <LiveTelemetryProvider>
          <div className="App">
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Login />} />
              <Route path="/register" element={<Register />} />

              {/* Protected Routes */}
              <Route path="/dashboard" element={
                <PrivateRoute>
                  <Dashboard />
                </PrivateRoute>
              } />
              <Route path="/livestock" element={
                <PrivateRoute>
                  <LivestockManagement />
                </PrivateRoute>
              } />
              <Route path="/livestock/:id" element={
                <PrivateRoute>
                  <LivestockDetail />
                </PrivateRoute>
              } />
              <Route path="/livestock/:id/analytics" element={
                <PrivateRoute>
                  <MovementAnalytics />
                </PrivateRoute>
              } />
              <Route path="/devices" element={
                <PrivateRoute>
                  <Devices />
                </PrivateRoute>
              } />
              <Route path="/breeds" element={
                <PrivateRoute requireAdmin>
                  <Breeds />
                </PrivateRoute>
              } />
              <Route path="/staff" element={
                <PrivateRoute requireAdmin>
                  <Staff />
                </PrivateRoute>
              } />
              <Route path="/predictions" element={
                <PrivateRoute>
                  <Predictions />
                </PrivateRoute>
              } />
              <Route path="/predictions/data" element={
                <PrivateRoute>
                  <PredictionData />
                </PrivateRoute>
              } />
              <Route path="/behaviour" element={
                <PrivateRoute>
                  <BehaviourAnalysis />
                </PrivateRoute>
              } />
              <Route path="/ai-orchestrator" element={
                <PrivateRoute>
                  <AIOrchestrator />
                </PrivateRoute>
              } />
              <Route path="/disease-risk" element={
                <PrivateRoute>
                  <DiseaseRisk />
                </PrivateRoute>
              } />
              <Route path="/profile" element={
                <PrivateRoute>
                  <UserProfile />
                </PrivateRoute>
              } />
              <Route path="/health-analytics" element={
                <PrivateRoute>
                  <HealthAnalytics />
                </PrivateRoute>
              } />
              <Route path="/map" element={
                <PrivateRoute>
                  <MapIntelligence />
                </PrivateRoute>
              } />
              <Route path="/alerts" element={
                <PrivateRoute>
                  <AlertsCenter />
                </PrivateRoute>
              } />
              <Route path="/farm" element={
                <PrivateRoute requireAdmin>
                  <FarmDetails />
                </PrivateRoute>
              } />
            </Routes>
          </div>
        </LiveTelemetryProvider>
      </AuthProvider>
    </Router>
  );
}

export default App;