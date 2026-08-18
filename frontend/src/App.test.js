import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import LivestockDetail from './pages/LivestockDetail';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/livestock/:id" element={<LivestockDetail />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;