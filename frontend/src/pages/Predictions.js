import React, { useState, useEffect, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Map as MapIcon, Activity, AlertTriangle, Cpu, Layers, Globe, Users, Settings, LogOut, Menu, Search, Bell, ShieldAlert, Sparkles, Thermometer, Battery, Zap, CheckCircle2, Navigation, MapPin, TrendingUp, TrendingDown, Minus, Info, X, ChevronRight
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { MapContainer, TileLayer, Circle, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css'; // Standard Premium Sidebar
import './Predictions.css'; // Refined High-Density Context Theme

// Fix Leaflet Default Icon Issue
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Mock Data
const healthForecastData = [
    { day: 'Today', temp: 38.5 },
    { day: 'Day 2', temp: 38.6 },
    { day: 'Day 3', temp: 38.8 },
    { day: 'Day 4', temp: 39.2 },
    { day: 'Day 5', temp: 39.6 }, // High fever peak
    { day: 'Day 6', temp: 39.3 },
    { day: 'Day 7', temp: 39.0 }
];

const highRiskAnimals = [
    { id: 'MIX005', current: '38.5°C', forecast: '39.6°C', risk: 'High', time: '48 hrs', confidence: 91 },
    { id: 'MIX012', current: '38.3°C', forecast: '39.2°C', risk: 'Medium', time: '72 hrs', confidence: 82 },
    { id: 'BRD044', current: '38.0°C', forecast: '38.9°C', risk: 'Low', time: '5 days', confidence: 65 },
];

const deviceFailures = [
    { id: 'DEV023', animal: 'MIX005', battery: '28%', risk: 'High', time: '5 days', confidence: 98 },
    { id: 'DEV011', animal: 'MIX008', battery: '34%', risk: 'Medium', time: '8 days', confidence: 85 },
];

const initialOpportunities = [
    { id: 1, type: 'treat', priority: 'urgent', badge: 'Today', title: 'Pre-emptive Treatment', desc: 'Administer antipyretics to MIX005 based on 91% fever confidence.', icon: 'opt-treat' },
    { id: 2, type: 'charge', priority: 'today', badge: 'This Week', title: 'Charge 4 Devices Now', desc: 'Prevent telemetry blackout by swapping batteries before Friday.', icon: 'opt-charge' },
    { id: 3, type: 'move', priority: 'plan', badge: 'Requires Planning', title: 'Relocate 12 Heads', desc: 'Move 12 animals to North pasture to reduce projected heat stress.', icon: 'opt-move' },
];

function Predictions() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [horizon, setHorizon] = useState('7D');
    const [opportunities, setOpportunities] = useState(initialOpportunities);

    // AI Inspector State
    const [inspectedAnimal, setInspectedAnimal] = useState(null);

    const handleDismiss = (id) => {
        setOpportunities(opportunities.filter(o => o.id !== id));
    };

    const handleInspect = (animal) => {
        setInspectedAnimal(animal);
    };

    // Custom Tooltip for light mode Recharts
    const CustomTooltip = ({ active, payload, label }) => {
        if (active && payload && payload.length) {
            return (
                <div style={{ background: '#ffffff', border: '1px solid #e2e8f0', padding: '12px', borderRadius: '12px', boxShadow: '0 4px 15px rgba(0,0,0,0.05)' }}>
                    <p style={{ margin: '0 0 4px 0', fontSize: '0.85rem', color: '#6b7d73', fontWeight: 600 }}>{label}</p>
                    <p style={{ margin: 0, fontWeight: 800, color: '#10b981', fontSize: '1.1rem' }}>
                        {payload[0].value} °C
                    </p>
                </div>
            );
        }
        return null;
    };

    const renderConfidence = (score) => (
        <div className="conf-indicator">
            <span className="conf-text">
                Confidence: {score}%
                <span className="pred-tooltip-wrapper">
                    <Info size={14} className="info-trigger" />
                    <span className="pred-tooltip">
                        This indicates GoMata's certainty that the prediction will occur. A {score}% score implies high historical correlation for similar telemetry patterns.
                    </span>
                </span>
            </span>
            <div className="conf-track">
                <div className="conf-fill" style={{ width: `${score}%` }}></div>
            </div>
        </div>
    );

    return (
        <div className="predictions-layout">
            {/* Standard GoMata Sidebar */}
            <aside className={`sidebar-premium ${!isSidebarOpen ? 'collapsed' : ''}`}>
                <div className="sidebar-logo">
                    <Activity size={28} className="brand-icon" />
                    <span>GoMata</span>
                </div>
                <div className="sidebar-scroll-container">
                    <nav className="sidebar-nav">
                        <div className="sidebar-section-title">Main</div>
                        <div className="nav-item-premium" onClick={() => navigate('/dashboard')}><LayoutDashboard className="nav-icon" /> <span>Overview</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/livestock')}><List className="nav-icon" /> <span>Farm & Livestock</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/map')}><MapIcon className="nav-icon" /> <span>Map Intelligence</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/health-analytics')}><Activity className="nav-icon" /> <span>Health Analytics</span></div>
                        <div className="nav-item-premium active" onClick={() => navigate('/predictions')}><Sparkles className="nav-icon" /> <span>AI Predictions</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/alerts')}><AlertTriangle className="nav-icon" /> <span>Alerts Center</span></div>
                        <div className="sidebar-section-title">Operations</div>
                        <div className="nav-item-premium" onClick={() => navigate('/devices')}><Cpu className="nav-icon" /> <span>Devices</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/farm')}><Globe className="nav-icon" /> <span>Farms & Locations</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/breeds')}><Layers className="nav-icon" /> <span>Breeds</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/staff')}><Users className="nav-icon" /> <span>Staff</span></div>
                    </nav>
                </div>
            </aside>

            {/* Main Dashboard Content */}
            <main className="predictions-main">

                {/* Header with GoMata AI */}
                <header className="pred-header">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)} style={{ background: '#f4f6f5', color: '#1a362a', border: 'none' }}>
                            <Menu size={20} />
                        </button>
                        <div className="pred-title-area" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
                            <h1 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}><Sparkles strokeWidth={2.5} size={28} color="#10b981" /> GoMata AI</h1>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <p style={{ margin: 0 }}>Predictive Intelligence & Foresight</p>
                                <span style={{ color: '#cbd5e1', fontSize: '1rem' }}>•</span>
                                <button className="btn-pulse-data" onClick={() => navigate('/predictions/data')}>
                                    <Activity size={14} /> View AI Data Breakdown
                                </button>
                            </div>
                        </div>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center' }}>
                        <div className="pred-horizon-controls">
                            <span style={{ color: '#6b7d73', fontSize: '0.85rem', fontWeight: 600, padding: '8px', opacity: 0.7 }}>Window:</span>
                            <button className={`pred-horizon-btn ${horizon === '24H' ? 'active' : ''}`} onClick={() => setHorizon('24H')}>24 hrs</button>
                            <button className={`pred-horizon-btn ${horizon === '3D' ? 'active' : ''}`} onClick={() => setHorizon('3D')}>3 days</button>
                            <button className={`pred-horizon-btn ${horizon === '7D' ? 'active' : ''}`} onClick={() => setHorizon('7D')}>7 days</button>
                            <button className={`pred-horizon-btn ${horizon === '30D' ? 'active' : ''}`} onClick={() => setHorizon('30D')}>30 days</button>
                        </div>

                        <div className="pred-user-controls">
                            <div className="pred-control-icon"><Search size={20} /></div>
                            <div className="pred-control-icon"><Bell size={20} /></div>
                            <div className="pred-control-icon" style={{ background: '#10b981', color: 'white' }}><UserAvatar name={user?.name} /></div>
                        </div>
                    </div>
                </header>

                {/* What Changed Summary Banner */}
                <div className="pred-delta-banner">
                    <div className="delta-icon"><Activity size={24} /></div>
                    <p className="delta-text">
                        <strong>What Changed Overnight:</strong> 2 animals moved from Low to High risk. Disease spread probability in Barn A increased by +3 animals. DEV023 battery dropped below 30%.
                    </p>
                </div>

                <div style={{ padding: '0 40px' }}>
                    {/* Top Stats Row */}
                    <div className="pred-stats-row" style={{ padding: 0 }}>
                        {/* Disease Spread - Navigational Gateway (New Behavior) */}
                        <div className="dribbble-stat-card solid-orange" style={{ cursor: 'pointer', transition: 'all 0.2s', border: '1px solid #fde68a' }} onClick={() => navigate('/disease-risk')} onMouseEnter={(e) => e.currentTarget.style.borderColor = '#f59e0b'} onMouseLeave={(e) => e.currentTarget.style.borderColor = '#fde68a'}>
                            <div className="stat-left">
                                <h3>Disease Spread Risk</h3>
                                <p style={{ color: '#f59e0b', fontSize: '1.2rem' }}>Medium Risk Herd</p>
                                <span style={{ fontSize: '0.9rem', color: '#b45309', fontWeight: 600, display: 'flex', alignItems: 'center', gap: '4px' }}>View Full Disease Risk Analysis <ChevronRight size={14} /></span>
                            </div>
                            <div className="stat-icon"><ShieldAlert size={32} color="#f59e0b" /></div>
                        </div>

                        {/* Infrastructure - Styled White with Accent Bar */}
                        <div className="dribbble-stat-card solid">
                            <div className="stat-left">
                                <h3>Infrastructure Stability</h3>
                                <p style={{ color: '#1a362a' }}>92% <span className="trend-badge down-good"><Minus size={16} /> Stable</span></p>
                                <span style={{ fontSize: '0.9rem', color: '#6b7d73', fontWeight: 500 }}>4 devices require imminent attention</span>
                            </div>
                            <div className="stat-icon"><Zap size={32} color="#3b82f6" /></div>
                        </div>
                    </div>

                    <div className="pred-dashboard-body" style={{ padding: '0 0 40px 0' }}>

                        {/* LEFT COLUMN: Deep Data */}
                        <div className="pred-col">

                            {/* Prediction Timeline Chart */}
                            <div className="dribbble-card">
                                <h3 className="card-title"><Thermometer size={24} color="#10b981" /> Health Trend Forecast</h3>
                                <div className="pred-chart-container">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={healthForecastData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#f0f2f1" vertical={false} />
                                            <XAxis dataKey="day" stroke="#6b7d73" fontSize={12} tickLine={false} axisLine={false} />
                                            <YAxis domain={[37.5, 40]} stroke="#6b7d73" fontSize={12} tickLine={false} axisLine={false} />
                                            <RechartsTooltip content={<CustomTooltip />} />
                                            <ReferenceLine y={39.0} stroke="#ef4444" strokeDasharray="5 5" label={{ position: 'top', value: 'Critical Fever Limit', fill: '#ef4444', fontSize: 10 }} />
                                            <Line type="monotone" dataKey="temp" stroke="#10b981" strokeWidth={4} dot={{ r: 5, fill: '#fff', stroke: '#10b981', strokeWidth: 3 }} activeDot={{ r: 8, fill: '#10b981', stroke: '#fff', strokeWidth: 3 }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Manage Zones Real Map with Legend & Popups */}
                            <div className="dribbble-card" style={{ padding: 24 }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 16, padding: '0 8px' }}>
                                    <h3 className="card-title" style={{ margin: 0 }}><MapPin size={24} color="#10b981" /> Risk Zone Intelligence</h3>
                                    <span style={{ fontSize: '0.85rem', color: '#10b981', fontWeight: 600, background: '#f0fdf4', padding: '4px 12px', borderRadius: '12px' }}>
                                        <Sparkles size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'text-bottom' }} />
                                        Click map zones for AI insights
                                    </span>
                                </div>
                                <div className="pred-map-wrapper">
                                    <MapContainer center={[34.0515, -118.2437]} zoom={15} style={{ height: '100%', width: '100%' }} zoomControl={false}>
                                        <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" attribution='&copy; OpenStreetMap' />

                                        {/* Barn A Risk Zone */}
                                        <Circle center={[34.053, -118.242]} radius={150} pathOptions={{ color: '#ef4444', fillColor: '#ef4444', fillOpacity: 0.2 }}>
                                            <Popup className="custom-popup">
                                                <div className="popup-header">Barn A</div>
                                                <div className="popup-body">
                                                    <p><span>Population:</span> <strong>45 Heads</strong></p>
                                                    <p><span>Predicted Risk:</span> <strong style={{ color: '#ef4444' }}>Thermal Shock / Fever</strong></p>
                                                    <div className="popup-action">Calculate Relocation Route →</div>
                                                </div>
                                            </Popup>
                                        </Circle>

                                        {/* North Pasture Safe Zone */}
                                        <Circle center={[34.050, -118.246]} radius={200} pathOptions={{ color: '#10b981', fillColor: '#10b981', fillOpacity: 0.2 }}>
                                            <Popup className="custom-popup">
                                                <div className="popup-header">North Pasture</div>
                                                <div className="popup-body">
                                                    <p><span>Population:</span> <strong>120 Heads</strong></p>
                                                    <p><span>Predicted Risk:</span> <strong style={{ color: '#10b981' }}>None (Stable)</strong></p>
                                                </div>
                                            </Popup>
                                        </Circle>
                                    </MapContainer>

                                    {/* Map Legend */}
                                    <div className="map-legend">
                                        <div className="legend-item"><div className="legend-color" style={{ background: 'rgba(239,68,68,0.4)', border: '2px solid #ef4444' }}></div> High Risk Area</div>
                                        <div className="legend-item"><div className="legend-color" style={{ background: 'rgba(16,185,129,0.4)', border: '2px solid #10b981' }}></div> Safe / Optimal</div>
                                    </div>
                                </div>
                            </div>

                            {/* High Risk Animals Table */}
                            <div className="dribbble-card" style={{ padding: '32px 0' }}>
                                <h3 className="card-title" style={{ padding: '0 32px' }}><AlertTriangle size={24} color="#ef4444" /> High Risk Animals ({horizon})</h3>
                                <div style={{ overflowX: 'auto' }}>
                                    <table className="pred-light-table" style={{ width: 'calc(100% - 64px)', margin: '0 32px' }}>
                                        <thead>
                                            <tr>
                                                <th>Animal ID</th>
                                                <th>Current Temp</th>
                                                <th>Forecast Peak</th>
                                                <th>Risk Probability</th>
                                                <th>Time to Risk</th>
                                                <th>Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {highRiskAnimals.map((a, i) => (
                                                <tr key={i}>
                                                    <td>{a.id}</td>
                                                    <td>{a.current}</td>
                                                    <td style={{ fontWeight: 700, color: '#1a362a' }}>{a.forecast}</td>
                                                    <td>
                                                        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                                                            <span className={`pill-status ${a.risk.toLowerCase()}`}>{a.risk}</span>
                                                            {renderConfidence(a.confidence)}
                                                        </div>
                                                    </td>
                                                    <td style={{ color: '#6b7d73' }}>{a.time}</td>
                                                    <td>
                                                        <button className="pred-action-btn" onClick={() => handleInspect(a)}>
                                                            Inspect <Sparkles size={14} />
                                                        </button>
                                                    </td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                        </div>

                        {/* RIGHT COLUMN: Summaries & Actions */}
                        <div className="pred-col">

                            {/* AI Risk Distribution */}
                            <div className="dribbble-card">
                                <h3 className="card-title"><Layers size={24} color="#3b82f6" /> Risk Distribution (Individual)</h3>
                                <div className="risk-dist-grid">
                                    <div className="risk-dist-item">
                                        <span className="risk-dist-num" style={{ color: '#ef4444' }}>5</span>
                                        <span className="risk-dist-label">High Risk</span>
                                    </div>
                                    <div className="risk-dist-item">
                                        <span className="risk-dist-num" style={{ color: '#f59e0b' }}>7</span>
                                        <span className="risk-dist-label">Medium Risk</span>
                                    </div>
                                    <div className="risk-dist-item">
                                        <span className="risk-dist-num" style={{ color: '#10b981' }}>130</span>
                                        <span className="risk-dist-label">Low Risk</span>
                                    </div>
                                </div>
                                <p style={{ fontSize: '0.85rem', color: '#64748b', marginTop: '16px', textAlign: 'center', padding: '12px', background: '#f8fafc', borderRadius: '12px' }}>
                                    See <span onClick={() => navigate('/disease-risk')} style={{ color: '#3b82f6', cursor: 'pointer', fontWeight: 600 }}>Disease Risk</span> page for herd-level spread analysis.
                                </p>
                            </div>

                            {/* Preventive Opportunities (Prioritized & Actionable) */}
                            <div className="dribbble-card">
                                <h3 className="card-title"><Sparkles size={24} color="#10b981" /> Optimization Path</h3>
                                <div className="opportunity-list">
                                    {opportunities.map(opt => (
                                        <div key={opt.id} className="opportunity-card">
                                            <div className={`opp-priority-tag ${opt.priority}`}>{opt.badge}</div>
                                            <div className={`opp-icon ${opt.icon}`}><Navigation size={24} strokeWidth={2.5} /></div>
                                            <div className="opp-content">
                                                <h4>{opt.title}</h4>
                                                <p>{opt.desc}</p>
                                                <div className="opp-actions">
                                                    <button className="btn-quick-fix">Action</button>
                                                    <button className="btn-dismiss" onClick={() => handleDismiss(opt.id)}>Dismiss</button>
                                                </div>
                                            </div>
                                        </div>
                                    ))}
                                    {opportunities.length === 0 && (
                                        <div style={{ textAlign: 'center', padding: '20px', color: '#6b7d73' }}>All optimizations scheduled!</div>
                                    )}
                                </div>
                            </div>

                            {/* Device Failure Table */}
                            <div className="dribbble-card" style={{ padding: '32px 0' }}>
                                <h3 className="card-title" style={{ padding: '0 32px' }}><Battery size={24} color="#10b981" /> Hardware Degradation</h3>
                                <div style={{ overflowX: 'auto' }}>
                                    <table className="pred-light-table" style={{ width: 'calc(100% - 64px)', margin: '0 32px' }}>
                                        <thead><tr><th>SYS ID</th><th>Battery</th><th>Risk</th><th>Failure T-Minus</th></tr></thead>
                                        <tbody>
                                            {deviceFailures.map((d, i) => (
                                                <tr key={i}>
                                                    <td>{d.id}<br /><span style={{ fontSize: '0.75rem', color: '#6b7d73' }}>{d.animal}</span></td>
                                                    <td style={{ color: '#ef4444', fontWeight: 700 }}>{d.battery}</td>
                                                    <td><span className={`pill-status ${d.risk.toLowerCase()}`}>{d.risk}</span></td>
                                                    <td style={{ fontWeight: 600 }}>{d.time}</td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </div>

                        </div>
                    </div>
                </div>
            </main>

            {/* Slide-out AI Inspector Panel */}
            <div className={`inspector-overlay ${inspectedAnimal ? 'open' : ''}`} onClick={(e) => {
                if (e.target.className.includes('overlay')) setInspectedAnimal(null);
            }}>
                <div className="inspector-panel">
                    <div className="insp-header">
                        <div>
                            <h2><Sparkles color="#10b981" /> {inspectedAnimal?.id}</h2>
                            <p>AI Diagnostic Profile</p>
                        </div>
                        <button className="btn-close-insp" onClick={() => setInspectedAnimal(null)}><X size={20} /></button>
                    </div>

                    <div className="insp-body">
                        <div className="insp-reasoning-box">
                            <h4><Activity size={18} /> Model Reasoning</h4>
                            <p>Core temperature rose 0.8°C over the last 6 hours coupled with a localized activity drop of 40%. Historical thermal signatures indicate a {inspectedAnimal?.confidence}% probability of severe clinical fever within {inspectedAnimal?.time}.</p>
                        </div>

                        <div className="insp-detail-row">
                            <span className="insp-detail-label">Current Telemetry</span>
                            <span className="insp-detail-val">{inspectedAnimal?.current}</span>
                        </div>
                        <div className="insp-detail-row">
                            <span className="insp-detail-label">Forecast Peak</span>
                            <span className="insp-detail-val" style={{ color: '#ef4444' }}>{inspectedAnimal?.forecast}</span>
                        </div>
                        <div className="insp-detail-row">
                            <span className="insp-detail-label">Hardware Link</span>
                            <span className="insp-detail-val">Active (DEV023)</span>
                        </div>
                        <div className="insp-detail-row">
                            <span className="insp-detail-label">Zone Bound</span>
                            <span className="insp-detail-val">Barn A</span>
                        </div>

                        {/* Mini trend chart representation for the inspector */}
                        <div className="insp-chart-mini">
                            <ResponsiveContainer width="100%" height="100%">
                                <LineChart data={[
                                    { h: '-4h', t: 38.0 }, { h: '-2h', t: 38.2 }, { h: 'Now', t: 38.5 }, { h: '+2h', t: 38.9 }, { h: '+4h', t: 39.6 }
                                ]}>
                                    <XAxis dataKey="h" fontSize={10} tickLine={false} axisLine={false} stroke="#94a3b8" />
                                    <Line type="monotone" dataKey="t" stroke="#ef4444" strokeWidth={3} dot={{ r: 3, fill: '#ef4444' }} />
                                </LineChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="insp-actions">
                        <button className="insp-btn primary">Notify Farm Veterinarian</button>
                        <button className="insp-btn secondary">Quarantine Animal (Barn C)</button>
                    </div>
                </div>
            </div>

        </div>
    );
}

// Helper simple avatar
const UserAvatar = ({ name }) => <span>{name ? name.charAt(0).toUpperCase() : 'A'}</span>;

export default Predictions;
