import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip as RechartsTooltip, BarChart, Bar, LineChart, Line, PieChart, Pie, Cell } from 'recharts';
import {
    Activity, Map as MapIcon, Sparkles, FileText, ArrowLeft, Bell, Settings, Filter, Download,
    Thermometer, CheckCircle, AlertTriangle, AlertCircle, ChevronDown, ChevronRight, LayoutDashboard,
    List, Cpu, Layers, Globe, Shield, LogOut, Users, TrendingUp, BarChart2, ShieldAlert, Link as LinkIcon, Menu, Radio, HeartPulse, Image, Search, Plus, User, Zap
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './HealthAnalytics.css';
import './Dashboard.css';
import { livestockAPI } from '../services/api';
// --- MOCK DATA ---
const mockHerdHealthDistribution = [
    { name: 'Healthy', value: 84, color: '#10b981' },
    { name: 'Warning', value: 10, color: '#f59e0b' },
    { name: 'Critical', value: 6, color: '#ef4444' }
];

const tempTrendData = [
    { time: '12 AM', avg: 38.2, max: 38.5, min: 38.0 },
    { time: '4 AM', avg: 38.1, max: 38.3, min: 37.9 },
    { time: '8 AM', avg: 38.4, max: 38.8, min: 38.1 },
    { time: '12 PM', avg: 38.6, max: 39.2, min: 38.3 },
    { time: '4 PM', avg: 38.5, max: 39.0, min: 38.2 },
    { time: '8 PM', avg: 38.3, max: 38.7, min: 38.0 }
];

const activityTrendData = [
    { day: 'Mon', active: 85, inactive: 15 },
    { day: 'Tue', active: 88, inactive: 12 },
    { day: 'Wed', active: 82, inactive: 18 },
    { day: 'Thu', active: 90, inactive: 10 },
    { day: 'Fri', active: 75, inactive: 25 },
    { day: 'Sat', active: 86, inactive: 14 },
    { day: 'Sun', active: 89, inactive: 11 }
];

const healthStatusTrendData = [
    { date: 'Day 1', healthy: 120, warning: 15, critical: 4 },
    { date: 'Day 2', healthy: 122, warning: 12, critical: 5 },
    { date: 'Day 3', healthy: 118, warning: 18, critical: 3 },
    { date: 'Day 4', healthy: 125, warning: 10, critical: 4 },
    { date: 'Day 5', healthy: 128, warning: 9, critical: 5 }
];

const zoneHealthData = [
    { zone: 'Barn A', healthy: 95, warning: 5, critical: 0 },
    { zone: 'Barn B', healthy: 82, warning: 12, critical: 6 },
    { zone: 'Barn C', healthy: 70, warning: 20, critical: 10 },
    { zone: 'Pasture 1', healthy: 98, warning: 2, critical: 0 }
];

function HealthAnalytics() {
    const navigate = useNavigate();
    const [dateRange, setDateRange] = useState('Last 7 days');
    const { user } = useContext(AuthContext);
    
    // Live Data State
    const [realHerdData, setRealHerdData] = useState([]);
    const [herdDist, setHerdDist] = useState(mockHerdHealthDistribution);
    const [highRisk, setHighRisk] = useState([]);
    const [stats, setStats] = useState({ healthy: 0, warning: 0, critical: 0, total: 0 });
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [showFarmModal, setShowFarmModal] = useState(false);
    const [showProfileModal, setShowProfileModal] = useState(false);
    const profileRef = useRef();

    useEffect(() => {
        const fetchStats = async () => {
            try {
                const response = await livestockAPI.getAll();
                if (response && response.data) {
                    const data = response.data;
                    setRealHerdData(data);
                    
                    let hCount = 0;
                    let wCount = 0;
                    let cCount = 0;
                    const risks = [];

                    data.forEach(item => {
                        const l = item.livestock;
                        const temp = item.latestSensorData?.temperature || 38.0;
                        let stat = 'Healthy';
                        let riskLvl = 'Low';

                        if (temp > 39.5) {
                            stat = 'Critical';
                            cCount++;
                            riskLvl = 'High';
                        } else if (temp > 39.0 || temp < 38.0) {
                            stat = 'Warning';
                            wCount++;
                            riskLvl = 'Medium';
                        } else {
                            hCount++;
                        }

                        if (stat !== 'Healthy') {
                            risks.push({
                                id: l.name || l.tagNumber,
                                temp: temp.toFixed(1),
                                risk: riskLvl,
                                trend: 'Fluctuating',
                                location: l.location || 'Unknown',
                                lastUpdate: item.latestSensorData?.timestamp ? new Date(item.latestSensorData.timestamp).toLocaleTimeString() : 'Recently'
                            });
                        }
                    });

                    setStats({ healthy: hCount, warning: wCount, critical: cCount, total: data.length });
                    
                    setHerdDist([
                        { name: 'Healthy', value: hCount, color: '#10b981' },
                        { name: 'Warning', value: wCount, color: '#f59e0b' },
                        { name: 'Critical', value: cCount, color: '#ef4444' }
                    ]);
                    
                    setHighRisk(risks.slice(0, 5)); // Show top 5 risks
                }
            } catch (err) {
                console.error("Failed to fetch livestock data for Health Analytics:", err);
            }
        };
        fetchStats();
    }, []);

    useEffect(() => {
        const listener = (event) => {
            if (!profileRef.current || profileRef.current.contains(event.target)) {
                return;
            }
            if (showProfileMenu) setShowProfileMenu(false);
        };
        document.addEventListener("click", listener);
        document.addEventListener("touchstart", listener);
        return () => {
            document.removeEventListener("click", listener);
            document.removeEventListener("touchstart", listener);
        };
    }, [showProfileMenu]);

    return (
        <div className="dashboard-layout-premium">
            {/* SIDEBAR */}
            <aside className={`sidebar-premium ${!isSidebarOpen ? 'collapsed' : ''}`}>
                <div className="sidebar-logo">
                    <Activity size={28} className="brand-icon" />
                    <span>GoMata</span>
                </div>

                <div className="sidebar-scroll-container">
                    <nav className="sidebar-nav">
                        <div className="sidebar-section-title">Main</div>
                        <div className="nav-item-premium" onClick={() => navigate('/dashboard')}>
                            <LayoutDashboard className="nav-icon" /> <span>Overview</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/livestock')}>
                            <List className="nav-icon" /> <span>Animals</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/map')}>
                            <MapIcon className="nav-icon" /> <span>Map Intelligence</span>
                        </div>
                        <div className="nav-item-premium active" onClick={() => navigate('/health-analytics')}>
                            <Activity className="nav-icon" /> <span>Health Analytics</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/alerts')}>
                            <AlertTriangle className="nav-icon" /> <span>Alerts Center</span>
                        </div>

                        <div className="sidebar-section-title">Operations</div>
                        <div className="nav-item-premium" onClick={() => navigate('/devices')}>
                            <Cpu className="nav-icon" /> <span>Devices</span>
                        </div>
                        {user?.type !== 'staff' && (
                            <>
                                <div className="nav-item-premium" onClick={() => navigate('/farm')}>
                                    <Globe className="nav-icon" /> <span>Farms & Locations</span>
                                </div>
                                <div className="nav-item-premium" onClick={() => navigate('/breeds')}>
                                    <Layers className="nav-icon" /> <span>Breeds</span>
                                </div>
                                <div className="nav-item-premium" onClick={() => navigate('/staff')}>
                                    <Users className="nav-icon" /> <span>Staff</span>
                                </div>
                            </>
                        )}

                        <div className="sidebar-section-title">Intelligence</div>
                        <div className="nav-item-premium" onClick={() => navigate('/ai-orchestrator')}>
                            <Zap className="nav-icon" /> <span>AI Orchestrator</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/predictions')}>
                            <TrendingUp className="nav-icon" /> <span>Predictions</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/behavior')}>
                            <BarChart2 className="nav-icon" /> <span>Behavior Analysis</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/disease-risk')}>
                            <ShieldAlert className="nav-icon" /> <span>Disease Risk</span>
                        </div>

                        {user?.type !== 'staff' && (
                            <>
                                <div className="sidebar-section-title">System</div>
                                <div className="nav-item-premium" onClick={() => navigate('/reports')}>
                                    <FileText className="nav-icon" /> <span>Reports</span>
                                </div>
                                <div className="nav-item-premium" onClick={() => navigate('/integrations')}>
                                    <LinkIcon className="nav-icon" /> <span>Integrations</span>
                                </div>
                                <div className="nav-item-premium" onClick={() => navigate('/settings')}>
                                    <Settings className="nav-icon" /> <span>Settings</span>
                                </div>
                            </>
                        )}
                    </nav>
                </div>

                <div className="sidebar-bottom">
                    <div className="nav-item-premium" onClick={() => setShowProfileModal(true)}>
                        <User className="nav-icon" /> <span>Profile</span>
                    </div>
                    <div className="nav-item-premium logout" onClick={() => navigate('/')}>
                        <LogOut className="nav-icon" /> <span>Logout</span>
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT */}
            <main className="main-content-premium">
                {/* TOP BAR */}
                <header className="topbar-premium" style={{ marginBottom: '32px' }}>
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div>
                            <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700, color: '#0f172a' }}>Health Analytics</h1>
                            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Monitor herd health trends and detect risks early.</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <div className="control-btn dropdown">
                            <span>{dateRange}</span>
                            <ChevronDown size={16} />
                        </div>
                        <div className="control-btn dropdown">
                            <span>Farm A</span>
                            <ChevronDown size={16} />
                        </div>
                        <div className="control-btn action">
                            <Download size={16} /> Export
                        </div>
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                <div style={{ padding: '0 32px 32px', display: 'flex', flexDirection: 'column', gap: '32px' }}>

                    {/* KPI Cards */}
                    <div className="kpi-grid">
                        <div className="kpi-card">
                            <div className="kpi-icon"><Activity size={24} color="#3b82f6" /></div>
                            <div className="kpi-info">
                                <span className="lbl">Herd Health Score</span>
                                <div className="val-row">
                                    <span className="val">{stats.total > 0 ? Math.round((stats.healthy / stats.total) * 100) : 0}%</span>
                                    <span className="trend stable">Live</span>
                                </div>
                            </div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-icon"><CheckCircle size={24} color="#10b981" /></div>
                            <div className="kpi-info">
                                <span className="lbl">Healthy Animals</span>
                                <div className="val-row">
                                    <span className="val">{stats.healthy}</span>
                                    <span className="lbl-sub">{stats.total > 0 ? Math.round((stats.healthy / stats.total) * 100) : 0}% of herd</span>
                                </div>
                            </div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-icon"><AlertTriangle size={24} color="#f59e0b" /></div>
                            <div className="kpi-info">
                                <span className="lbl">Warning Animals</span>
                                <div className="val-row">
                                    <span className="val">{stats.warning}</span>
                                    {stats.warning > 0 && <span className="trend down">Needs Attention</span>}
                                </div>
                            </div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-icon"><AlertCircle size={24} color="#ef4444" /></div>
                            <div className="kpi-info">
                                <span className="lbl">Critical Animals</span>
                                <div className="val-row">
                                    <span className="val">{stats.critical}</span>
                                    {stats.critical > 0 && <span className="trend up alert">High Risk</span>}
                                </div>
                            </div>
                        </div>
                        <div className="kpi-card">
                            <div className="kpi-icon"><Thermometer size={24} color="#6366f1" /></div>
                            <div className="kpi-info">
                                <span className="lbl">Avg Temperature</span>
                                <div className="val-row">
                                    <span className="val">38.4°C</span>
                                    <span className="trend stable">Normal</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="content-grid-hybrid">
                        {/* Left Column (Charts & Tables) */}
                        <div className="charts-col">

                            {/* Row 1: Distribution & Temp Trend */}
                            <div className="chart-row">
                                <div className="health-card dist-card">
                                    <h3>Health Distribution</h3>
                                    <ResponsiveContainer width="100%" height={220}>
                                        <PieChart>
                                            <Pie data={herdDist} cx="50%" cy="50%" innerRadius={60} outerRadius={80} paddingAngle={5} dataKey="value">
                                                {herdDist.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                                ))}
                                            </Pie>
                                            <RechartsTooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.1)' }} />
                                        </PieChart>
                                    </ResponsiveContainer>
                                    <div className="legend-row">
                                        <div className="leg-item"><div className="dot healthy"></div>{stats.total > 0 ? Math.round((stats.healthy / stats.total) * 100) : 0}% Healthy</div>
                                        <div className="leg-item"><div className="dot warning"></div>{stats.total > 0 ? Math.round((stats.warning / stats.total) * 100) : 0}% Warning</div>
                                        <div className="leg-item"><div className="dot critical"></div>{stats.total > 0 ? Math.round((stats.critical / stats.total) * 100) : 0}% Critical</div>
                                    </div>
                                </div>
                                <div className="health-card temp-trend-card">
                                    <h3>Temperature Trend</h3>
                                    <ResponsiveContainer width="100%" height={220}>
                                        <LineChart data={tempTrendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                            <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                            <YAxis domain={['dataMin - 0.5', 'dataMax + 0.5']} axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                            <RechartsTooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.1)' }} />
                                            <Line type="monotone" dataKey="avg" stroke="#3b82f6" strokeWidth={3} dot={{ r: 4, fill: '#3b82f6' }} />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* Row 2: Status Trend & Activity */}
                            <div className="chart-row">
                                <div className="health-card status-trend-card">
                                    <h3>Health Status Trend</h3>
                                    <ResponsiveContainer width="100%" height={220}>
                                        <AreaChart data={healthStatusTrendData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                            <XAxis dataKey="date" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                            <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                            <RechartsTooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.1)' }} />
                                            <Area type="monotone" dataKey="healthy" stackId="1" stroke="#10b981" fill="#d1fae5" />
                                            <Area type="monotone" dataKey="warning" stackId="1" stroke="#f59e0b" fill="#fef3c7" />
                                            <Area type="monotone" dataKey="critical" stackId="1" stroke="#ef4444" fill="#fee2e2" />
                                        </AreaChart>
                                    </ResponsiveContainer>
                                </div>
                                <div className="health-card zone-card">
                                    <h3>Zone Health Comparison</h3>
                                    <ResponsiveContainer width="100%" height={220}>
                                        <BarChart data={zoneHealthData} layout="vertical" margin={{ top: 10, right: 10, left: 10, bottom: 0 }}>
                                            <XAxis type="number" hide />
                                            <YAxis dataKey="zone" type="category" axisLine={false} tickLine={false} tick={{ fill: '#64748b', fontSize: 12, fontWeight: 500 }} width={80} />
                                            <RechartsTooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 8px 24px rgba(0,0,0,0.1)' }} />
                                            <Bar dataKey="healthy" stackId="a" fill="#10b981" radius={[0, 0, 0, 0]} />
                                            <Bar dataKey="warning" stackId="a" fill="#f59e0b" />
                                            <Bar dataKey="critical" stackId="a" fill="#ef4444" radius={[0, 4, 4, 0]} />
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                            </div>

                            {/* High Risk Animals Table */}
                            <div className="health-card risk-table-card">
                                <div className="card-header-flex">
                                    <h3>High Priority Risks</h3>
                                    <button className="view-all-btn">View All</button>
                                </div>
                                <div className="table-responsive">
                                    <table className="premium-table">
                                        <thead>
                                            <tr>
                                                <th>Animal ID</th>
                                                <th>Temperature</th>
                                                <th>Risk Level</th>
                                                <th>Trend</th>
                                                <th>Location</th>
                                                <th>Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {highRisk.length > 0 ? highRisk.map((animal, i) => (
                                                <tr key={i}>
                                                    <td className="fw-600">{animal.id}</td>
                                                    <td className="fw-600">{animal.temp}°C</td>
                                                    <td>
                                                        <span className={`risk-badge ${animal.risk.toLowerCase()}`}>{animal.risk}</span>
                                                    </td>
                                                    <td>
                                                        <span className={`trend-text ${animal.trend === 'Rising' ? 'danger' : 'safe'}`}>{animal.trend}</span>
                                                    </td>
                                                    <td>{animal.location}</td>
                                                    <td>
                                                        <button className="action-sm-btn" onClick={() => navigate(`/livestock/${animal.id}`)}>Locate <ChevronRight size={14} /></button>
                                                    </td>
                                                </tr>
                                            )) : (
                                                <tr>
                                                    <td colSpan="6" style={{ textAlign: 'center', padding: '24px', color: '#64748b' }}>
                                                        No high-priority risks detected. Herd is healthy.
                                                    </td>
                                                </tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>

                        {/* Right Column: AI Insights */}
                        <div className="ai-col">
                            <div className="ai-insight-panel">
                                <div className="panel-header">
                                    <Sparkles size={24} color="#10b981" />
                                    <h2>GoMata AI Insights</h2>
                                </div>

                                <div className="insight-body">
                                    <div className="insight-block main-summary">
                                        <h4>Herd Overview</h4>
                                        <p>Herd health is generally stable. However, localized elevated temperatures indicate early signs of thermal stress in enclosed areas.</p>
                                    </div>

                                    <div className="insight-block warning-block">
                                        <h4>Early Warning Indicators</h4>
                                        <ul className="warning-list">
                                            <li><AlertTriangle size={16} /> <strong>{stats.warning} animals</strong> showing early signs of fluctuation.</li>
                                            <li><Activity size={16} /> <strong>{stats.critical} animals</strong> exhibiting high-risk vitals.</li>
                                            <li><Thermometer size={16} /> Elevated ambient risk detected in assigned zones.</li>
                                        </ul>
                                    </div>

                                    <div className="insight-block action-block">
                                        <h4>Recommended Actions</h4>
                                        <div className="action-item">
                                            <div className="act-idx">1</div>
                                            <p>Activate supplemental ventilation in <strong>Barn B</strong>.</p>
                                        </div>
                                        <div className="action-item">
                                            <div className="act-idx">2</div>
                                            <p>Visually inspect <strong>MIX005</strong> and <strong>COW042</strong> immediately.</p>
                                        </div>
                                        <div className="action-item">
                                            <div className="act-idx">3</div>
                                            <p>Review water station logs in Barn C.</p>
                                        </div>
                                    </div>
                                </div>

                                <button className="ai-full-btn">
                                    View Detailed AI Report <ArrowLeft size={16} style={{ transform: 'rotate(180deg)' }} />
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default HealthAnalytics;
