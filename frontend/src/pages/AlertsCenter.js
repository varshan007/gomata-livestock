import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    Activity, Map as MapIcon, Sparkles, FileText, Settings, Download,
    Thermometer, AlertTriangle, ChevronDown, LayoutDashboard, CheckCircle2,
    List, Cpu, Layers, Globe, LogOut, Users, TrendingUp, BarChart2, ShieldAlert, Link as LinkIcon, Menu, Search, User, Zap, MapPin, RefreshCw, Clock, Filter, CheckCircle
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './AlertsCenter.css';
import './Dashboard.css';

function AlertsCenter() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [showProfileModal, setShowProfileModal] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const profileRef = useRef();

    // Data States
    const [alerts, setAlerts] = useState([]);
    const [activeSeverity, setActiveSeverity] = useState('All');
    const [selectedAlert, setSelectedAlert] = useState(null);

    // Fetch real alerts from API
    useEffect(() => {
        const fetchAlerts = async () => {
            if (!user) return;
            const token = localStorage.getItem('token');
            if (!token) return;
            try {
                const res = await fetch('http://localhost:8000/api/alerts', {
                    headers: { Authorization: `Bearer ${token}` }
                });
                const json = await res.json();
                const data = json.data || json;

                const formatted = (Array.isArray(data) ? data : []).map(a => {
                    // Use denormalized metadata (new), fall back to populated ref (old)
                    const animalName = a.animalName || a.livestockId?.name || a.livestockId?.livestock_id || 'Unknown';
                    const farmName = a.farmName || 'Unknown farm';
                    const zoneName = a.zoneName || '';
                    const breed = a.breed || '';
                    const deviceId = a.deviceId || '';
                    const probPercent = a.diseaseProbability != null
                        ? (a.diseaseProbability * 100).toFixed(1) + '%'
                        : '';

                    const severityLower = (a.severity || 'Medium').toLowerCase();
                    const ts = a.timestamp ? new Date(a.timestamp) : new Date();
                    const minutesAgo = Math.round((Date.now() - ts.getTime()) / 60000);
                    let timeStr = minutesAgo < 1 ? 'Just now' :
                        minutesAgo < 60 ? `${minutesAgo} min ago` :
                            minutesAgo < 1440 ? `${Math.round(minutesAgo / 60)}h ago` :
                                ts.toLocaleDateString();

                    // Build short display message
                    const shortMsg = a.diseaseProbability != null
                        ? `${probPercent} disease risk detected`
                        : (a.message || `${a.alertType} alert`);

                    return {
                        id: a._id,
                        animalId: animalName,
                        type: a.alertType || 'Health',
                        severity: severityLower,
                        message: shortMsg,
                        location: zoneName ? `${farmName} / ${zoneName}` : farmName,
                        time: timeStr,
                        status: a.resolved ? 'Resolved' : (a.status || 'Active'),
                        details: a.message || 'No additional details.',
                        device: deviceId || 'N/A',
                        breed: breed,
                        probPercent: probPercent,
                        explanation: a.explanation || '',
                        livestockObjId: a.livestockId?._id || a.livestockId
                    };
                });

                setAlerts(formatted);
                if (formatted.length > 0 && !selectedAlert) {
                    setSelectedAlert(formatted[0]);
                }
            } catch (err) {
                console.error("Error fetching alerts:", err);
            }
        };
        fetchAlerts();
        const interval = setInterval(fetchAlerts, 30000); // Refresh every 30s
        return () => clearInterval(interval);
    }, [user]);

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

    const handleRefresh = () => {
        // Trigger re-fetch
        setAlerts([...alerts]);
    };

    const severityFilters = ['All', 'Critical', 'Warning', 'Device', 'Resolved'];

    const filteredAlerts = alerts.filter(a => {
        const matchesSearch = (a.animalId || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
            (a.message || '').toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSeverity = activeSeverity === 'All' ? true :
            activeSeverity === 'Resolved' ? a.status === 'Resolved' :
                a.severity.toLowerCase() === activeSeverity.toLowerCase();
        return matchesSearch && matchesSeverity;
    });

    // Compute KPI stats from live data
    const criticalCount = alerts.filter(a => a.severity === 'critical').length;
    const warningCount = alerts.filter(a => a.severity === 'high' || a.severity === 'warning' || a.severity === 'medium').length;
    const deviceCount = alerts.filter(a => a.type === 'Battery' || a.type === 'Device').length;
    const totalCount = alerts.length;

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
                        <div className="nav-item-premium" onClick={() => navigate('/health-analytics')}>
                            <Activity className="nav-icon" /> <span>Health Analytics</span>
                        </div>
                        <div className="nav-item-premium active" onClick={() => navigate('/alerts')}>
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

                        {user?.type !== 'staff' && (
                            <>
                                <div className="sidebar-section-title">System</div>
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
            <main className="alerts-main" style={{ padding: '0' }}>
                {/* TOP BAR */}
                <header className="topbar-premium" style={{ marginBottom: '16px', padding: '32px 40px 16px 40px' }}>
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div>
                            <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700, color: '#0f172a' }}>Alert Centre</h1>
                            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Monitor and respond to livestock health and device alerts.</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <div className="control-btn" onClick={handleRefresh}>
                            <CheckCircle2 size={16} /> Mark all read
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

                {/* KPI Summary Cards */}
                <div className="alerts-summary-grid">
                    <div className="summary-card">
                        <div className="sc-header">
                            <div className="sc-icon red"><ShieldAlert size={24} /></div>
                            <span className="sc-title">Critical Alerts</span>
                        </div>
                        <div className="sc-value val red">{criticalCount}</div>
                        <span className="sc-desc">Health & Temperature</span>
                    </div>
                    <div className="summary-card">
                        <div className="sc-header">
                            <div className="sc-icon yellow"><AlertTriangle size={24} /></div>
                            <span className="sc-title">Warnings</span>
                        </div>
                        <div className="sc-value val yellow">{warningCount}</div>
                        <span className="sc-desc">Activity & Location Breaches</span>
                    </div>
                    <div className="summary-card">
                        <div className="sc-header">
                            <div className="sc-icon blue"><Cpu size={24} /></div>
                            <span className="sc-title">Device Link</span>
                        </div>
                        <div className="sc-value val blue">{deviceCount}</div>
                        <span className="sc-desc">Low Battery & Offline</span>
                    </div>
                    <div className="summary-card">
                        <div className="sc-header">
                            <div className="sc-icon grey"><Filter size={24} /></div>
                            <span className="sc-title">Total (24h)</span>
                        </div>
                        <div className="sc-value val dark">{totalCount}</div>
                        <span className="sc-desc">Across All Active Farms</span>
                    </div>
                </div>

                <div className="alerts-content-split">
                    {/* Left Column (Filters + List) */}
                    <div className="left-alert-column">
                        <div className="filters-bar">
                            <div className="filter-group">
                                {severityFilters.map(filter => (
                                    <div
                                        key={filter}
                                        className={`filter-pill ${activeSeverity === filter ? 'active' : ''}`}
                                        onClick={() => setActiveSeverity(filter)}
                                    >
                                        {filter}
                                    </div>
                                ))}
                            </div>
                            <div className="search-box">
                                <Search size={16} color="#94a3b8" />
                                <input
                                    type="text"
                                    placeholder="Search animal ID or alert..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>
                        </div>

                        <div className="alert-rows-container">
                            {filteredAlerts.map(alert => (
                                <div
                                    key={alert.id}
                                    className={`alert-item ${alert.severity} ${selectedAlert?.id === alert.id ? 'active' : ''}`}
                                    onClick={() => setSelectedAlert(alert)}
                                >
                                    <div className="alert-icon-col">
                                        {alert.severity === 'critical' && <ShieldAlert size={24} color="#ef4444" />}
                                        {alert.severity === 'warning' && <AlertTriangle size={24} color="#f59e0b" />}
                                        {alert.severity === 'device' && <Cpu size={24} color="#3b82f6" />}
                                        {alert.severity === 'resolved' && <CheckCircle size={24} color="#94a3b8" />}
                                    </div>
                                    <div className="alert-main-col">
                                        <div className="alert-header">
                                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                <span className="ah-id">{alert.animalId !== 'N/A' ? alert.animalId : 'System'}</span>
                                            </div>
                                            <span className="ah-time">{alert.time}</span>
                                        </div>
                                        <p className="alert-msg">{alert.message}</p>
                                        <div className="alert-meta">
                                            <div className="meta-item"><MapPin size={14} /> {alert.location}</div>
                                            <div className="meta-item"><Activity size={14} /> {alert.type}</div>
                                        </div>
                                    </div>
                                </div>
                            ))}
                            {filteredAlerts.length === 0 && (
                                <div style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                                    No alerts match your current filters.
                                </div>
                            )}
                        </div>
                    </div>

                    {/* Right Column (Detail Panel) */}
                    {selectedAlert ? (
                        <div className="right-detail-panel">
                            <div className="detail-card">
                                <div className="detail-header">
                                    <div className={`detail-badge ${selectedAlert.severity}`}>
                                        {selectedAlert.severity === 'critical' && <ShieldAlert size={14} />}
                                        {selectedAlert.severity === 'warning' && <AlertTriangle size={14} />}
                                        {selectedAlert.severity === 'high' && <AlertTriangle size={14} />}
                                        {selectedAlert.severity === 'device' && <Cpu size={14} />}
                                        {selectedAlert.severity === 'resolved' && <CheckCircle size={14} />}
                                        {selectedAlert.severity}
                                    </div>
                                    <h2>{selectedAlert.animalId}</h2>
                                    {selectedAlert.breed && <p style={{ margin: '2px 0 0', fontSize: '0.85rem', color: '#64748b' }}>Breed: {selectedAlert.breed}</p>}
                                    <p className="detail-msg">{selectedAlert.message}</p>
                                </div>

                                <div className="detail-grid">
                                    <div>
                                        <div className="d-label">Location</div>
                                        <div className="d-val"><MapPin size={16} color="#64748b" /> {selectedAlert.location}</div>
                                    </div>
                                    <div>
                                        <div className="d-label">Detected Time</div>
                                        <div className="d-val"><Clock size={16} color="#64748b" /> {selectedAlert.time}</div>
                                    </div>
                                    <div>
                                        <div className="d-label">Alert Type</div>
                                        <div className="d-val"><Activity size={16} color="#64748b" /> {selectedAlert.type}</div>
                                    </div>
                                    <div>
                                        <div className="d-label">Device ID</div>
                                        <div className="d-val"><Cpu size={16} color="#64748b" /> {selectedAlert.device}</div>
                                    </div>
                                    {selectedAlert.probPercent && (
                                        <div>
                                            <div className="d-label">Disease Probability</div>
                                            <div className="d-val" style={{ color: '#ef4444', fontWeight: 700 }}>
                                                <Thermometer size={16} color="#ef4444" /> {selectedAlert.probPercent}
                                            </div>
                                        </div>
                                    )}
                                </div>

                                <div style={{ marginBottom: '24px' }}>
                                    <div className="d-label" style={{ marginBottom: '8px' }}>Detailed Context</div>
                                    <div style={{ fontSize: '0.9rem', color: '#475569', lineHeight: '1.5', background: '#f8fafc', padding: '12px', borderRadius: '12px' }}>
                                        {selectedAlert.details}
                                    </div>
                                </div>

                                {selectedAlert.severity === 'critical' && (
                                    <div className="ai-recommendation">
                                        <h4><Sparkles size={18} /> AI Health Explanation</h4>
                                        {selectedAlert.explanation ? (
                                            <div style={{
                                                fontSize: '0.9rem', color: '#475569', lineHeight: '1.6',
                                                whiteSpace: 'pre-wrap', background: '#f0fdf4',
                                                padding: '14px', borderRadius: '12px', border: '1px solid #bbf7d0'
                                            }}>
                                                {selectedAlert.explanation}
                                            </div>
                                        ) : (
                                            <ul className="ai-rec-list">
                                                <li><CheckCircle2 size={16} color="#14b8a6" /> Dispatch veterinary staff to {selectedAlert.location} immediately.</li>
                                                <li><CheckCircle2 size={16} color="#14b8a6" /> Perform physical examination on {selectedAlert.animalId} — check for fever, dehydration, and respiratory distress.</li>
                                                <li><CheckCircle2 size={16} color="#14b8a6" /> Isolate {selectedAlert.animalId} to quarantine zone if signs of contagious disease are observed.</li>
                                            </ul>
                                        )}
                                    </div>
                                )}

                                <div className="detail-actions">
                                    {selectedAlert.animalId !== 'N/A' && (
                                        <button className="action-btn primary" onClick={() => navigate(`/livestock/${selectedAlert.livestockObjId || selectedAlert.animalId}`)}>
                                            View Animal Details
                                        </button>
                                    )}
                                    <button
                                        className="action-btn"
                                        style={{ background: 'linear-gradient(135deg, #10b981, #059669)', color: '#fff', border: 'none' }}
                                        onClick={() => navigate('/ai-orchestrator', {
                                            state: {
                                                alertId: selectedAlert.id,
                                                animalId: selectedAlert.livestockObjId,
                                                animalName: selectedAlert.animalId,
                                                autoQuery: `Tell me more about the health alert for ${selectedAlert.animalId}. What actions should I take?`
                                            }
                                        })}
                                    >
                                        <Sparkles size={18} /> Ask GoMata AI
                                    </button>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="right-detail-panel" style={{ alignItems: 'center', justifyContent: 'center', color: '#94a3b8' }}>
                            <AlertTriangle size={48} style={{ opacity: 0.2, marginBottom: '16px' }} />
                            Select an alert to view details
                        </div>
                    )}
                </div>
            </main>
        </div>
    );
}

export default AlertsCenter;
