import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Map as MapIcon, Activity, AlertTriangle, Cpu, Layers, Globe, Users, Settings, LogOut, ChevronDown, User, Menu,
    Search, Plus, Download, Upload, Edit, Eye, Trash2, Battery, Filter, CheckCircle2, ChevronRight, Link as LinkIcon, X, Wifi, WifiOff, BatteryWarning
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css'; // Reuse core premium styles
import './Devices.css';

// Mock Data replaced by live IoT endpoints in Phase 9

function Devices() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterStatus, setFilterStatus] = useState('All');
    const [activeDevice, setActiveDevice] = useState(null); // Detailed side panel
    const [showAddModal, setShowAddModal] = useState(false);
    const profileRef = useRef();

    useEffect(() => {
        const listener = (event) => {
            if (!profileRef.current || profileRef.current.contains(event.target)) return;
            if (showProfileMenu) setShowProfileMenu(false);
        };
        document.addEventListener("click", listener);
        return () => document.removeEventListener("click", listener);
    }, [showProfileMenu]);

    const [liveDevices, setLiveDevices] = useState([]);

    useEffect(() => {
        const fetchDevices = async () => {
            if (!user) return;
            const token = localStorage.getItem('token');
            if (!token) return;
            try {
                const res = await fetch(`${process.env.REACT_APP_API_URL || 'https://gomata-backend.onrender.com/api'}/devices`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                const json = await res.json();
                const data = json.data || json;

                // Format mapped payload for existing UI render functions
                const formatted = (Array.isArray(data) ? data : []).map(d => {
                    let health = 'Good';
                    if (d.batteryRaw !== null && d.batteryRaw < 20) health = 'Needs charging';
                    if (d.status === 'Offline') health = 'Needs Maintenance';

                    let signalStr = 'No Signal';
                    if (d.signal > -60) signalStr = 'Strong';
                    else if (d.signal > -80) signalStr = 'Medium';
                    else if (d.signal > -100) signalStr = 'Weak';

                    return {
                        ...d,
                        battery: d.batteryRaw || 0, // Fallback integer
                        signal: signalStr,
                        firmware: 'v1.2.3', // Mock firmware string for now
                        health: health
                    };
                });
                setLiveDevices(formatted);
            } catch (err) {
                console.error("Error fetching devices:", err);
            }
        };
        fetchDevices();
        const intv = setInterval(fetchDevices, 10000);
        return () => clearInterval(intv);
    }, [user]);

    // Derived KPI
    const totalDevices = liveDevices.length;
    const onlineDevices = liveDevices.filter(d => d.status.toLowerCase() !== 'offline').length;
    const offlineDevices = liveDevices.filter(d => d.status.toLowerCase() === 'offline').length;
    const unassignedDevices = liveDevices.filter(d => d.animal === 'Unassigned').length;
    const lowBatteryDevices = liveDevices.filter(d => d.battery < 20).length;

    // Filter Logic
    const filteredDevices = liveDevices.filter(d => {
        const matchesSearch = d.id.toLowerCase().includes(searchTerm.toLowerCase()) || d.animal.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesStatus = filterStatus === 'All' ||
            (filterStatus === 'Online' && d.status === 'Online') ||
            (filterStatus === 'Offline' && d.status === 'Offline') ||
            (filterStatus === 'Unassigned' && d.animal === 'Unassigned') ||
            (filterStatus === 'Low Battery' && d.battery <= 20);
        return matchesSearch && matchesStatus;
    });

    const renderBattery = (level) => {
        let fillClass = 'high';
        if (level < 20) fillClass = 'low';
        else if (level <= 50) fillClass = 'medium';

        return (
            <div className="battery-cell">
                <div className="battery-visual">
                    <div className={`battery-fill ${fillClass}`} style={{ width: `${level}%` }}></div>
                </div>
                <span className={`battery-text ${fillClass === 'low' ? 'low' : ''}`}>{level}%</span>
            </div>
        );
    };

    const renderSignal = (strength) => {
        let bars = 4;
        let activeClass = 'active';
        if (strength === 'No Signal') { bars = 0; }
        else if (strength === 'Weak') { bars = 1; activeClass = 'active weak'; }
        else if (strength === 'Medium') { bars = 2; activeClass = 'active medium'; }
        else if (strength === 'Strong') { bars = 4; }

        return (
            <div className="signal-cell" title={strength}>
                {[...Array(4)].map((_, i) => (
                    <div key={i} className={`signal-bar ${i < bars ? activeClass : ''}`} style={{ height: `${(i + 1) * 3 + 2}px` }}></div>
                ))}
            </div>
        );
    };

    const renderDevicePanel = () => {
        if (!activeDevice) return null;
        return (
            <div className="device-detail-panel">
                <div className="dd-header">
                    <button className="dd-close" onClick={() => setActiveDevice(null)}><X size={20} /></button>
                    <h2 className="dd-title">Device ID: {activeDevice.id}</h2>
                    <div className="dd-status-badges">
                        <span className="dd-badge" style={{ color: activeDevice.status === 'Online' ? '#10b981' : (activeDevice.status === 'Offline' ? '#ef4444' : '#f59e0b') }}>
                            {activeDevice.status}
                        </span>
                        <span className="dd-badge" style={{ color: '#64748b' }}>{activeDevice.type}</span>
                    </div>
                </div>

                <div className="dd-content">
                    <div className="dd-section">
                        <h4>Assignment Info</h4>
                        <div className="dd-grid">
                            <div className="dd-item full">
                                <label>Assigned Animal</label>
                                <div className="val" style={{ color: '#3b82f6', cursor: 'pointer' }} onClick={() => activeDevice.animal !== 'Unassigned' && navigate(`/livestock/${activeDevice.animal}`)}>
                                    {activeDevice.animal} {activeDevice.animal !== 'Unassigned' && <LinkIcon size={12} />}
                                </div>
                            </div>
                            <div className="dd-item">
                                <label>Assigned Location</label>
                                <div className="val">{activeDevice.location}</div>
                            </div>
                            <div className="dd-item">
                                <label>Assigned Date</label>
                                <div className="val">{activeDevice.animal !== 'Unassigned' ? 'Jan 12 2026' : '--'}</div>
                            </div>
                        </div>
                    </div>

                    <div className="dd-section">
                        <h4>Telemetry Status</h4>
                        <div className="dd-grid">
                            <div className="dd-item">
                                <label>Last Sync</label>
                                <div className="val">{activeDevice.lastSync}</div>
                            </div>
                            <div className="dd-item">
                                <label>Last Data Received</label>
                                <div className="val">{activeDevice.lastSync}</div>
                            </div>
                            <div className="dd-item full">
                                <label>Firmware Version</label>
                                <div className="val">{activeDevice.firmware}</div>
                            </div>
                        </div>
                    </div>

                    <div className="dd-section">
                        <h4>Device Health</h4>
                        <div className="dd-grid">
                            <div className="dd-item">
                                <label>Connectivity</label>
                                <div className="val">{activeDevice.signal === 'No Signal' ? 'Disconnected' : 'Stable'}</div>
                            </div>
                            <div className="dd-item">
                                <label>Battery Level</label>
                                <div className="val">{renderBattery(activeDevice.battery)}</div>
                            </div>
                            <div className="dd-item full">
                                <label>General Health</label>
                                <div className="val">{activeDevice.health}</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="dd-actions">
                    <button className="mgt-btn primary">Reassign Device</button>
                    {activeDevice.animal !== 'Unassigned' && (
                        <button className="mgt-btn" onClick={() => navigate(`/livestock/${activeDevice.animal}`)}>View Animal Profile</button>
                    )}
                    <button className="mgt-btn">Mark for Maintenance</button>
                </div>
            </div>
        );
    };

    const renderAddModal = () => {
        if (!showAddModal) return null;
        return (
            <div className="mgt-modal-overlay" onClick={() => setShowAddModal(false)}>
                <div className="mgt-modal-content" onClick={e => e.stopPropagation()}>
                    <div className="modal-header">
                        <h2>Register New Device</h2>
                        <button className="icon-btn-premium close-btn" onClick={() => setShowAddModal(false)}><X size={20} /></button>
                    </div>
                    <div className="modal-body">
                        <div className="form-section">
                            <h4>Device Information</h4>
                            <div className="form-grid">
                                <div className="input-group">
                                    <label>Device ID *</label>
                                    <input type="text" placeholder="e.g. DEV100" />
                                </div>
                                <div className="input-group">
                                    <label>Device Type *</label>
                                    <select><option>Neck Collar</option><option>Ear Tag</option></select>
                                </div>
                                <div className="input-group">
                                    <label>Firmware Version</label>
                                    <input type="text" placeholder="e.g. v1.2.0" />
                                </div>
                                <div className="input-group">
                                    <label>Purchase Date</label>
                                    <input type="text" placeholder="Jan 2026" />
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="modal-footer">
                        <button className="mgt-btn" onClick={() => setShowAddModal(false)}>Cancel</button>
                        <button className="mgt-btn primary">Save Device</button>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="devices-layout">
            {/* SIDEBAR Reused */}
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
                            <List className="nav-icon" /> <span>Farm & Livestock</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/map')}>
                            <MapIcon className="nav-icon" /> <span>Map Intelligence</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/health-analytics')}>
                            <Activity className="nav-icon" /> <span>Health Analytics</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/alerts')}>
                            <AlertTriangle className="nav-icon" /> <span>Alerts Center</span>
                        </div>

                        <div className="sidebar-section-title">Operations</div>
                        <div className="nav-item-premium active" onClick={() => navigate('/devices')}>
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
                    <div className="nav-item-premium" onClick={() => ({})}>
                        <User className="nav-icon" /> <span>Profile</span>
                    </div>
                    <div className="nav-item-premium logout" onClick={() => navigate('/')}>
                        <LogOut className="nav-icon" /> <span>Logout</span>
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT */}
            <main className="devices-main">
                {/* TOP BAR */}
                <header className="mgt-header">
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div className="mgt-title-area">
                            <h1>Devices</h1>
                            <p>Manage livestock monitoring devices and connectivity</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        {user?.type !== 'staff' && (
                            <>
                                <button className="mgt-btn"><Download size={16} /> Export Data</button>
                                <button className="mgt-btn"><LinkIcon size={16} /> Batch Assign</button>
                                <button className="mgt-btn primary" onClick={() => setShowAddModal(true)}><Plus size={16} /> Add Device</button>
                            </>
                        )}
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                {/* DEVICE KPIs */}
                <div className="devices-kpi-grid">
                    <div className="device-kpi-card">
                        <span className="dkpi-label">Total Devices</span>
                        <span className="dkpi-value">{totalDevices}</span>
                    </div>
                    <div className="device-kpi-card">
                        <span className="dkpi-label">Online Devices</span>
                        <span className="dkpi-value green">{onlineDevices}</span>
                    </div>
                    <div className="device-kpi-card">
                        <span className="dkpi-label">Offline Devices</span>
                        <span className="dkpi-value red">{offlineDevices}</span>
                    </div>
                    <div className="device-kpi-card">
                        <span className="dkpi-label">Unassigned</span>
                        <span className="dkpi-value blue">{unassignedDevices}</span>
                    </div>
                    <div className="device-kpi-card">
                        <span className="dkpi-label">Low Battery</span>
                        <span className="dkpi-value orange">{lowBatteryDevices}</span>
                    </div>
                </div>

                {/* TABLE SECTION */}
                <div className="device-content-wrapper">
                    <div className="devices-table-card">
                        <div className="mgt-table-header">
                            <div className="mgt-search">
                                <Search size={16} color="#94a3b8" />
                                <input
                                    type="text"
                                    placeholder="Search Device ID or Animal..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>
                            <div className="mgt-filters">
                                <select className="mgt-filter-select" value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}>
                                    <option value="All">All Status</option>
                                    <option value="Online">Online</option>
                                    <option value="Offline">Offline</option>
                                    <option value="Unassigned">Unassigned</option>
                                    <option value="Low Battery">Low Battery</option>
                                </select>
                                <select className="mgt-filter-select">
                                    <option>All Types</option>
                                    <option>Neck Collar</option>
                                    <option>Ear Tag</option>
                                </select>
                            </div>
                        </div>

                        <div className="mgt-table-wrapper">
                            <table className="mgt-data-table">
                                <thead>
                                    <tr>
                                        <th>Device ID</th>
                                        <th>Assigned Animal</th>
                                        <th>Type</th>
                                        <th>Status</th>
                                        <th>Battery Level</th>
                                        <th>Signal</th>
                                        <th>Last Sync</th>
                                        <th>Location</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredDevices.map(d => (
                                        <tr key={d.id} onClick={() => setActiveDevice(d)} style={{ cursor: 'pointer' }}>
                                            <td className="cell-id"><span className="cell-device"><Cpu size={14} color="#64748b" /> {d.id}</span></td>
                                            <td style={{ fontWeight: d.animal !== 'Unassigned' ? 700 : 500, color: d.animal !== 'Unassigned' ? '#0f172a' : '#94a3b8' }}>{d.animal}</td>
                                            <td className="cell-sub">{d.type}</td>
                                            <td className={`device-status ${d.status.toLowerCase().replace(' ', '-')}`}>{d.status}</td>
                                            <td>{renderBattery(d.battery)}</td>
                                            <td>{renderSignal(d.signal)}</td>
                                            <td style={{ fontWeight: 600, color: '#64748b' }}>{d.lastSync}</td>
                                            <td className="cell-sub">{d.location}</td>
                                            <td>
                                                <div className="row-actions" onClick={e => e.stopPropagation()}>
                                                    <button className="action-icon" onClick={() => setActiveDevice(d)}><Eye size={16} /></button>
                                                    <button className="action-icon" title="Edit/Assign"><LinkIcon size={16} /></button>
                                                </div>
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                {/* Overlays */}
                {renderDevicePanel()}
                {renderAddModal()}

            </main>
        </div>
    );
}

export default Devices;
