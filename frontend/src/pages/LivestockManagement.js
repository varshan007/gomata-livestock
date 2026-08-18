import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Map as MapIcon, Activity, AlertTriangle, Cpu, Layers, Globe, Users, Settings, LogOut, ChevronDown, User, Menu,
    Search, Plus, Download, Upload, Edit, Eye, Trash2, Battery, Filter, CheckCircle2, ChevronRight, Link as LinkIcon, X
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css';
import './Dashboard.css';

// Mock Data removed in Phase 9 - Hydrated by real REST APIs

function LivestockManagement() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [activeTab, setActiveTab] = useState('registry'); // registry | structure | devices
    const [searchTerm, setSearchTerm] = useState('');
    const [activeModal, setActiveModal] = useState(null); // 'register', 'zone', 'assign', or null
    const profileRef = useRef();

    // Data States
    const [livestock, setLivestock] = useState([]);
    const [farms, setFarms] = useState([]);
    const [devices, setDevices] = useState([]);
    const [stats, setStats] = useState({ total: 0, active: 0, offline: 0, zones: 0 });

    useEffect(() => {
        const fetchManagementData = async () => {
            if (!user) return;
            const token = localStorage.getItem('token');
            if (!token) return;
            try {
                const headers = { Authorization: `Bearer ${token}` };

                const [lsRes, fRes, dRes] = await Promise.all([
                    fetch(`${process.env.REACT_APP_API_URL || 'https://gomata-backend.onrender.com/api'}/livestock`, { headers }),
                    fetch(`${process.env.REACT_APP_API_URL || 'https://gomata-backend.onrender.com/api'}/farms`, { headers }),
                    fetch(`${process.env.REACT_APP_API_URL || 'https://gomata-backend.onrender.com/api'}/devices`, { headers })
                ]);

                const lsJson = await lsRes.json();
                const fJson = await fRes.json();
                const dJson = await dRes.json();

                const lsData = lsJson.data || lsJson;
                const fData = fJson.data || fJson;
                const dData = dJson.data || dJson;

                // Format Livestock correctly for the table
                const formattedLivestock = (Array.isArray(lsData) ? lsData : []).map(l => ({
                    id: l.livestock._id,
                    tag: l.livestock.tagNumber,
                    breed: l.livestock.breed,
                    age: 'N/A',
                    weight: 'N/A',
                    device: (Array.isArray(dData) ? dData : []).find(d => d.animal?.includes(l.livestock._id))?.id || 'Unassigned',
                    zone: l.livestock.location,
                    added: l.latestSensorData.timestamp ? new Date(l.latestSensorData.timestamp).toLocaleDateString() : 'N/A',
                    lastData: l.latestSensorData.timestamp ? new Date(l.latestSensorData.timestamp).toLocaleTimeString() : '--',
                    status: l.latestSensorData.battery < 20 ? 'offline' : 'active'
                }));

                // Format Farms correctly for the table
                const formattedFarms = [];
                let totalZones = 0;
                fData.forEach(f => {
                    if (f.zones && f.zones.length > 0) {
                        f.zones.forEach(z => {
                            totalZones++;

                            let coordStr = '--';
                            if (z.geofence?.type === 'Point') {
                                coordStr = `${z.geofence.coordinates[1].toFixed(4)}, ${z.geofence.coordinates[0].toFixed(4)}`;
                            } else if (z.geofence?.type === 'Polygon' && z.geofence.coordinates && z.geofence.coordinates[0]) {
                                coordStr = `${z.geofence.coordinates[0][0][1].toFixed(4)}, ${z.geofence.coordinates[0][0][0].toFixed(4)}`;
                            }

                            formattedFarms.push({
                                name: f.name,
                                zone: z.name,
                                coords: coordStr,
                                count: lsData.filter(l => l.livestock.zoneId === z._id).length,
                                created: new Date(z.createdAt || f.createdAt).toLocaleDateString()
                            });
                        });
                    } else {
                        formattedFarms.push({
                            name: f.name,
                            zone: 'Unassigned',
                            coords: '--',
                            count: 0,
                            created: new Date(f.createdAt).toLocaleDateString()
                        });
                    }
                });

                setLivestock(formattedLivestock);
                setFarms(formattedFarms);
                setDevices(dData);

                setStats({
                    total: formattedLivestock.length,
                    active: dData.filter(d => d.status.toLowerCase() !== 'offline').length,
                    offline: dData.filter(d => d.status.toLowerCase() === 'offline').length,
                    zones: totalZones
                });

            } catch (err) {
                console.error("Error fetching Management Data:", err);
            }
        };

        fetchManagementData();
    }, [user]);

    useEffect(() => {
        const listener = (event) => {
            if (!profileRef.current || profileRef.current.contains(event.target)) return;
            if (showProfileMenu) setShowProfileMenu(false);
        };
        document.addEventListener("click", listener);
        return () => document.removeEventListener("click", listener);
    }, [showProfileMenu]);

    const renderRegistryTable = () => (
        <table className="mgt-data-table">
            <thead>
                <tr>
                    <th>Animal ID / Tag</th>
                    <th>Breed / Details</th>
                    <th>Assigned Device</th>
                    <th>Zone</th>
                    <th>Last Data</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {livestock.filter(a => a.id.toLowerCase().includes(searchTerm.toLowerCase())).map((animal, idx) => (
                    <tr key={idx}>
                        <td>
                            <div className="cell-id">{animal.id}</div>
                            <div className="cell-sub">{animal.tag}</div>
                        </td>
                        <td>
                            <div style={{ fontWeight: 600, color: '#334155' }}>{animal.breed}</div>
                            <div className="cell-sub">{animal.age} • {animal.weight}</div>
                        </td>
                        <td>
                            <span className="cell-device"><Cpu size={14} /> {animal.device}</span>
                        </td>
                        <td>{animal.zone}</td>
                        <td style={{ fontWeight: 600, color: '#64748b' }}>{animal.lastData}</td>
                        <td>
                            <span className={`status-badge ${animal.status}`}>
                                <div className="status-dot"></div> {animal.status}
                            </span>
                        </td>
                        <td>
                            <div className="row-actions">
                                <button className="action-icon" title="View Details" onClick={() => navigate(`/livestock/${animal.id}`)}><Eye size={16} /></button>
                                {user?.type !== 'staff' && <button className="action-icon" title="Edit"><Edit size={16} /></button>}
                            </div>
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );

    const renderFarmTable = () => (
        <table className="mgt-data-table">
            <thead>
                <tr>
                    <th>Farm Name</th>
                    <th>Zone / Barn</th>
                    <th>Coordinates</th>
                    <th>Livestock Count</th>
                    <th>Date Configured</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {farms.map((farm, idx) => (
                    <tr key={idx}>
                        <td className="cell-id">{farm.name}</td>
                        <td style={{ fontWeight: 600 }}>{farm.zone}</td>
                        <td className="cell-sub">{farm.coords}</td>
                        <td className="cell-id">{farm.count}</td>
                        <td style={{ color: '#64748b' }}>{farm.created}</td>
                        <td>
                            {user?.type !== 'staff' && <button className="action-icon" title="Edit Zone"><Edit size={16} /></button>}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );

    const renderDeviceTable = () => (
        <table className="mgt-data-table">
            <thead>
                <tr>
                    <th>Device ID</th>
                    <th>Assigned Animal</th>
                    <th>Type</th>
                    <th>Battery</th>
                    <th>Last Sync</th>
                    <th>Status</th>
                    <th>Actions</th>
                </tr>
            </thead>
            <tbody>
                {devices.map((dev, idx) => (
                    <tr key={idx}>
                        <td>
                            <span className="cell-device"><Cpu size={14} /> {dev.id}</span>
                        </td>
                        <td className={dev.animal === 'Unassigned' ? 'cell-sub' : 'cell-id'}>{dev.animal}</td>
                        <td style={{ color: '#64748b' }}>{dev.type}</td>
                        <td>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', fontWeight: 600 }}>
                                <Battery size={14} color={dev.battery === '5%' ? '#ef4444' : '#10b981'} /> {dev.battery}
                            </div>
                        </td>
                        <td style={{ fontWeight: 600, color: '#64748b' }}>{dev.lastSync}</td>
                        <td>
                            <span style={{
                                fontWeight: 600,
                                color: (dev.status === 'Online' || dev.status === 'Active') ? '#10b981' : (dev.status === 'Ready' ? '#3b82f6' : '#ef4444')
                            }}>{dev.status}</span>
                        </td>
                        <td>
                            {user?.type !== 'staff' && <button className="action-icon" title={dev.animal === 'Unassigned' ? 'Assign Device' : 'Reassign'}><LinkIcon size={16} /></button>}
                        </td>
                    </tr>
                ))}
            </tbody>
        </table>
    );

    // Modal Renders
    const renderRegisterModal = () => (
        <div className="mgt-modal-overlay" onClick={() => setActiveModal(null)}>
            <div className="mgt-modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>Register New Animal</h2>
                    <button className="icon-btn-premium close-btn" onClick={() => setActiveModal(null)}><X size={20} /></button>
                </div>
                <div className="modal-body">
                    <div className="form-section">
                        <h4>A. Basic Information</h4>
                        <div className="form-grid">
                            <div className="input-group">
                                <label>Animal ID *</label>
                                <input type="text" placeholder="e.g. MIX001" />
                            </div>
                            <div className="input-group">
                                <label>Tag ID *</label>
                                <input type="text" placeholder="e.g. TAG1023" />
                            </div>
                            <div className="input-group">
                                <label>Species *</label>
                                <select><option>Cow</option><option>Buffalo</option><option>Goat</option><option>Sheep</option><option>Other</option></select>
                            </div>
                            <div className="input-group">
                                <label>Breed *</label>
                                <select><option>Holstein</option><option>Jersey</option><option>Gir</option><option>HF Cross</option></select>
                            </div>
                            <div className="input-group">
                                <label>Gender *</label>
                                <select><option>Female</option><option>Male</option></select>
                            </div>
                            <div className="input-group">
                                <label>Age or DOB *</label>
                                <input type="text" placeholder="e.g. 4 years" />
                            </div>
                        </div>
                    </div>

                    <div className="form-section">
                        <h4>B. Farm Assignment</h4>
                        <div className="form-grid">
                            <div className="input-group">
                                <label>Assign Farm *</label>
                                <select><option>Main Farm</option><option>Farm 2</option></select>
                            </div>
                            <div className="input-group">
                                <label>Assign Zone / Barn *</label>
                                <select><option>Barn A</option><option>Barn B</option><option>Grazing Area</option></select>
                            </div>
                        </div>
                    </div>

                    <div className="form-section">
                        <h4>C. Device Assignment (Optional)</h4>
                        <div className="input-group full-width">
                            <label>Assign Device ID</label>
                            <select><option>Select unassigned device...</option><option>DEV031</option><option>DEV040</option></select>
                        </div>
                    </div>

                    <div className="form-section">
                        <h4>D. Additional Info (Optional)</h4>
                        <div className="form-grid">
                            <div className="input-group">
                                <label>Weight (kg)</label>
                                <input type="number" placeholder="520" />
                            </div>
                            <div className="input-group">
                                <label>Notes</label>
                                <input type="text" placeholder="e.g. Vaccinated Jan 2026" />
                            </div>
                        </div>
                    </div>
                </div>
                <div className="modal-footer">
                    <button className="mgt-btn" onClick={() => setActiveModal(null)}>Cancel</button>
                    <button className="mgt-btn primary">Register Animal</button>
                </div>
            </div>
        </div>
    );

    const renderZoneModal = () => (
        <div className="mgt-modal-overlay" onClick={() => setActiveModal(null)}>
            <div className="mgt-modal-content" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>Configure New Zone</h2>
                    <button className="icon-btn-premium close-btn" onClick={() => setActiveModal(null)}><X size={20} /></button>
                </div>
                <div className="modal-body">
                    <div className="form-section">
                        <h4>A. Zone Basic Information</h4>
                        <div className="form-grid">
                            <div className="input-group full-width">
                                <label>Zone Name *</label>
                                <input type="text" placeholder="e.g. Barn A" />
                            </div>
                            <div className="input-group">
                                <label>Farm *</label>
                                <select><option>Main Farm</option><option>Farm 2</option></select>
                            </div>
                            <div className="input-group">
                                <label>Zone Type *</label>
                                <select><option>Barn</option><option>Grazing Area</option><option>Quarantine</option><option>Milking Area</option></select>
                            </div>
                        </div>
                    </div>

                    <div className="form-section">
                        <h4>B. Location boundaries</h4>
                        <div className="map-placeholder">
                            <MapIcon size={48} color="#cbd5e1" />
                            <p>Draw zone boundary on map</p>
                            <button className="mgt-btn" style={{ marginTop: '12px' }}>Open Map Editor</button>
                        </div>
                    </div>

                    <div className="form-section">
                        <h4>C. Optional Settings</h4>
                        <div className="form-grid">
                            <div className="input-group">
                                <label>Max Capacity</label>
                                <input type="number" placeholder="e.g. 50" />
                            </div>
                            <div className="input-group">
                                <label>Notes</label>
                                <input type="text" placeholder="e.g. Used for sick animals" />
                            </div>
                        </div>
                    </div>
                </div>
                <div className="modal-footer">
                    <button className="mgt-btn" onClick={() => setActiveModal(null)}>Cancel</button>
                    <button className="mgt-btn primary">Create Zone</button>
                </div>
            </div>
        </div>
    );

    const renderBatchAssignModal = () => (
        <div className="mgt-modal-overlay" onClick={() => setActiveModal(null)}>
            <div className="mgt-modal-content large" onClick={e => e.stopPropagation()}>
                <div className="modal-header">
                    <h2>Batch Assign Devices</h2>
                    <button className="icon-btn-premium close-btn" onClick={() => setActiveModal(null)}><X size={20} /></button>
                </div>
                <div className="modal-body">
                    <div className="wizard-split">
                        <div className="wizard-panel">
                            <h4>Step 1: Select Animals</h4>
                            <p className="wizard-desc">Choose which animals need sensors.</p>
                            <div className="wizard-list">
                                <label className="wizard-item"><input type="checkbox" defaultChecked /> <span>MIX001 (Holstein) - Barn A</span></label>
                                <label className="wizard-item"><input type="checkbox" defaultChecked /> <span>MIX002 (Jersey) - Barn B</span></label>
                                <label className="wizard-item"><input type="checkbox" defaultChecked /> <span>MIX003 (Angus) - Grazing</span></label>
                                <label className="wizard-item"><input type="checkbox" /> <span>MIX004 (Holstein) - Barn A</span></label>
                            </div>
                        </div>
                        <div className="wizard-panel">
                            <h4>Step 2: Select Devices</h4>
                            <p className="wizard-desc">Choose available devices to assign.</p>
                            <div className="wizard-list">
                                <label className="wizard-item"><input type="checkbox" defaultChecked /> <span>DEV021 (Ready, 100%)</span></label>
                                <label className="wizard-item"><input type="checkbox" defaultChecked /> <span>DEV022 (Ready, 98%)</span></label>
                                <label className="wizard-item"><input type="checkbox" defaultChecked /> <span>DEV023 (Ready, 100%)</span></label>
                                <label className="wizard-item"><input type="checkbox" /> <span>DEV024 (Ready, 85%)</span></label>
                            </div>
                        </div>
                    </div>

                    <div className="wizard-summary">
                        <CheckCircle2 size={24} color="#10b981" />
                        <div className="summary-texts">
                            <strong>Auto-mapping Ready</strong>
                            <p>3 animals selected will be mapped to 3 available devices.</p>
                        </div>
                    </div>
                </div>
                <div className="modal-footer">
                    <button className="mgt-btn" onClick={() => setActiveModal(null)}>Cancel</button>
                    <button className="mgt-btn success">Complete Assignment</button>
                </div>
            </div>
        </div>
    );

    return (
        <div className="management-layout">
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
                        <div className="nav-item-premium active" onClick={() => navigate('/livestock')}>
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
                    <div className="nav-item-premium" onClick={() => ({})}>
                        <User className="nav-icon" /> <span>Profile</span>
                    </div>
                    <div className="nav-item-premium logout" onClick={() => navigate('/')}>
                        <LogOut className="nav-icon" /> <span>Logout</span>
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT */}
            <main className="management-main">
                {/* TOP BAR */}
                <header className="topbar-premium" style={{ marginBottom: '16px', padding: '32px 40px 16px 40px' }}>
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div className="mgt-title-area">
                            <h1>Farm & Livestock Management</h1>
                            <p>Manage farm structure, livestock registry, and device assignments.</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        {user?.type !== 'staff' && (
                            <>
                                <button className="mgt-btn"><Upload size={16} /> Import</button>
                                <button className="mgt-btn"><Download size={16} /> Export</button>
                                <button className="mgt-btn primary" onClick={() => setActiveModal('register')}><Plus size={16} /> Add Livestock</button>
                            </>
                        )}
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                {/* TABS */}
                <div className="mgt-tabs-container">
                    <div className="mgt-tabs">
                        <div className={`mgt-tab ${activeTab === 'registry' ? 'active' : ''}`} onClick={() => setActiveTab('registry')}>
                            <List size={16} /> Livestock Registry
                        </div>
                        <div className={`mgt-tab ${activeTab === 'structure' ? 'active' : ''}`} onClick={() => setActiveTab('structure')}>
                            <Globe size={16} /> Farm Structure
                        </div>
                        <div className={`mgt-tab ${activeTab === 'devices' ? 'active' : ''}`} onClick={() => setActiveTab('devices')}>
                            <Cpu size={16} /> Device Assignment
                        </div>
                    </div>
                </div>

                {/* CONTENT SPLIT */}
                <div className="mgt-content-split">

                    {/* LEFT COLUMN: Data Table Card */}
                    {loading && <div className="loading-bar"></div>}
            
                    <div style={{ background: 'red', color: 'white', padding: '10px', textAlign: 'center', zIndex: 9999, fontWeight: 'bold' }}>
                        DEBUG INFO: API URL = {process.env.REACT_APP_API_URL || 'Not Set'} | Animals = {livestock ? livestock.length : 'null'} | Status = {livestock ? 'Loaded' : 'Waiting'}
                    </div>

                    <div className="mgt-table-card">
                        <div className="mgt-table-header">
                            <div className="mgt-search">
                                <Search size={16} color="#94a3b8" />
                                <input
                                    type="text"
                                    placeholder="Search by ID, Tag, or Breed..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>
                            <div className="mgt-filters">
                                <select className="mgt-filter-select">
                                    <option>All Zones</option>
                                    <option>Barn A</option>
                                    <option>Barn B</option>
                                </select>
                                <select className="mgt-filter-select">
                                    <option>Filter Status</option>
                                    <option>Active</option>
                                    <option>Inactive</option>
                                </select>
                            </div>
                        </div>

                        <div className="mgt-table-wrapper">
                            {activeTab === 'registry' && renderRegistryTable()}
                            {activeTab === 'structure' && renderFarmTable()}
                            {activeTab === 'devices' && renderDeviceTable()}
                        </div>
                    </div>

                    {/* RIGHT COLUMN: Quick Summary Panel */}
                    <div className="mgt-summary-panel">
                        <div className="summary-gradient-card">
                            <h3>Overview Summary</h3>
                            <div className="stat-row">
                                <span className="stat-label">Total Livestock</span>
                                <span className="stat-val">{stats.total}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Active Devices</span>
                                <span className="stat-val val-green">{stats.active}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Offline Devices</span>
                                <span className="stat-val val-red">{stats.offline}</span>
                            </div>
                            <div className="stat-row">
                                <span className="stat-label">Configured Zones</span>
                                <span className="stat-val">{stats.zones}</span>
                            </div>
                        </div>

                        {user?.type !== 'staff' && (
                            <div className="summary-gradient-card">
                                <h3>Quick Actions</h3>
                                <div className="quick-actions-list">
                                    <button className="mgt-btn" style={{ width: '100%', justifyContent: 'flex-start' }} onClick={() => setActiveModal('register')}>
                                        <Plus size={16} /> Register Animal
                                    </button>
                                    <button className="mgt-btn" style={{ width: '100%', justifyContent: 'flex-start' }} onClick={() => setActiveModal('zone')}>
                                        <Globe size={16} /> Configure New Zone
                                    </button>
                                    <button className="mgt-btn" style={{ width: '100%', justifyContent: 'flex-start' }} onClick={() => setActiveModal('assign')}>
                                        <LinkIcon size={16} /> Batch Assign Devices
                                    </button>
                                </div>
                            </div>
                        )}
                    </div>

                </div>

                {/* Modals placed at bottom of page but fixed to viewport via CSS */}
                {activeModal === 'register' && renderRegisterModal()}
                {activeModal === 'zone' && renderZoneModal()}
                {activeModal === 'assign' && renderBatchAssignModal()}

            </main>
        </div>
    );
}

export default LivestockManagement;
