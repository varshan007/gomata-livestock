import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Map as MapIcon, Activity, AlertTriangle, Cpu, Layers, Globe, Users, Settings, LogOut, ChevronDown, User, Menu,
    Search, Plus, Download, Upload, Edit, Eye, Trash2, Battery, Filter, CheckCircle2, ChevronRight, Link as LinkIcon, X, MapPin
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css'; // Reuse core premium styles
import './FarmDetails.css';

// Mock Data Replaced with Live Endpoints in Phase 9

function FarmDetails() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterType, setFilterType] = useState('All');
    const [activeZone, setActiveZone] = useState(null); // Detailed side panel
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

    const [liveLocations, setLiveLocations] = useState([]);
    const [stats, setStats] = useState({ parentFarms: 0, activeZones: 0, totalArea: 0, quarantine: 0, totalHoused: 0 });

    useEffect(() => {
        const fetchFarms = async () => {
            if (!user) return;
            const token = localStorage.getItem('token');
            if (!token) return;
            try {
                const headers = { Authorization: `Bearer ${token}` };
                const [fRes, lsRes] = await Promise.all([
                    fetch('http://localhost:8000/api/farms', { headers }),
                    fetch('http://localhost:8000/api/livestock', { headers })
                ]);
                const fJson = await fRes.json();
                const lsJson = await lsRes.json();

                const fData = fJson.data || fJson;
                const lsData = lsJson.data || lsJson;

                const formatted = [];
                let totArea = 0;
                let qCount = 0;

                (Array.isArray(fData) ? fData : []).forEach(f => {
                    if (f.zones) {
                        f.zones.forEach(z => {
                            const animalsInZone = (Array.isArray(lsData) ? lsData : []).filter(l => l.livestock?.zoneId === z._id).length;
                            totArea += (z.areaSize || 0);
                            const typeStr = z.locationType === 'Point' ? 'Circular Mapping' : 'Polygon Region';
                            if (z.name.toLowerCase().includes('quarantine')) qCount++;

                            let coordStr = '--';
                            if (z.geofence?.type === 'Point') {
                                coordStr = `${z.geofence.coordinates[1]?.toFixed(4)}, ${z.geofence.coordinates[0]?.toFixed(4)}`;
                            } else if (z.geofence?.type === 'Polygon' && z.geofence.coordinates && z.geofence.coordinates[0]) {
                                coordStr = `${z.geofence.coordinates[0][0][1]?.toFixed(4)}, ${z.geofence.coordinates[0][0][0]?.toFixed(4)}`;
                            }

                            formatted.push({
                                id: z._id.substring(0, 8).toUpperCase(),
                                farm: f.name,
                                type: typeStr,
                                name: z.name,
                                animals: animalsInZone,
                                capacity: 150, // Mock fixed capacity
                                area: `${z.areaSize || 0} Ac`,
                                coordinates: coordStr,
                                manager: user.name || 'System Admin',
                                status: 'Active'
                            });
                        });
                    }
                });

                setLiveLocations(formatted);
                setStats({
                    parentFarms: fData.length,
                    activeZones: formatted.length,
                    totalArea: totArea.toFixed(1),
                    quarantine: qCount,
                    totalHoused: (Array.isArray(lsData) ? lsData : []).length
                });

            } catch (err) {
                console.error("Error fetching farms:", err);
            }
        };
        fetchFarms();
    }, [user]);

    // Filter Logic
    const filteredLocations = liveLocations.filter(l => {
        const matchesSearch = l.name.toLowerCase().includes(searchTerm.toLowerCase()) || l.farm.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesType = filterType === 'All' || l.type.includes(filterType);
        return matchesSearch && matchesType;
    });

    const renderOccupancy = (current, max) => {
        const percentage = Math.round((current / max) * 100);
        let fillClass = 'optimal';
        if (percentage < 30) fillClass = 'low';
        else if (percentage > 90) fillClass = 'full';

        return (
            <div className="occupancy-cell" title={`${percentage}% Full`}>
                <div className="occ-visual">
                    <div className={`occ-fill ${fillClass}`} style={{ width: `${percentage}%` }}></div>
                </div>
                <span className={`occ-text ${fillClass === 'full' ? 'full' : ''}`}>{current} / {max}</span>
            </div>
        );
    };

    const renderZonePanel = () => {
        if (!activeZone) return null;
        return (
            <div className="farm-detail-panel">
                <div className="fd-header">
                    <button className="fd-close" onClick={() => setActiveZone(null)}><X size={20} /></button>
                    <h2 className="fd-title">{activeZone.name}</h2>
                    <div className="dd-status-badges">
                        <span className="dd-badge" style={{ color: activeZone.status === 'Active' ? '#10b981' : (activeZone.status === 'Inactive' ? '#94a3b8' : '#f59e0b') }}>
                            {activeZone.status}
                        </span>
                        <span className="dd-badge" style={{ color: '#3b82f6' }}>{activeZone.type}</span>
                    </div>
                </div>

                <div className="fd-content">
                    <div className="map-viz-container">
                        <MapPin size={40} color="#059669" style={{ marginBottom: '8px' }} />
                        <span style={{ fontWeight: 600, color: '#047857' }}>Map Visualization Area</span>
                        <span style={{ fontSize: '0.8rem', color: '#10b981' }}>Coordinates: {activeZone.coordinates}</span>
                    </div>

                    <div className="fd-section">
                        <h4>Location Info</h4>
                        <div className="fd-grid">
                            <div className="fd-item full">
                                <label>Parent Farm</label>
                                <div className="val">{activeZone.farm}</div>
                            </div>
                            <div className="fd-item">
                                <label>Zone ID</label>
                                <div className="val">{activeZone.id}</div>
                            </div>
                            <div className="fd-item">
                                <label>Total Area</label>
                                <div className="val">{activeZone.area}</div>
                            </div>
                        </div>
                    </div>

                    <div className="fd-section">
                        <h4>Capacity & Metrics</h4>
                        <div className="fd-grid">
                            <div className="fd-item">
                                <label>Current Livestock</label>
                                <div className="val" style={{ color: '#3b82f6', cursor: 'pointer' }} onClick={() => navigate('/livestock')}>
                                    {activeZone.animals} Head <LinkIcon size={12} />
                                </div>
                            </div>
                            <div className="fd-item">
                                <label>Max Capacity</label>
                                <div className="val">{activeZone.capacity} Head</div>
                            </div>
                            <div className="fd-item full">
                                <label>Occupancy Rate</label>
                                <div className="val">{renderOccupancy(activeZone.animals, activeZone.capacity)}</div>
                            </div>
                        </div>
                    </div>

                    <div className="fd-section">
                        <h4>Management</h4>
                        <div className="fd-grid">
                            <div className="fd-item full">
                                <label>Zone Manager</label>
                                <div className="val">{activeZone.manager}</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="fd-actions">
                    {user?.type !== 'staff' && <button className="mgt-btn" onClick={() => setActiveZone(null)}>Edit Boundaries</button>}
                    <button className="mgt-btn primary">Manage Livestock</button>
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
                        <h2>Configure New Zone</h2>
                        <button className="icon-btn-premium close-btn" onClick={() => setShowAddModal(false)}><X size={20} /></button>
                    </div>
                    <div className="modal-body">
                        <div className="form-section">
                            <h4>Zone Information</h4>
                            <div className="form-grid">
                                <div className="input-group">
                                    <label>Zone Name *</label>
                                    <input type="text" placeholder="e.g. Barn C" />
                                </div>
                                <div className="input-group">
                                    <label>Zone Type *</label>
                                    <select><option>Barn / Indoor</option><option>Grazing Area</option><option>Quarantine</option><option>Milking Area</option></select>
                                </div>
                                <div className="input-group full-width">
                                    <label>Assign to Farm *</label>
                                    <select><option>Main Farm HQ</option><option>North Pasture</option><option>East Ridge</option><option>+ Create New Farm</option></select>
                                </div>
                                <div className="input-group">
                                    <label>Max Capacity</label>
                                    <input type="number" placeholder="150" />
                                </div>
                                <div className="input-group">
                                    <label>Area Size</label>
                                    <input type="text" placeholder="e.g. 10 Acres" />
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="modal-footer">
                        <button className="mgt-btn" onClick={() => setShowAddModal(false)}>Cancel</button>
                        <button className="mgt-btn primary">Create Zone</button>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="farm-layout">
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
                        <div className="nav-item-premium active" onClick={() => navigate('/farm')}>
                            <Globe className="nav-icon" /> <span>Farms & Locations</span>
                        </div>
                        {user?.type !== 'staff' && (
                            <>
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
            <main className="farm-main">
                {/* TOP BAR */}
                <header className="mgt-header">
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div className="mgt-title-area">
                            <h1>Farms & Locations</h1>
                            <p>Manage agricultural zones, barns, boundaries, and physical infrastructure</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <button className="mgt-btn" onClick={() => navigate('/map')}><MapIcon size={16} /> Map View</button>
                        {user?.type !== 'staff' && <button className="mgt-btn primary" onClick={() => setShowAddModal(true)}><Plus size={16} /> Add Zone</button>}
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                {/* KPI CARDS */}
                <div className="farm-kpi-grid">
                    <div className="farm-kpi-card">
                        <span className="fkpi-label">Total Parent Farms</span>
                        <span className="fkpi-value">{stats.parentFarms}</span>
                    </div>
                    <div className="farm-kpi-card">
                        <span className="fkpi-label">Active Zones</span>
                        <span className="fkpi-value green">{stats.activeZones}</span>
                    </div>
                    <div className="farm-kpi-card">
                        <span className="fkpi-label">Total Area</span>
                        <span className="fkpi-value blue">{stats.totalArea} Ac</span>
                    </div>
                    <div className="farm-kpi-card">
                        <span className="fkpi-label">Quarantine Pens</span>
                        <span className="fkpi-value orange">{stats.quarantine}</span>
                    </div>
                    <div className="farm-kpi-card">
                        <span className="fkpi-label">Total Housed</span>
                        <span className="fkpi-value purple">{stats.totalHoused}</span>
                    </div>
                </div>

                {/* TABLE SECTION */}
                <div className="farm-content-wrapper">
                    <div className="farm-table-card">
                        <div className="mgt-table-header">
                            <div className="mgt-search">
                                <Search size={16} color="#94a3b8" />
                                <input
                                    type="text"
                                    placeholder="Search by Farm or Zone Name..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>
                            <div className="mgt-filters">
                                <select className="mgt-filter-select" value={filterType} onChange={(e) => setFilterType(e.target.value)}>
                                    <option value="All">All Types</option>
                                    <option value="Barn">Barn / Indoor</option>
                                    <option value="Grazing">Grazing Area</option>
                                    <option value="Quarantine">Quarantine</option>
                                    <option value="Milking">Milking Area</option>
                                </select>
                            </div>
                        </div>

                        <div className="mgt-table-wrapper">
                            <table className="mgt-data-table">
                                <thead>
                                    <tr>
                                        <th>Zone Name / ID</th>
                                        <th>Parent Farm</th>
                                        <th>Type</th>
                                        <th>Occupancy</th>
                                        <th>Area Size</th>
                                        <th>Manager</th>
                                        <th>Status</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredLocations.map(l => (
                                        <tr key={l.id} onClick={() => setActiveZone(l)} style={{ cursor: 'pointer' }}>
                                            <td>
                                                <div className="cell-id">{l.name}</div>
                                                <div className="cell-sub">{l.id}</div>
                                            </td>
                                            <td style={{ fontWeight: 600, color: '#334155' }}>{l.farm}</td>
                                            <td><span className="zone-badge">{l.type}</span></td>
                                            <td>{renderOccupancy(l.animals, l.capacity)}</td>
                                            <td className="cell-sub">{l.area}</td>
                                            <td style={{ color: '#475569' }}>{l.manager}</td>
                                            <td style={{
                                                fontWeight: 600,
                                                color: l.status === 'Active' ? '#10b981' : (l.status === 'Inactive' ? '#94a3b8' : '#f59e0b')
                                            }}>{l.status}</td>
                                            <td>
                                                <div className="row-actions" onClick={e => e.stopPropagation()}>
                                                    <button className="action-icon" onClick={() => setActiveZone(l)}><Eye size={16} /></button>
                                                    <button className="action-icon" title="Edit"><Edit size={16} /></button>
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
                {renderZonePanel()}
                {renderAddModal()}

            </main>
        </div>
    );
}

export default FarmDetails;
