import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Map as MapIcon, Activity, AlertTriangle, Cpu, Layers, Globe, Users, Settings, LogOut, ChevronDown, User, Menu,
    Search, Plus, Download, Upload, Edit, Eye, Trash2, Battery, Filter, CheckCircle2, ChevronRight, Link as LinkIcon, X, Shield, Phone, Mail, Clock, ShieldAlert, Key
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import { staffAPI, farmsAPI } from '../services/api';
import './LivestockManagement.css'; // Reuse core premium styles
import './Staff.css';

function Staff() {
    const navigate = useNavigate();
    const { user, logout } = useContext(AuthContext);

    // Live Data States
    const [staffList, setStaffList] = useState([]);
    const [farmsList, setFarmsList] = useState([]);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterRole, setFilterRole] = useState('All');
    const [activeStaff, setActiveStaff] = useState(null); // Detailed side panel
    const [showAddModal, setShowAddModal] = useState(false);

    // New Staff Form State
    const [formData, setFormData] = useState({
        name: '', email: '', phone: '', position: 'Worker', role: 'Viewer',
        assignedFarms: [], assignedZones: [], assignedDevices: [],
        primaryResponsibility: 'Farm Operations', assignedShift: 'Full Day',
        alertPreferences: [], status: 'Active'
    });
    const profileRef = useRef();

    useEffect(() => {
        if (user && user.type === 'staff') {
            // Staff members are legally locked out of the Staff Management UI
            navigate('/dashboard');
            return;
        }

        const loadData = async () => {
            try {
                const [staffRes, farmsRes] = await Promise.all([
                    staffAPI.getAll(),
                    farmsAPI.getAll()
                ]);
                setStaffList(staffRes.data);
                setFarmsList(farmsRes.data);
            } catch (err) {
                console.error("Failed to load Staff data", err);
            }
        };
        loadData();
    }, [user, navigate]);

    useEffect(() => {
        const listener = (event) => {
            if (!profileRef.current || profileRef.current.contains(event.target)) return;
            if (showProfileMenu) setShowProfileMenu(false);
        };
        document.addEventListener("click", listener);
        return () => document.removeEventListener("click", listener);
    }, [showProfileMenu]);

    // Derived KPI
    const totalStaff = staffList.length;
    const activeStaffCount = staffList.length; // Mocked as active by default
    const adminsCount = 1; // You
    const farmWorkersCount = staffList.filter(s => s.role === 'Operator').length;
    const vetsCount = staffList.filter(s => s.role === 'Farm Manager').length;

    // Filter Logic
    const filteredStaff = staffList.filter(s => {
        const matchesSearch = s.name.toLowerCase().includes(searchTerm.toLowerCase()) || s.email.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesRole = filterRole === 'All' || s.role === filterRole;
        return matchesSearch && matchesRole;
    });

    const handleCreateStaff = async (e) => {
        e.preventDefault();
        try {
            const res = await staffAPI.create(formData);
            setStaffList([...staffList, res.data]);
            setShowAddModal(false);
            setFormData({
                name: '', email: '', phone: '', position: 'Worker', role: 'Viewer',
                assignedFarms: [], assignedZones: [], assignedDevices: [],
                primaryResponsibility: 'Farm Operations', assignedShift: 'Full Day',
                alertPreferences: [], status: 'Active'
            });
            alert(`User created! Their initial Staff ID is: ${res.data.userId}`);
        } catch (err) {
            console.error("Error creating staff:", err);
            alert(err.response?.data?.message || "Error creating user");
        }
    };

    const renderAccessBadge = (level) => {
        let cls = 'read-only';
        if (level === 'Full') cls = 'full';
        else if (level === 'Manager') cls = 'manager';
        else if (level === 'Medical') cls = 'medical';
        else if (level === 'Limited') cls = 'limited';

        return <span className={`access-badge ${cls}`}>{level} Access</span>;
    };

    const renderStatus = (status) => {
        return (
            <div className={`staff-status ${status.toLowerCase()}`}>
                <div className="dot"></div>
                {status}
            </div>
        );
    };

    const renderPermissionsList = (level) => {
        // Mocking permission logic based on level
        const perms = {
            'Full': [
                { name: 'All Animals & Devices', granted: true },
                { name: 'System Settings & Staff', granted: true },
                { name: 'Delete Farm Data', granted: true }
            ],
            'Manager': [
                { name: 'Animals, Devices & Alerts', granted: true },
                { name: 'Analytics & Maps', granted: true },
                { name: 'System Settings & Staff', granted: false },
                { name: 'Delete Farm Data', granted: false }
            ],
            'Medical': [
                { name: 'Animal Health & Alerts', granted: true },
                { name: 'Edit Vital Thresholds', granted: true },
                { name: 'Edit Farm Infrastructure', granted: false }
            ],
            'Limited': [
                { name: 'View Assigned Animals', granted: true },
                { name: 'Acknowledge Basic Alerts', granted: true },
                { name: 'Access Overall Analytics', granted: false }
            ],
            'Read-Only': [
                { name: 'View System Analytics', granted: true },
                { name: 'Edit Data or Acknowledge Alerts', granted: false }
            ]
        };

        const list = perms[level] || perms['Read-Only'];

        return (
            <div className="permission-list">
                {list.map((p, i) => (
                    <div key={i} className={`perm-item ${!p.granted ? 'denied' : ''}`}>
                        {p.granted ? <CheckCircle2 size={16} /> : <X size={16} />}
                        {p.name}
                    </div>
                ))}
            </div>
        );
    };

    const renderStaffPanel = () => {
        if (!activeStaff) return null;
        return (
            <div className="staff-detail-panel">
                <div className="sd-header">
                    <button className="sd-close" onClick={() => setActiveStaff(null)}><X size={20} /></button>
                    <div className="sd-avatar">{activeStaff.name.charAt(0)}</div>
                    <div className="sd-header-text">
                        <h2 className="sd-title">{activeStaff.name}</h2>
                        <p className="sd-subtitle">{activeStaff.role}</p>
                    </div>
                </div>

                <div className="sd-content">
                    <div className="sd-section">
                        <h4><User size={16} /> Basic Contact</h4>
                        <div className="sd-grid">
                            <div className="sd-item full">
                                <label>Email Address</label>
                                <div className="val" style={{ color: '#3b82f6', display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <Mail size={14} /> {activeStaff.email}
                                </div>
                            </div>
                            <div className="sd-item full">
                                <label>Phone Number</label>
                                <div className="val" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                    <Phone size={14} color="#64748b" /> {activeStaff.phone}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="sd-section">
                        <h4><MapIcon size={16} /> Assigned Locations</h4>
                        <div className="sd-grid">
                            <div className="sd-item">
                                <label>Farm Assignment</label>
                                <div className="val">{activeStaff.assignedFarm}</div>
                            </div>
                            <div className="sd-item">
                                <label>Zone Restrictions</label>
                                <div className="val">{activeStaff.assignedZone}</div>
                            </div>
                        </div>
                    </div>

                    <div className="sd-section">
                        <h4><Shield size={16} /> System Access Control</h4>
                        <div style={{ marginBottom: 16 }}>
                            {renderAccessBadge(activeStaff.accessLevel)}
                        </div>
                        {renderPermissionsList(activeStaff.accessLevel)}
                    </div>

                    <div className="sd-section">
                        <h4><Activity size={16} /> Activity & Status</h4>
                        <div className="sd-grid">
                            <div className="sd-item">
                                <label>Account Status</label>
                                <div className="val">{renderStatus(activeStaff.status)}</div>
                            </div>
                            <div className="sd-item">
                                <label>Last Login</label>
                                <div className="val" style={{ display: 'flex', alignItems: 'center', gap: 4 }}><Clock size={12} /> {activeStaff.lastLogin}</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="sd-actions">
                    <div className="btn-row">
                        <button className="mgt-btn primary">Edit Assigned Routes</button>
                        <button className="mgt-btn"><Key size={16} style={{ marginRight: 6 }} /> Reset Password</button>
                    </div>
                    <button className="mgt-btn danger">Deactivate / Suspend Account</button>
                </div>
            </div>
        );
    };

    const renderAddModal = () => {
        if (!showAddModal) return null;

        // Flatten zones across selected farms
        const availableZones = farmsList
            .filter(f => formData.assignedFarms.includes(f._id))
            .flatMap(f => f.zones || []);

        return (
            <div className="mgt-modal-overlay" onClick={() => setShowAddModal(false)}>
                <div className="mgt-modal-content" onClick={e => e.stopPropagation()} style={{ maxWidth: '800px', maxHeight: '90vh', overflowY: 'auto' }}>
                    <div className="modal-header">
                        <h2>Invite Staff Member</h2>
                        <button type="button" className="icon-btn-premium close-btn" onClick={() => setShowAddModal(false)}><X size={20} /></button>
                    </div>
                    <form onSubmit={handleCreateStaff}>
                        <div className="modal-body" style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '24px' }}>
                            {/* LEFT COLUMN */}
                            <div>
                                <div className="form-section">
                                    <h4>Personal Details</h4>
                                    <div className="form-grid">
                                        <div className="input-group full-width">
                                            <label>Full Name *</label>
                                            <input type="text" placeholder="e.g. Rahul Sharma" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} required />
                                        </div>
                                        <div className="input-group full-width">
                                            <label>Phone Number *</label>
                                            <input type="tel" placeholder="+1 (555) 000-0000" value={formData.phone} onChange={e => setFormData({ ...formData, phone: e.target.value })} required />
                                        </div>
                                        <div className="input-group full-width">
                                            <label>Email Address</label>
                                            <input type="email" placeholder="name@domain.com (Optional)" value={formData.email} onChange={e => setFormData({ ...formData, email: e.target.value })} />
                                        </div>
                                        <div className="input-group full-width">
                                            <label>Staff ID (Auto)</label>
                                            <input type="text" value="Generated automatically upon creation" disabled style={{ background: '#f1f5f9', color: '#64748b', fontStyle: 'italic' }} />
                                        </div>
                                    </div>
                                </div>

                                <div className="form-section">
                                    <h4>Role & Access</h4>
                                    <div className="form-grid">
                                        <div className="input-group full-width">
                                            <label>Role / Position *</label>
                                            <select value={formData.position} onChange={e => setFormData({ ...formData, position: e.target.value })}>
                                                <option value="Farm Manager">Farm Manager</option>
                                                <option value="Supervisor">Supervisor</option>
                                                <option value="Veterinarian">Veterinarian</option>
                                                <option value="Technician">Technician</option>
                                                <option value="Worker">Worker</option>
                                            </select>
                                        </div>
                                        <div className="input-group full-width">
                                            <label>System Access Level *</label>
                                            <select value={formData.role} onChange={e => setFormData({ ...formData, role: e.target.value })}>
                                                <option value="Admin">Admin (Full farm control)</option>
                                                <option value="Manager">Manager (Manage livestock & staff in assigned farms)</option>
                                                <option value="Operator">Operator (Monitor & update livestock)</option>
                                                <option value="Viewer">Viewer (Read-only access)</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="form-section">
                                    <h4>Account Status</h4>
                                    <div className="form-grid">
                                        <div className="input-group full-width">
                                            <label>Status *</label>
                                            <select value={formData.status} onChange={e => setFormData({ ...formData, status: e.target.value })}>
                                                <option value="Active">Active</option>
                                                <option value="Inactive">Inactive</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {/* RIGHT COLUMN */}
                            <div>
                                <div className="form-section">
                                    <h4>Operational Assignment</h4>
                                    <div className="form-grid">
                                        <div className="input-group full-width">
                                            <label>Assign to Farm *</label>
                                            <select multiple size="3" onChange={e => {
                                                const options = [...e.target.options].filter(o => o.selected).map(o => o.value);
                                                setFormData({ ...formData, assignedFarms: options, assignedZones: [] });
                                            }} required>
                                                {farmsList.map(f => (
                                                    <option key={f._id} value={f._id}>{f.name}</option>
                                                ))}
                                            </select>
                                        </div>
                                        <div className="input-group full-width">
                                            <label>Restrict to Zone (Optional)</label>
                                            <select multiple size="3" onChange={e => {
                                                const options = [...e.target.options].filter(o => o.selected).map(o => o.value);
                                                setFormData({ ...formData, assignedZones: options });
                                            }}>
                                                {availableZones.map(z => (
                                                    <option key={z._id} value={z._id}>{z.name}</option>
                                                ))}
                                                {availableZones.length === 0 && <option disabled>Select a Farm first...</option>}
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="form-section">
                                    <h4>Responsibilities</h4>
                                    <div className="form-grid">
                                        <div className="input-group full-width">
                                            <label>Primary Responsibility</label>
                                            <select value={formData.primaryResponsibility} onChange={e => setFormData({ ...formData, primaryResponsibility: e.target.value })}>
                                                <option value="Livestock Monitoring">Livestock Monitoring</option>
                                                <option value="Health Monitoring">Health Monitoring</option>
                                                <option value="Device Maintenance">Device Maintenance</option>
                                                <option value="Farm Operations">Farm Operations</option>
                                                <option value="Breeding Management">Breeding Management</option>
                                            </select>
                                        </div>
                                        <div className="input-group full-width">
                                            <label>Assigned Shift</label>
                                            <select value={formData.assignedShift} onChange={e => setFormData({ ...formData, assignedShift: e.target.value })}>
                                                <option value="Morning">Morning</option>
                                                <option value="Afternoon">Afternoon</option>
                                                <option value="Night">Night</option>
                                                <option value="Full Day">Full Day</option>
                                            </select>
                                        </div>
                                    </div>
                                </div>

                                <div className="form-section">
                                    <h4>Alert Preferences</h4>
                                    <div className="form-grid">
                                        <div className="input-group full-width" style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                                            {['Health alerts', 'Geofence breach', 'Device offline', 'Low battery'].map(pref => (
                                                <label key={pref} style={{ display: 'flex', alignItems: 'center', gap: '8px', cursor: 'pointer', fontWeight: 400, color: '#334155' }}>
                                                    <input
                                                        type="checkbox"
                                                        checked={formData.alertPreferences.includes(pref)}
                                                        onChange={e => {
                                                            const newPrefs = e.target.checked
                                                                ? [...formData.alertPreferences, pref]
                                                                : formData.alertPreferences.filter(p => p !== pref);
                                                            setFormData({ ...formData, alertPreferences: newPrefs });
                                                        }}
                                                    />
                                                    {pref}
                                                </label>
                                            ))}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div className="modal-footer">
                            <button type="button" className="mgt-btn" onClick={() => setShowAddModal(false)}>Cancel</button>
                            <button type="submit" className="mgt-btn primary" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                                <Shield size={16} /> Create Secure User
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        );
    };

    return (
        <div className="staff-layout">
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
                        <div className="nav-item-premium" onClick={() => navigate('/farm')}>
                            <Globe className="nav-icon" /> <span>Farms & Locations</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/breeds')}>
                            <Layers className="nav-icon" /> <span>Breeds</span>
                        </div>
                        <div className="nav-item-premium active" onClick={() => navigate('/staff')}>
                            <Users className="nav-icon" /> <span>Staff</span>
                        </div>

                        <div className="sidebar-section-title">System</div>
                        <div className="nav-item-premium" onClick={() => navigate('/settings')}>
                            <Settings className="nav-icon" /> <span>Settings</span>
                        </div>
                    </nav>
                </div>

                <div className="sidebar-bottom">
                    <div className="nav-item-premium" onClick={() => navigate('/profile')}>
                        <User className="nav-icon" /> <span>Profile</span>
                    </div>
                    <div className="nav-item-premium logout" onClick={() => navigate('/')}>
                        <LogOut className="nav-icon" /> <span>Logout</span>
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT */}
            <main className="staff-main">
                {/* TOP BAR */}
                <header className="mgt-header staff-header">
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div className="mgt-title-area">
                            <h1>Workforce & Access</h1>
                            <p>Manage farm personnel, security roles, and system permission levels</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <button className="mgt-btn"><Download size={16} /> Export List</button>
                        <button className="mgt-btn primary" onClick={() => setShowAddModal(true)}><Plus size={16} /> Invite User</button>
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'A'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                {/* KPI CARDS */}
                <div className="staff-kpi-grid">
                    <div className="staff-kpi-card">
                        <span className="skpi-label">Total Staff</span>
                        <span className="skpi-value">{totalStaff}</span>
                    </div>
                    <div className="staff-kpi-card">
                        <span className="skpi-label">Active Users</span>
                        <span className="skpi-value green">{activeStaffCount}</span>
                    </div>
                    <div className="staff-kpi-card">
                        <span className="skpi-label">System Admins</span>
                        <span className="skpi-value blue">{adminsCount}</span>
                    </div>
                    <div className="staff-kpi-card">
                        <span className="skpi-label">Farm Workers</span>
                        <span className="skpi-value">{farmWorkersCount}</span>
                    </div>
                    <div className="staff-kpi-card">
                        <span className="skpi-label">Veterinarians</span>
                        <span className="skpi-value purple">{vetsCount}</span>
                    </div>
                </div>

                {/* TABLE SECTION */}
                <div className="staff-content-wrapper">
                    <div className="staff-table-card">
                        <div className="mgt-table-header">
                            <div className="mgt-search">
                                <Search size={16} color="#94a3b8" />
                                <input
                                    type="text"
                                    placeholder="Search by Name, Email, or Phone..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>
                            <div className="mgt-filters">
                                <select className="mgt-filter-select" value={filterRole} onChange={(e) => setFilterRole(e.target.value)}>
                                    <option value="All">All Roles</option>
                                    <option value="Admin">Admin</option>
                                    <option value="Farm Manager">Farm Manager</option>
                                    <option value="Veterinarian">Veterinarian</option>
                                    <option value="Farm Worker">Farm Worker</option>
                                    <option value="Viewer">Viewer</option>
                                </select>
                            </div>
                        </div>

                        <div className="mgt-table-wrapper">
                            <table className="mgt-data-table">
                                <thead>
                                    <tr>
                                        <th>Name & Contact</th>
                                        <th>Role / Position</th>
                                        <th>Assigned Location</th>
                                        <th>Security Level</th>
                                        <th>Account Status</th>
                                        <th>Last Login Activity</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredStaff.map(s => (
                                        <tr key={s._id} onClick={() => setActiveStaff({ ...s, accessLevel: s.role === 'Viewer' ? 'Read-Only' : s.role === 'Operator' ? 'Limited' : 'Manager', phone: s.phone || '--', status: s.status || 'Active', lastLogin: new Date(s.createdAt).toLocaleString(), assignedFarm: s.assignedFarms.length + ' Farms', assignedZone: s.assignedZones.length + ' Zones' })} style={{ cursor: 'pointer' }}>
                                            <td>
                                                <div className="cell-id" style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                                                    <div className="micro-avatar" style={{ minWidth: 28, minHeight: 28, borderRadius: '50%', background: '#e0e7ff', color: '#3b82f6', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '11px', fontWeight: 700 }}>{s.name.charAt(0)}</div>
                                                    <div>
                                                        <div style={{ fontWeight: 600, color: '#1e293b' }}>{s.name}</div>
                                                        <div style={{ fontSize: '0.8rem', color: '#64748b' }}>{s.userId || s.email}</div>
                                                    </div>
                                                </div>
                                                <div className="cell-sub" style={{ marginTop: '6px', fontSize: '0.8rem' }}>{s.phone}</div>
                                            </td>
                                            <td style={{ fontWeight: 500, color: '#334155' }}>{s.position || 'Worker'}</td>
                                            <td>
                                                <div style={{ color: '#0f172a', fontWeight: 500 }}>{s.assignedFarms?.length || 0} Farms</div>
                                                <div className="cell-sub">{s.assignedZones?.length || 0} Restricted Zones</div>
                                            </td>
                                            <td>{renderAccessBadge(s.role)}</td>
                                            <td>{renderStatus(s.status || 'Active')}</td>
                                            <td className="cell-sub">{new Date(s.createdAt).toLocaleDateString()}</td>
                                            <td>
                                                <div className="row-actions" onClick={e => e.stopPropagation()}>
                                                    <button className="action-icon" onClick={() => setActiveStaff({ ...s, accessLevel: s.role === 'Viewer' ? 'Read-Only' : s.role === 'Operator' ? 'Limited' : 'Manager', phone: '--', status: 'Active', lastLogin: new Date(s.createdAt).toLocaleString(), assignedFarm: s.assignedFarms.length + ' Farms', assignedZone: s.assignedZones.length + ' Zones' })} title="View Permissions"><ShieldAlert size={16} /></button>
                                                    <button className="action-icon" title="Edit Staff"><Edit size={16} /></button>
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
                {renderStaffPanel()}
                {renderAddModal()}

            </main>
        </div>
    );
}

export default Staff;
