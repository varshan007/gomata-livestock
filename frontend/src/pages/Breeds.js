import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Map as MapIcon, Activity, AlertTriangle, Cpu, Layers, Globe, Users, Settings, LogOut, ChevronDown, User, Menu,
    Search, Plus, Download, Upload, Edit, Eye, Trash2, Battery, Filter, CheckCircle2, ChevronRight, Link as LinkIcon, X, Info
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css'; // Reuse core premium styles
import './Breeds.css';

// Mock Data replaced by live MongoDB aggregate queries

function Breeds() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const [filterSpecies, setFilterSpecies] = useState('All');
    const [activeBreed, setActiveBreed] = useState(null); // Detailed side panel
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

    // Data states
    const [liveBreeds, setLiveBreeds] = useState([]);

    useEffect(() => {
        const fetchBreeds = async () => {
            if (!user) return;
            const token = localStorage.getItem('token');
            if (!token) return;
            try {
                const res = await fetch(`${process.env.REACT_APP_API_URL || 'https://gomata-backend.onrender.com/api'}/livestock/breeds/summary`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                const json = await res.json();
                const data = json.data || json;

                const formatted = (Array.isArray(data) ? data : []).map((b, idx) => ({
                    id: `BR${String(idx + 1).padStart(3, '0')}`,
                    name: b.breed,
                    species: b.species,
                    origin: 'Unknown',
                    count: b.count,
                    avgTemp: '38.5°C', // Simulated mock for now
                    avgActivity: 'Normal',
                    healthScore: 90 + Math.floor(Math.random() * 8), // Simulated health ~90-98
                    dateAdded: b.dateAdded ? new Date(b.dateAdded).toLocaleDateString() : 'N/A',
                    tempRange: '38.0°C – 38.6°C',
                    hrRange: '60–80 bpm',
                    description: 'No detailed ML tracking taxonomy defined for this class yet.'
                }));

                setLiveBreeds(formatted);
            } catch (err) {
                console.error("Error fetching breeds summary:", err);
            }
        };
        fetchBreeds();
    }, [user]);

    // Derived KPI
    const totalBreeds = liveBreeds.length;
    const totalAnimals = liveBreeds.reduce((acc, curr) => acc + curr.count, 0);
    const sortedByCount = [...liveBreeds].sort((a, b) => b.count - a.count);
    const mostCommon = sortedByCount[0]?.name || 'N/A';
    const leastCommon = sortedByCount[sortedByCount.length - 1]?.name || 'N/A';
    const bestHealth = [...liveBreeds].sort((a, b) => b.healthScore - a.healthScore)[0]?.name || 'N/A';

    // Filter Logic
    const filteredBreeds = liveBreeds.filter(b => {
        const matchesSearch = b.name.toLowerCase().includes(searchTerm.toLowerCase());
        const matchesSpecies = filterSpecies === 'All' || b.species === filterSpecies;
        return matchesSearch && matchesSpecies;
    });

    const renderHealthScore = (score) => {
        let colorClass = 'optimal';
        if (score < 80) colorClass = 'warning';
        if (score < 60) colorClass = 'critical';

        return (
            <div className={`health-score-pill ${colorClass}`}>
                {score}%
            </div>
        );
    };

    const renderBreedPanel = () => {
        if (!activeBreed) return null;
        return (
            <div className="breed-detail-panel">
                <div className="bd-header">
                    <button className="bd-close" onClick={() => setActiveBreed(null)}><X size={20} /></button>
                    <h2 className="bd-title">{activeBreed.name}</h2>
                    <div className="bd-status-badges">
                        <span className="bd-badge species">{activeBreed.species}</span>
                        <span className="bd-badge origin"><Globe size={12} style={{ marginRight: 4 }} /> {activeBreed.origin || 'Unknown'}</span>
                    </div>
                </div>

                <div className="bd-content">
                    <div className="bd-section">
                        <h4><Info size={16} /> Basic Intelligence</h4>
                        <div className="bd-grid">
                            <div className="bd-item">
                                <label>Registry ID</label>
                                <div className="val">{activeBreed.id}</div>
                            </div>
                            <div className="bd-item">
                                <label>Total In Fleet</label>
                                <div className="val" style={{ color: '#3b82f6', fontWeight: 700 }}>{activeBreed.count} Animals</div>
                            </div>
                            <div className="bd-item full description-box">
                                <label>AI Processing Context</label>
                                <div className="val text">{activeBreed.description}</div>
                            </div>
                        </div>
                    </div>

                    <div className="bd-section alerts-section">
                        <h4><Activity size={16} /> Health Reference Ranges</h4>
                        <p className="section-desc">These baseline parameters are used by the AI Orchestrator to detect anomalies and trigger health alerts.</p>
                        <div className="reference-ranges">
                            <div className="ref-card temp">
                                <span className="ref-label">Normal Temp Range</span>
                                <span className="ref-val">{activeBreed.tempRange}</span>
                            </div>
                            <div className="ref-card heart">
                                <span className="ref-label">Normal HR Range</span>
                                <span className="ref-val">{activeBreed.hrRange}</span>
                            </div>
                            <div className="ref-card activity">
                                <span className="ref-label">Baseline Activity</span>
                                <span className="ref-val">{activeBreed.avgActivity}</span>
                            </div>
                        </div>
                    </div>

                    <div className="bd-section">
                        <h4><Layers size={16} /> Fleet Distribution</h4>
                        <div className="distribution-list">
                            <div className="dist-item">
                                <span>Barn A</span>
                                <strong>25 Head</strong>
                            </div>
                            <div className="dist-item">
                                <span>Barn B</span>
                                <strong>{(activeBreed.count - 25 > 0) ? activeBreed.count - 25 : 0} Head</strong>
                            </div>
                        </div>
                    </div>

                    <div className="bd-section">
                        <h4>Analytics Overview</h4>
                        <div className="mini-analytics-grid">
                            <div className="ma-card good">
                                <span>Healthy</span>
                                <strong>{Math.floor(activeBreed.count * 0.9)}</strong>
                            </div>
                            <div className="ma-card warning">
                                <span>Warning</span>
                                <strong>{Math.floor(activeBreed.count * 0.08)}</strong>
                            </div>
                            <div className="ma-card critical">
                                <span>Critical</span>
                                <strong>{Math.ceil(activeBreed.count * 0.02)}</strong>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bd-actions">
                    <button className="mgt-btn" onClick={() => setActiveBreed(null)}>Edit Reference Data</button>
                    <button className="mgt-btn primary" onClick={() => navigate('/health-analytics')}>View Breed Health Analytics</button>
                </div>
            </div>
        );
    };

    const renderAddModal = () => {
        if (!showAddModal) return null;
        return (
            <div className="mgt-modal-overlay" onClick={() => setShowAddModal(false)}>
                <div className="mgt-modal-content large" onClick={e => e.stopPropagation()}>
                    <div className="modal-header">
                        <h2>Register New Breed</h2>
                        <button className="icon-btn-premium close-btn" onClick={() => setShowAddModal(false)}><X size={20} /></button>
                    </div>
                    <div className="modal-body">
                        <div className="form-section">
                            <h4>Taxonomy & Origin</h4>
                            <div className="form-grid">
                                <div className="input-group">
                                    <label>Breed Name *</label>
                                    <input type="text" placeholder="e.g. Angus" />
                                </div>
                                <div className="input-group">
                                    <label>Species *</label>
                                    <select><option>Cow</option><option>Buffalo</option><option>Goat</option><option>Sheep</option></select>
                                </div>
                                <div className="input-group full-width">
                                    <label>Origin Country</label>
                                    <input type="text" placeholder="e.g. Scotland" />
                                </div>
                            </div>
                        </div>

                        <div className="form-section gradient-section">
                            <h4 style={{ color: '#1e40af' }}>AI Health Reference Baselines</h4>
                            <p style={{ fontSize: '0.85rem', color: '#3b82f6', marginBottom: 16 }}>Set the exact physiological baselines so the AI can accurately flag anomalies.</p>
                            <div className="form-grid">
                                <div className="input-group">
                                    <label>Normal Temperature Range (°C) *</label>
                                    <div style={{ display: 'flex', gap: 8 }}>
                                        <input type="number" step="0.1" placeholder="Min (e.g. 38.0)" />
                                        <input type="number" step="0.1" placeholder="Max (e.g. 38.6)" />
                                    </div>
                                </div>
                                <div className="input-group">
                                    <label>Normal Heart Rate (BPM)</label>
                                    <div style={{ display: 'flex', gap: 8 }}>
                                        <input type="number" placeholder="Min" />
                                        <input type="number" placeholder="Max" />
                                    </div>
                                </div>
                                <div className="input-group full-width">
                                    <label>Breed-Specific Notes for ML Context</label>
                                    <input type="text" placeholder="Are there specific conditions (like thick coats) the AI should know?" />
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="modal-footer">
                        <button className="mgt-btn" onClick={() => setShowAddModal(false)}>Cancel</button>
                        <button className="mgt-btn primary">Register Standard</button>
                    </div>
                </div>
            </div>
        );
    };

    return (
        <div className="breeds-layout">
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
                        <div className="nav-item-premium active" onClick={() => navigate('/breeds')}>
                            <Layers className="nav-icon" /> <span>Breeds</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/staff')}>
                            <Users className="nav-icon" /> <span>Staff</span>
                        </div>

                        <div className="sidebar-section-title">System</div>
                        <div className="nav-item-premium" onClick={() => navigate('/settings')}>
                            <Settings className="nav-icon" /> <span>Settings</span>
                        </div>
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
            <main className="breeds-main">
                {/* TOP BAR */}
                <header className="mgt-header" style={{ background: 'linear-gradient(135deg, #fdf4ff 0%, #e0e7ff 100%)' }}>
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div className="mgt-title-area">
                            <h1>Breed Taxonomy & Intelligence</h1>
                            <p>Manage species baseline profiles and health reference thresholds for AI</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <button className="mgt-btn"><Download size={16} /> Export Data</button>
                        <button className="mgt-btn"><Upload size={16} /> Import Base</button>
                        <button className="mgt-btn primary" onClick={() => setShowAddModal(true)}><Plus size={16} /> Add Breed</button>
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                {/* KPI CARDS */}
                <div className="breeds-kpi-grid">
                    <div className="breeds-kpi-card">
                        <span className="bkpi-label">Total Breeds Cataloged</span>
                        <span className="bkpi-value">{totalBreeds}</span>
                    </div>
                    <div className="breeds-kpi-card">
                        <span className="bkpi-label">Total Classified Animals</span>
                        <span className="bkpi-value blue">{totalAnimals}</span>
                    </div>
                    <div className="breeds-kpi-card">
                        <span className="bkpi-label">Most Common Breed</span>
                        <span className="bkpi-value text-val">{mostCommon}</span>
                    </div>
                    <div className="breeds-kpi-card">
                        <span className="bkpi-label">Least Common Breed</span>
                        <span className="bkpi-value text-val">{leastCommon}</span>
                    </div>
                    <div className="breeds-kpi-card highlight-card">
                        <span className="bkpi-label">Highest Avg Health</span>
                        <span className="bkpi-value green">{bestHealth}</span>
                    </div>
                </div>

                {/* TABLE SECTION */}
                <div className="breeds-content-wrapper">
                    <div className="breeds-table-card">
                        <div className="mgt-table-header">
                            <div className="mgt-search">
                                <Search size={16} color="#94a3b8" />
                                <input
                                    type="text"
                                    placeholder="Search Breed Name..."
                                    value={searchTerm}
                                    onChange={(e) => setSearchTerm(e.target.value)}
                                />
                            </div>
                            <div className="mgt-filters">
                                <select className="mgt-filter-select" value={filterSpecies} onChange={(e) => setFilterSpecies(e.target.value)}>
                                    <option value="All">All Species</option>
                                    <option value="Cow">Cow</option>
                                    <option value="Buffalo">Buffalo</option>
                                    <option value="Goat">Goat</option>
                                    <option value="Sheep">Sheep</option>
                                </select>
                            </div>
                        </div>

                        <div className="mgt-table-wrapper">
                            <table className="mgt-data-table">
                                <thead>
                                    <tr>
                                        <th>Breed Name</th>
                                        <th>Species</th>
                                        <th>Total Animals</th>
                                        <th>Avg Temp Baseline</th>
                                        <th>Base Activity</th>
                                        <th>Avg Health Fleet</th>
                                        <th>Date Added</th>
                                        <th>Actions</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredBreeds.map(b => (
                                        <tr key={b.id} onClick={() => setActiveBreed(b)} style={{ cursor: 'pointer' }}>
                                            <td style={{ fontWeight: 700, color: '#0f172a', fontSize: '1.05rem' }}>{b.name}</td>
                                            <td><span className="species-badge">{b.species}</span></td>
                                            <td style={{ fontWeight: 600, color: '#3b82f6' }}>{b.count} Head</td>
                                            <td style={{ color: '#64748b' }}>{b.avgTemp}</td>
                                            <td className="cell-sub">{b.avgActivity}</td>
                                            <td>{renderHealthScore(b.healthScore)}</td>
                                            <td className="cell-sub">{b.dateAdded}</td>
                                            <td>
                                                <div className="row-actions" onClick={e => e.stopPropagation()}>
                                                    <button className="action-icon" onClick={() => setActiveBreed(b)} title="View AI Intel"><Activity size={16} /></button>
                                                    <button className="action-icon" title="Edit Guidelines"><Edit size={16} /></button>
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
                {renderBreedPanel()}
                {renderAddModal()}

            </main>
        </div>
    );
}

export default Breeds;
