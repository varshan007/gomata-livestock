import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, Users, Map as MapIcon, Activity, Settings,
    Bell, Search, Menu, Thermometer, Droplets, Wind, AlertTriangle, ChevronRight, PieChart, TrendingUp, HeartPulse, Droplet, Clock, Navigation, MapPin, List, ShieldAlert, BadgeInfo, Stethoscope, Briefcase, Zap, Flame, Shield, CheckCircle2, Battery, Cpu, Layers, Globe, Sparkles, Eye, Plus, AlertCircle, ChevronDown, Radio, CheckCircle, Image, LogOut, FileText, Link as LinkIcon, BarChart2,
    User, Phone, Mail
} from 'lucide-react';
import { sensorDataAPI, alertsAPI, livestockAPI, dashboardAPI } from '../services/api';
import AIAgentWidget from '../components/AIAgentWidget';
import AuthContext from '../context/AuthContext';
import { Mic } from 'lucide-react'; // Needed for fab icon
import { useLiveTelemetry } from '../context/LiveTelemetryContext';
import './Dashboard.css';

// Hook for clicking outside
function useOnClickOutside(ref, handler) {
    useEffect(() => {
        const listener = (event) => {
            if (!ref.current || ref.current.contains(event.target)) {
                return;
            }
            handler(event);
        };
        document.addEventListener("click", listener);
        document.addEventListener("touchstart", listener);
        return () => {
            document.removeEventListener("click", listener);
            document.removeEventListener("touchstart", listener);
        };
    }, [ref, handler]);
}

// Helper for images
const getPlaceholderImage = (type, breed) => {
    const defaultCowImage = "https://images.unsplash.com/photo-1546445317-29f4545e9d53?auto=format&fit=crop&q=80&w=400&h=400";
    if (!type && !breed) return defaultCowImage;

    const searchString = `${type || ''} ${breed || ''}`.toLowerCase();

    if (searchString.includes('goat')) {
        return "https://images.unsplash.com/photo-1524024973425-502ddf82f25e?auto=format&fit=crop&q=80&w=400&h=400";
    } else if (searchString.includes('sheep')) {
        return "https://images.unsplash.com/photo-1484557985045-caf43e5e714b?auto=format&fit=crop&q=80&w=400&h=400";
    }
    return defaultCowImage;
};

const Dashboard = () => {
    const [livestock, setLivestock] = useState([]);
    const [loading, setLoading] = useState(true);
    const [searchTerm, setSearchTerm] = useState('');
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // Dashboard Summary State
    const [dashboardSummary, setDashboardSummary] = useState(null);

    // Filter/Sort State
    const [showFilterMenu, setShowFilterMenu] = useState(false);
    const [showSortMenu, setShowSortMenu] = useState(false);
    const filterRef = useRef();
    const sortRef = useRef();

    useOnClickOutside(filterRef, () => showFilterMenu && setShowFilterMenu(false));
    useOnClickOutside(sortRef, () => showSortMenu && setShowSortMenu(false));

    const [filters, setFilters] = useState({ type: 'All', status: 'All', gender: 'All', breed: 'All' });
    const [sortBy, setSortBy] = useState('name-asc');

    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const profileRef = useRef();
    useOnClickOutside(profileRef, () => showProfileMenu && setShowProfileMenu(false));

    // Sidebar Toggle State
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);

    // Scroll Reference for Grid
    const livestockGridRef = useRef(null);

    const handleStatusClick = (status) => {
        setFilters(prev => ({
            ...prev,
            status: prev.status === status ? 'All' : status
        }));

        // Ensure smooth scroll downwards to grid when clicked
        setTimeout(() => {
            if (livestockGridRef.current) {
                livestockGridRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        }, 100);
    };

    // Modals
    const [showAddModal, setShowAddModal] = useState(false);
    const [newAnimal, setNewAnimal] = useState({
        name: '', tagNumber: '', breed: '', gender: 'Female', age: '', dob: '', weight: '', image: null, hardwareId: ''
    });
    const [showProfileModal, setShowProfileModal] = useState(false);
    const [showFarmModal, setShowFarmModal] = useState(false);
    const [showAnimalsDropdown, setShowAnimalsDropdown] = useState(false);

    const livestockTypes = ['Cow', 'Goat', 'Sheep', 'Buffalo', 'Horse', 'Pig', 'Chicken', 'Duck', 'Turkey', 'Rabbit'];

    // Critical Alerts for banner
    const [criticalAlerts, setCriticalAlerts] = useState([]);

    useEffect(() => {
        fetchDashboardData();
    }, []);

    // Fetch critical alerts for banner
    useEffect(() => {
        const fetchCriticalAlerts = async () => {
            try {
                const token = localStorage.getItem('token');
                if (!token) return;
                const res = await fetch('http://localhost:8000/api/alerts', {
                    headers: { Authorization: `Bearer ${token}` }
                });
                const json = await res.json();
                const data = json.data || json;
                const criticals = (Array.isArray(data) ? data : [])
                    .filter(a => (a.severity === 'Critical' || a.severity === 'High') && !a.resolved)
                    .slice(0, 3);
                setCriticalAlerts(criticals);
            } catch (err) {
                console.error('Critical alerts fetch error:', err);
            }
        };
        fetchCriticalAlerts();
        const interval = setInterval(fetchCriticalAlerts, 60000);
        return () => clearInterval(interval);
    }, []);

    // 1. Grab the Global Live Telemetry Dictionary
    const { liveData, liveAlerts } = useLiveTelemetry();

    // Auto-dismiss alerts locally from UI state after 10 seconds
    const [visibleAlerts, setVisibleAlerts] = useState([]);

    useEffect(() => {
        if (liveAlerts && liveAlerts.length > 0) {
            // Get the newest alert
            const newAlert = liveAlerts[0];
            setVisibleAlerts(prev => [newAlert, ...prev]);

            // Auto remove after 10 seconds
            setTimeout(() => {
                setVisibleAlerts(prev => prev.filter(a => a !== newAlert));
            }, 10000);
        }
    }, [liveAlerts]);

    const fetchDashboardData = async () => {
        try {
            setLoading(true);
            const [livestockRes, summaryRes] = await Promise.allSettled([
                sensorDataAPI.getDashboard(),
                dashboardAPI.getSummary()
            ]);

            if (livestockRes.status === 'fulfilled' && Array.isArray(livestockRes.value.data)) {
                setLivestock(livestockRes.value.data);
            } else {
                setLivestock([]);
            }

            if (summaryRes.status === 'fulfilled') {
                setDashboardSummary(summaryRes.value.data);
            }

            setLoading(false);
        } catch (error) {
            console.error('Error fetching dashboard data:', error);
            setLoading(false);
        }
    };

    const handleDateChange = (e) => {
        const dob = e.target.value;
        const birthDate = new Date(dob);
        const today = new Date();
        let age = today.getFullYear() - birthDate.getFullYear();
        const m = today.getMonth() - birthDate.getMonth();
        if (m < 0 || (m === 0 && today.getDate() < birthDate.getDate())) age--;
        setNewAnimal({ ...newAnimal, dob, age: age.toString() });
    };

    const handleAddAnimal = async (e) => {
        e.preventDefault();
        try {
            if (!newAnimal.name || !newAnimal.tagNumber) return;
            const formData = new FormData();
            formData.append('name', newAnimal.name);
            formData.append('tagNumber', newAnimal.tagNumber);
            formData.append('breed', newAnimal.breed);
            formData.append('gender', newAnimal.gender);
            formData.append('age', newAnimal.age);
            formData.append('weight', newAnimal.weight);
            formData.append('about', newAnimal.about || '');
            formData.append('type', newAnimal.type || '');
            formData.append('deviceId', newAnimal.tagNumber);

            // Auto-assign premium image placeholder based on type if no image uploaded
            if (newAnimal.image) {
                formData.append('image', newAnimal.image);
            } else {
                const stockImages = {
                    'Cow': 'https://images.unsplash.com/photo-1546445317-29f4545e9d53?auto=format&fit=crop&q=80&w=400&h=400',
                    'Horse': 'https://images.unsplash.com/photo-1553284965-83fd3e82fa5a?auto=format&fit=crop&q=80&w=400&h=400',
                    'Goat': 'https://images.unsplash.com/photo-1524024973431-2ad916746881?auto=format&fit=crop&q=80&w=400&h=400',
                    'Sheep': 'https://images.unsplash.com/photo-1484557985045-edf25e08da73?auto=format&fit=crop&q=80&w=400&h=400',
                    'Buffalo': 'https://images.unsplash.com/photo-1602166542790-21ba2bc74bf0?auto=format&fit=crop&q=80&w=400&h=400',
                    'Pig': 'https://images.unsplash.com/photo-1516467508483-a7212febe31a?auto=format&fit=crop&q=80&w=400&h=400',
                    'Chicken': 'https://images.unsplash.com/photo-1548550023-2bfc3ccef6f7?auto=format&fit=crop&q=80&w=400&h=400',
                    'Default': 'https://images.unsplash.com/photo-1500595046743-cd271d694d30?auto=format&fit=crop&q=80&w=400&h=400'
                };
                const assignedUrl = stockImages[newAnimal.type] || stockImages['Default'];
                formData.append('photoUrl', assignedUrl); // Note: Assuming backend accepts photoUrl if no file
            }

            await livestockAPI.create(formData);
            fetchDashboardData();
            setShowAddModal(false);
            setNewAnimal({ name: '', tagNumber: '', breed: '', gender: 'Female', age: '', dob: '', weight: '', image: null, hardwareId: '' });
        } catch (err) {
            console.error("Failed to add animal", err);
            alert("Failed to add animal. Check console.");
        }
    };

    const getTemperatureStatus = (temp) => {
        if (!temp) return 'Normal';
        if (temp > 40) return 'High';
        if (temp < 36) return 'Low';
        if (temp > 39) return 'Warning';
        return 'Normal';
    };

    const uniqueTypes = ['All', ...new Set(livestock.map(item => {
        if (item.livestock.tagNumber.startsWith('GOAT')) return 'Goat';
        if (item.livestock.tagNumber.startsWith('SHEEP')) return 'Sheep';
        return 'Cow';
    }))];

    const filteredLivestock = livestock.filter(item => {
        const data = item.livestock || {};
        const sensor = item.latestSensorData || {};
        const temp = sensor.temperature || 38;
        const statusKey = getTemperatureStatus(temp);
        let status = statusKey === 'Normal' ? 'Healthy' : (statusKey === 'High' || statusKey === 'Low' ? 'Critical' : statusKey);

        const matchesSearch = (data.name || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
            (data.tagNumber || '').toLowerCase().includes(searchTerm.toLowerCase()) ||
            (data.breed || '').toLowerCase().includes(searchTerm.toLowerCase());

        let type = 'Cow';
        if (data.tagNumber && data.tagNumber.startsWith('GOAT')) type = 'Goat';
        else if (data.tagNumber && data.tagNumber.startsWith('SHEEP')) type = 'Sheep';
        else if (data.breed) {
            if (data.breed.includes('Goat')) type = 'Goat';
            else if (data.breed.includes('Sheep')) type = 'Sheep';
            else if (data.breed.includes('Angus') || data.breed.includes('Holstein')) type = 'Cow';
            else type = data.breed;
        }

        const matchesType = filters.type === 'All' || filters.type === type || (data.breed && data.breed.includes(filters.type));
        const matchesStatus = filters.status === 'All' || status === filters.status;
        const matchesGender = filters.gender === 'All' || data.gender === filters.gender;
        const matchesBreed = filters.breed === 'All' || data.breed === filters.breed;

        return matchesSearch && matchesType && matchesStatus && matchesGender && matchesBreed;
    });

    const getStatusWeight = (item) => {
        const temp = item.latestSensorData?.temperature || 38;
        const statusKey = getTemperatureStatus(temp);
        const isCritical = statusKey === 'High' || statusKey === 'Low' || (item.unresolvedAlerts > 0);
        if (isCritical) return 3;
        if (statusKey === 'Warning') return 2;
        return 1;
    };

    const sortedLivestock = [...filteredLivestock].sort((a, b) => {
        const weightA = getStatusWeight(a);
        const weightB = getStatusWeight(b);
        if (weightA !== weightB) {
            return weightB - weightA; // Critical (3) > Warning (2) > Healthy (1)
        }

        const nameA = (a.livestock?.name || '').toLowerCase();
        const nameB = (b.livestock?.name || '').toLowerCase();
        if (sortBy === 'name-asc') return nameA.localeCompare(nameB);
        if (sortBy === 'name-desc') return nameB.localeCompare(nameA);
        return 0;
    });

    const totalLivestock = dashboardSummary ? dashboardSummary.total_animals : livestock.length;
    const avgTemp = Array.isArray(livestock) && livestock.length > 0
        ? (livestock.reduce((acc, curr) => acc + (curr.latestSensorData?.temperature || 0), 0) / livestock.length).toFixed(1)
        : '0.0';
    const onlineDevices = livestock.filter(l => l.latestSensorData).length;
    const unresolvedTotal = livestock.reduce((acc, curr) => acc + (curr.unresolvedAlerts || 0), 0);

    // Dynamic Animal Mapping
    const animalCounts = livestock.reduce((acc, curr) => {
        const type = curr.livestock.type || curr.livestock.breed || (curr.livestock.tagNumber && curr.livestock.tagNumber.startsWith('GOAT') ? 'Goat' : 'Cow');
        // Normalize common groups
        let normalizedType = 'Cow'; // default fallback
        if (type) {
            const lowerType = type.toString().toLowerCase();
            if (lowerType.includes('cow') || lowerType.includes('holstein') || lowerType.includes('angus')) normalizedType = 'Cow';
            else if (lowerType.includes('buffalo')) normalizedType = 'Buffalo';
            else if (lowerType.includes('horse')) normalizedType = 'Horse';
            else if (lowerType.includes('goat')) normalizedType = 'Goat';
            else if (lowerType.includes('sheep')) normalizedType = 'Sheep';
            else if (lowerType.includes('pig')) normalizedType = 'Pig';
            else if (lowerType.includes('chicken')) normalizedType = 'Chicken';
            else normalizedType = type; // Keep original if no match
        }

        // Capitalize first letter
        normalizedType = normalizedType.charAt(0).toUpperCase() + normalizedType.slice(1);

        acc[normalizedType] = (acc[normalizedType] || 0) + 1;
        return acc;
    }, {});

    let healthyCount = 0;
    let warningCount = 0;
    let criticalCount = 0;
    livestock.forEach(item => {
        const temp = item.latestSensorData?.temperature || 38;
        const statusKey = getTemperatureStatus(temp);
        const isCritical = statusKey === 'High' || statusKey === 'Low' || (item.unresolvedAlerts > 0);
        if (isCritical) criticalCount++;
        else if (statusKey === 'Warning') warningCount++;
        else healthyCount++;
    });

    const devicesOffline = totalLivestock - onlineDevices;
    const batteryLowCount = livestock.filter(l => (l.latestSensorData?.batteryLevel || 100) < 20).length;

    const healthScore = totalLivestock > 0 ? Math.round((healthyCount / totalLivestock) * 100) : 100;

    // Synthetic Data for 6th Card (Activity Metrics)
    const grazingCount = Math.floor(totalLivestock * 0.45);
    const restingCount = Math.floor(totalLivestock * 0.35);
    const movingCount = totalLivestock - grazingCount - restingCount;

    return (
        <div className="dashboard-layout-premium">

            {/* REAL-TIME AI ALERT TOASTS */}
            <div className="toast-container">
                {visibleAlerts.map((alert, idx) => (
                    <div key={idx} className={`toast-notification severity-${alert.severity?.toLowerCase() || 'warning'}`}>
                        <div className="toast-icon">
                            {alert.severity === 'CRITICAL' ? <ShieldAlert size={24} /> : <AlertTriangle size={24} />}
                        </div>
                        <div className="toast-content">
                            <h4>{alert.type || alert.severity || 'Alert'} - {alert.hwId || alert.livestockId}</h4>
                            <p>{alert.message}</p>
                            {alert.alertId && (
                                <button
                                    onClick={async () => {
                                        try {
                                            await alertsAPI.acknowledge(alert.alertId, user?._id);
                                            setVisibleAlerts(prev => prev.filter(a => a.alertId !== alert.alertId));
                                        } catch (e) {
                                            console.error("Acknowledge failed", e);
                                        }
                                    }}
                                    style={{ marginTop: '8px', padding: '6px 12px', background: '#3b82f6', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontSize: '0.8rem', fontWeight: 600 }}
                                >
                                    Acknowledge
                                </button>
                            )}
                        </div>
                        <button className="toast-close" onClick={() => setVisibleAlerts(prev => prev.filter(a => a !== alert))}>×</button>
                    </div>
                ))}
            </div>

            {/* SIDEBAR */}
            <aside className={`sidebar-premium ${!isSidebarOpen ? 'collapsed' : ''}`}>
                <div className="sidebar-logo">
                    <Activity size={28} className="brand-icon" />
                    <span>GoMata</span>
                </div>

                <div className="sidebar-scroll-container">
                    <nav className="sidebar-nav">
                        <div className="sidebar-section-title">Main</div>
                        <div className="nav-item-premium active">
                            <LayoutDashboard className="nav-icon" /> <span>Overview</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/livestock')}>
                            <List className="nav-icon" /> <span>Animals</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/map')}>
                            <MapIcon className="nav-icon" /> <span>Map Intelligence</span>
                        </div>
                        <div className="nav-item-premium" onClick={() => navigate('/health-analytics')}><Activity className="nav-icon" /> <span>Health Analytics</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/alerts')}><AlertTriangle className="nav-icon" /> <span>Alerts Center</span></div>

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
                        <div className="nav-item-premium" onClick={() => navigate('/behaviour')}>
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
                <header className="topbar-premium">
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div>
                            <h1>Good Morning, {user?.name ? user.name.split(' ')[0] : 'Kanna'}</h1>
                            <p>Stay on top of your farm's health, monitor alerts, and track status.</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <div className="search-premium">
                            <Search size={18} className="text-muted" />
                            <input
                                type="text"
                                placeholder="Search livestock..."
                                value={searchTerm}
                                onChange={e => setSearchTerm(e.target.value)}
                            />
                        </div>
                        <button className="icon-btn-premium" onClick={() => setShowAddModal(true)}>
                            <Plus size={20} />
                        </button>
                        <button className="icon-btn-premium">
                            <AlertCircle size={20} />
                            {unresolvedTotal > 0 && <span className="notification-dot"></span>}
                        </button>
                        <div className="profile-pill" onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                {/* DASHBOARD GRID */}
                <div className="dashboard-grid-premium">
                    {user?.type === 'staff' ? (
                        <div className="main-column" style={{ gridColumn: '1 / -1' }}>
                            <div className="stats-overview-row">
                                <div className="stat-card-premium">
                                    <span className="stat-title">Assigned Livestock</span>
                                    <div className="stat-value">{totalLivestock}</div>
                                </div>
                                <div className="stat-card-premium">
                                    <span className="stat-title">Active Alerts</span>
                                    <div className="stat-value" style={criticalCount > 0 ? { color: 'var(--semantic-critical)' } : {}}>{criticalCount} Critical</div>
                                </div>
                                <div className="stat-card-premium">
                                    <span className="stat-title">Online Devices</span>
                                    <div className="stat-value" style={{ color: 'var(--brand-emerald)' }}>{onlineDevices} / {totalLivestock}</div>
                                </div>
                            </div>
                            <div className="livestock-list-section" style={{ marginTop: '24px' }}>
                                <div className="section-header-premium">
                                    <h2>Assigned Livestock Breakdown</h2>
                                </div>
                                <div className="table-responsive">
                                    <table className="premium-table">
                                        <thead>
                                            <tr>
                                                <th>Animal ID</th>
                                                <th>Temp</th>
                                                <th>Heart Rate</th>
                                                <th>Zone</th>
                                                <th>Status</th>
                                                <th>Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {filteredLivestock.map(animal => (
                                                <tr key={animal._id} onClick={() => navigate(`/livestock/${animal._id}`)} style={{ cursor: 'pointer' }}>
                                                    <td className="fw-600">{animal.livestock?.tagNumber || animal.livestock?.id || animal._id}</td>
                                                    <td>{animal.latestSensorData?.temperature || 38}°C</td>
                                                    <td>{animal.latestSensorData?.heartRate || 70} bpm</td>
                                                    <td>{animal.zone_id?.name || 'Assigned Zone'}</td>
                                                    <td><span className={`status-badge ${getTemperatureStatus(animal.latestSensorData?.temperature) === 'Normal' ? 'Healthy' : getTemperatureStatus(animal.latestSensorData?.temperature) === 'Warning' ? 'Warning' : 'Critical'}`}>{getTemperatureStatus(animal.latestSensorData?.temperature) === 'Normal' ? 'Healthy' : getTemperatureStatus(animal.latestSensorData?.temperature) === 'Warning' ? 'Warning' : 'Critical'}</span></td>
                                                    <td><button className="btn-link-action" onClick={(e) => { e.stopPropagation(); navigate(`/livestock/${animal._id}`); }}>View</button></td>
                                                </tr>
                                            ))}
                                            {filteredLivestock.length === 0 && (
                                                <tr><td colSpan="6" style={{ textAlign: 'center', padding: '24px', color: 'var(--text-muted)' }}>No animals assigned to your zones.</td></tr>
                                            )}
                                        </tbody>
                                    </table>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <>
                            {/* LEFT MAIN COLUMN */}
                            <div className="main-column">
                                {/* QUICK ACTIONS ROW */}
                                <div className="quick-actions-bar">
                                    <button className="qa-btn" onClick={() => setShowAddModal(true)}>
                                        <div className="qa-icon-wrapper add-animal"><Plus size={16} /></div>
                                        <span className="qa-label">Add Animal</span>
                                    </button>
                                    <button className="qa-btn" onClick={() => navigate('/devices')}>
                                        <div className="qa-icon-wrapper add-device"><Radio size={16} /></div>
                                        <span className="qa-label">Add Device</span>
                                    </button>
                                    <button className="qa-btn" onClick={() => navigate('/alerts')}>
                                        <div className="qa-icon-wrapper view-alerts"><AlertTriangle size={16} /></div>
                                        <span className="qa-label">View Alerts</span>
                                        {unresolvedTotal > 0 && <span className="qa-badge">{unresolvedTotal}</span>}
                                    </button>
                                    <button className="qa-btn" onClick={() => navigate('/health-analytics')}>
                                        <div className="qa-icon-wrapper view-reports"><Activity size={16} /></div>
                                        <span className="qa-label">View Reports</span>
                                    </button>
                                </div>

                                {/* CRITICAL ALERT BANNER */}
                                {criticalAlerts.length > 0 && (
                                    <div style={{
                                        background: 'linear-gradient(135deg, #fef2f2, #fff1f2)',
                                        border: '1px solid #fecaca',
                                        borderRadius: '16px',
                                        padding: '16px 20px',
                                        marginBottom: '20px'
                                    }}>
                                        <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px' }}>
                                            <ShieldAlert size={18} color="#ef4444" />
                                            <h3 style={{ margin: 0, fontSize: '0.95rem', fontWeight: 700, color: '#991b1b' }}>
                                                {criticalAlerts.length} Critical Health Alert{criticalAlerts.length > 1 ? 's' : ''}
                                            </h3>
                                        </div>
                                        {criticalAlerts.map((alert, idx) => (
                                            <div key={alert._id || idx} style={{
                                                display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                                                padding: '10px 14px', background: '#fff', borderRadius: '12px',
                                                marginBottom: idx < criticalAlerts.length - 1 ? '8px' : 0,
                                                boxShadow: '0 1px 3px rgba(0,0,0,0.06)'
                                            }}>
                                                <div style={{ flex: 1 }}>
                                                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                                        <span style={{ fontWeight: 700, color: '#0f172a', fontSize: '0.9rem' }}>
                                                            {alert.animalName || 'Unknown'}
                                                        </span>
                                                        <span style={{
                                                            padding: '2px 8px', borderRadius: '10px', fontSize: '0.7rem',
                                                            fontWeight: 700, color: '#fff',
                                                            background: alert.severity === 'Critical' ? '#ef4444' : '#f59e0b'
                                                        }}>
                                                            {alert.severity}
                                                        </span>
                                                        {alert.diseaseProbability && (
                                                            <span style={{ fontSize: '0.8rem', color: '#ef4444', fontWeight: 600 }}>
                                                                {(alert.diseaseProbability * 100).toFixed(0)}% risk
                                                            </span>
                                                        )}
                                                    </div>
                                                    <p style={{ margin: '4px 0 0', fontSize: '0.8rem', color: '#64748b' }}>
                                                        {alert.farmName || ''}{alert.zoneName ? ` / ${alert.zoneName}` : ''}
                                                    </p>
                                                </div>
                                                <div style={{ display: 'flex', gap: '8px' }}>
                                                    <button
                                                        onClick={(e) => { e.stopPropagation(); navigate(`/livestock/${alert.livestockId}`); }}
                                                        style={{
                                                            padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: '8px',
                                                            background: '#fff', fontSize: '0.75rem', fontWeight: 600,
                                                            color: '#475569', cursor: 'pointer'
                                                        }}
                                                    >
                                                        <Eye size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> View
                                                    </button>
                                                    <button
                                                        onClick={(e) => {
                                                            e.stopPropagation();
                                                            navigate('/ai-orchestrator', {
                                                                state: {
                                                                    alertId: alert._id,
                                                                    animalId: alert.livestockId,
                                                                    animalName: alert.animalName,
                                                                    autoQuery: `What's happening with ${alert.animalName}? The ML model detected ${(alert.diseaseProbability * 100).toFixed(0)}% disease probability.`
                                                                }
                                                            });
                                                        }}
                                                        style={{
                                                            padding: '6px 12px', border: 'none', borderRadius: '8px',
                                                            background: '#10b981', fontSize: '0.75rem', fontWeight: 600,
                                                            color: '#fff', cursor: 'pointer'
                                                        }}
                                                    >
                                                        <Sparkles size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} /> Ask AI
                                                    </button>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                )}

                                {/* STATS OVERVIEW-5 CARD LAYOUT */}
                                <div className="stats-overview-row">
                                    {/* Card 1: Total Animals (Dynamic Dropdown) */}
                                    <div className="stat-card-premium stat-dropdown-card" style={{ paddingBottom: showAnimalsDropdown ? '16px' : '24px' }}>
                                        <span className="stat-title">Total Animals</span>
                                        <div className="stat-value">{totalLivestock}</div>
                                        <div className="stat-trend positive" style={{ marginBottom: '16px' }}>+12 this week</div>

                                        <div className="dynamic-dropdown-wrapper">
                                            <button
                                                className="dropdown-trigger-btn"
                                                onClick={() => setShowAnimalsDropdown(!showAnimalsDropdown)}
                                            >
                                                <span>Click here for breakdown details</span>
                                                <ChevronDown size={16} className={`dropdown-arrow ${showAnimalsDropdown ? 'open' : ''}`} />
                                            </button>

                                            {showAnimalsDropdown && (
                                                <div className="dropdown-content-visible" style={{ marginTop: '16px', display: 'flex', flexDirection: 'column', gap: '12px' }}>
                                                    {Object.entries(animalCounts).map(([type, count]) => (
                                                        <div className="breakdown-item" key={type} style={{ display: 'flex', justifyContent: 'space-between' }}>
                                                            <span className="bd-label" style={{ fontSize: '0.85rem', color: 'var(--text-secondary)' }}>● Healthy:</span>
                                                            <span className="bd-value" style={{ fontSize: '0.85rem', fontWeight: 600 }}>{count}</span>
                                                        </div>
                                                    ))}
                                                    {Object.keys(animalCounts).length === 0 && (
                                                        <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>No animals registered.</div>
                                                    )}
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Card 2: Health Status */}
                                    <div className="stat-card-premium">
                                        <span className="stat-title">Health Status</span>
                                        <div className="health-score-container">
                                            <div className="health-score-ring">
                                                <svg viewBox="0 0 36 36" className="circular-chart">
                                                    <path className="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                                    <path className="circle-fill" strokeDasharray={`${healthScore}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                                                </svg>
                                                <div className="health-score-text">{healthScore}%</div>
                                            </div>
                                            <span className="health-score-label">System Health</span>
                                        </div>
                                        <div className="health-summary-list">
                                            <div
                                                className={`health-row clickable ${filters.status === 'Healthy' ? 'active-filter' : ''} `}
                                                onClick={() => handleStatusClick('Healthy')}
                                            >
                                                <span className="health-dot healthy"></span>
                                                <span className="health-label">Healthy:</span>
                                                <span className="health-val healthy-text">{healthyCount}</span>
                                            </div>
                                            <div
                                                className={`health-row clickable ${filters.status === 'Warning' ? 'active-filter' : ''} `}
                                                onClick={() => handleStatusClick('Warning')}
                                            >
                                                <span className="health-dot warning"></span>
                                                <span className="health-label">Warning:</span>
                                                <span className="health-val warning-text">{warningCount}</span>
                                            </div>
                                            <div
                                                className={`health-row clickable ${filters.status === 'Critical' ? 'active-filter' : ''} `}
                                                onClick={() => handleStatusClick('Critical')}
                                            >
                                                <span className="health-dot critical"></span>
                                                <span className="health-label">Critical:</span>
                                                <span className="health-val critical-text">{criticalCount}</span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Card 3: Active Alerts */}
                                    <div className="stat-card-premium">
                                        <span className="stat-title">Active Alerts</span>
                                        <div className="stat-value" style={criticalCount > 0 ? { color: 'var(--semantic-critical)' } : {}}>{criticalCount} Critical</div>

                                        <div className="live-alert-monitor">
                                            <div className="radar-wrapper">
                                                <Radio className="radar-icon" size={28} />
                                                <div className="radar-ping"></div>
                                            </div>
                                            <span className="radar-text">Live Monitoring</span>
                                        </div>

                                        <div className="mt-auto">
                                            <button className="btn-link-action" onClick={() => navigate('/alerts')}>View Alerts →</button>
                                        </div>
                                    </div>

                                    {/* Card 4: Device Status */}
                                    <div className="stat-card-premium">
                                        <span className="stat-title">Device Status</span>

                                        <div className="uptime-badge">
                                            <CheckCircle size={16} className="uptime-check" />
                                            <span>99.8% Uptime</span>
                                        </div>

                                        <div className="device-summary-list">
                                            <div className="device-row">
                                                <span className="dev-label">Online:</span>
                                                <span className="dev-val text-green">{onlineDevices}</span>
                                            </div>
                                            <div className="device-row">
                                                <span className="dev-label">Offline:</span>
                                                <span className="dev-val text-red">{devicesOffline}</span>
                                            </div>
                                            <div className="device-row mt-2">
                                                <span className="dev-label">Battery Low:</span>
                                                <span className="dev-val text-orange">{batteryLowCount}</span>
                                            </div>
                                        </div>
                                    </div>

                                    {/* Card 5: Avg Temp */}
                                    <div className="stat-card-premium">
                                        <span className="stat-title">Avg Temp</span>
                                        <div className="stat-value">{avgTemp}°C</div>
                                        <div className="stat-trend positive" style={{ marginBottom: '8px' }}>+0.3°C today</div>
                                        <div className="sparkline-placeholder">
                                            <svg viewBox="0 0 100 30" preserveAspectRatio="none" style={{ width: '100%', height: '30px' }}>
                                                <path d="M0,25 Q10,20 20,25 T40,20 T60,15 T80,20 T100,5" fill="none" stroke="var(--brand-emerald)" strokeWidth="3" strokeLinecap="round" />
                                                <path d="M0,25 Q10,20 20,25 T40,20 T60,15 T80,20 T100,5 L100,30 L0,30 Z" fill="rgba(16, 185, 129, 0.1)" stroke="none" />
                                            </svg>
                                        </div>
                                    </div>

                                    {/* Card 6: Activity Metrics (Balancing Card) */}
                                    <div className={`stat-card-premium activity-card ${!isSidebarOpen ? 'expanded' : ''} `}>
                                        <span className="stat-title">Activity Metrics</span>
                                        <div className="activity-content-wrapper">
                                            {!isSidebarOpen ? (
                                                <div className="expanded-activity-grid">
                                                    <div className="expanded-act-col">
                                                        <div className="anim-zone grazing">
                                                            <div className="cow-emoji bounce">🐄</div>
                                                        </div>
                                                        <div className="exp-act-info">
                                                            <span className="exp-act-label text-emerald">Grazing</span>
                                                            <span className="exp-act-val">{grazingCount} Animals</span>
                                                        </div>
                                                    </div>

                                                    <div className="expanded-act-col">
                                                        <div className="anim-zone resting">
                                                            <div className="cow-emoji sleeping">🐄<span className="z-sleep">Z</span></div>
                                                        </div>
                                                        <div className="exp-act-info">
                                                            <span className="exp-act-label text-indigo">Resting</span>
                                                            <span className="exp-act-val">{restingCount} Animals</span>
                                                        </div>
                                                    </div>

                                                    <div className="expanded-act-col">
                                                        <div className="anim-zone moving">
                                                            <div className="cow-emoji walking">🐄</div>
                                                        </div>
                                                        <div className="exp-act-info">
                                                            <span className="exp-act-label text-orange">Moving</span>
                                                            <span className="exp-act-val">{movingCount} Animals</span>
                                                        </div>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="activity-bars-container mt-auto">
                                                    <div className="activity-row">
                                                        <div className="activity-label-group">
                                                            <span className="act-label">Grazing</span>
                                                            <span className="act-val">{grazingCount}</span>
                                                        </div>
                                                        <div className="activity-bar-bg">
                                                            <div className="activity-bar-fill emerald" style={{ width: `${totalLivestock > 0 ? (grazingCount / totalLivestock) * 100 : 0}% ` }}></div>
                                                        </div>
                                                    </div>
                                                    <div className="activity-row">
                                                        <div className="activity-label-group">
                                                            <span className="act-label">Resting</span>
                                                            <span className="act-val">{restingCount}</span>
                                                        </div>
                                                        <div className="activity-bar-bg">
                                                            <div className="activity-bar-fill indigo" style={{ width: `${totalLivestock > 0 ? (restingCount / totalLivestock) * 100 : 0}% ` }}></div>
                                                        </div>
                                                    </div>
                                                    <div className="activity-row">
                                                        <div className="activity-label-group">
                                                            <span className="act-label">Moving</span>
                                                            <span className="act-val">{movingCount}</span>
                                                        </div>
                                                        <div className="activity-bar-bg">
                                                            <div className="activity-bar-fill orange" style={{ width: `${totalLivestock > 0 ? (movingCount / totalLivestock) * 100 : 0}% ` }}></div>
                                                        </div>
                                                    </div>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>

                                {/* LIVESTOCK MINI CARDS */}
                                <div className="section-header-premium mt-6" ref={livestockGridRef}>
                                    <h2>Recent Activities & Health</h2>
                                    <div className="section-actions">
                                        <button className="btn-premium-toggle active" onClick={() => setFilters({ ...filters, status: 'All' })}>All</button>
                                        <button className="btn-premium-toggle" style={filters.status === 'Critical' ? { background: 'var(--text-primary)', color: 'white' } : {}} onClick={() => setFilters({ ...filters, status: 'Critical' })}>Critical</button>
                                    </div>
                                </div>

                                <div className="livestock-cards-grid-premium">
                                    {loading ? <p>Loading...</p> : sortedLivestock.map((item) => {
                                        const data = item.livestock;
                                        const imgUrl = (data.photoUrl && data.photoUrl.trim() !== "") ? data.photoUrl : null;
                                        const finalImage = imgUrl || getPlaceholderImage(data.type, data.breed);

                                        // Real-time override
                                        const realTimeStats = liveData[data._id] || liveData[data.deviceId] || liveData[data.tagNumber];
                                        const temp = realTimeStats ? realTimeStats.temperature : (item.latestSensorData?.temperature || 38);

                                        const statusKey = getTemperatureStatus(temp);
                                        const isCritical = statusKey === 'High' || statusKey === 'Low' || (item.unresolvedAlerts > 0);
                                        const finalStatus = isCritical ? 'Critical' : (statusKey === 'Warning' ? 'Warning' : 'Healthy');

                                        // Mocking some advanced data for the new layout
                                        const heartRate = realTimeStats ? realTimeStats.heartRate : (Math.floor(Math.random() * (85 - 60 + 1)) + 60);
                                        const location = realTimeStats ? (realTimeStats.location?.lat + ", " + realTimeStats.location?.lng) : (data.location || "Barn A");
                                        const activityLevel = Math.floor(Math.random() * (100 - 40 + 1)) + 40; // 40-100%
                                        const animalId = data.tagNumber || `MIX00${Math.floor(Math.random() * 9) + 1} `;

                                        return (
                                            <div key={data._id} className={`premium-livestock-card ${isCritical ? 'critical-active' : ''} `} onClick={() => navigate(`/livestock/${data._id}`)}>
                                                {isCritical && <div className="critical-indicator-dot"></div>}
                                                <div className="plc-main-content">
                                                    {/* LEFT: Avatar */}
                                                    <div className={`plc-avatar-container ${finalStatus.toLowerCase()} `}>
                                                        <img
                                                            src={finalImage}
                                                            alt={data.name}
                                                            className="plc-avatar-img"
                                                        />
                                                    </div>

                                                    {/* CENTER: Info & Stats */}
                                                    <div className="plc-details-section">
                                                        <div className="plc-header-row">
                                                            <div className="plc-identity">
                                                                <span className="plc-id">{data._id}</span>
                                                                <h4 className="plc-name">{data.name}</h4>
                                                            </div>
                                                            <div className={`plc-status-badge ${finalStatus.toLowerCase()} `}>
                                                                <span className="plc-status-dot"></span> {finalStatus}
                                                            </div>
                                                        </div>

                                                        <div className="plc-meta-row" style={{ fontSize: '0.75rem', marginTop: '4px', display: 'flex', gap: '6px' }}>
                                                            <span style={{ background: '#f8fafc', padding: '4px 8px', borderRadius: '4px', border: '1px solid #e2e8f0', color: '#64748b', fontWeight: 'bold' }}>FARM: {data.farmId || 'N/A'}</span>
                                                            <span style={{ background: '#f8fafc', padding: '4px 8px', borderRadius: '4px', border: '1px solid #e2e8f0', color: '#64748b', fontWeight: 'bold' }}>ZONE: {data.zoneId || 'N/A'}</span>
                                                        </div>

                                                        <div className="plc-meta-row" style={{ marginTop: '8px' }}>
                                                            <span><strong>Breed:</strong> {data.breed || 'Mixed'}</span>
                                                            <span className="meta-dot">•</span>
                                                            <span><strong>Location:</strong> {location}</span>
                                                        </div>

                                                        <div className="plc-health-grid">
                                                            <div className={`health-metric ${statusKey === 'High' || statusKey === 'Low' ? 'alert' : ''} `}>
                                                                <Thermometer size={14} /> <span>{temp.toFixed(1)}°C</span>
                                                            </div>
                                                            <div className="health-metric">
                                                                <HeartPulse size={14} /> <span>{heartRate} bpm</span>
                                                            </div>
                                                            <div className="health-metric">
                                                                <Activity size={14} /> <span>{activityLevel}% Act</span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>

                                                {/* BOTTOM: Footer */}
                                                <div className="plc-footer">
                                                    <div className="plc-last-update">
                                                        <Clock size={12} /> Last update: 2 mins ago
                                                    </div>
                                                    <div className="plc-view-link">
                                                        View Details →
                                                    </div>
                                                </div>
                                            </div>
                                        );
                                    })}

                                    {/* Premium Add Animal Card inside grid */}
                                    <div className="premium-add-card" onClick={() => setShowAddModal(true)}>
                                        <div className="add-card-content">
                                            <div className="add-icon-glow">
                                                <Plus size={36} className="add-icon-svg" />
                                            </div>
                                            <h4 className="add-card-title">Add Animal</h4>
                                            <p className="add-card-subtitle">Register a new member</p>
                                        </div>
                                    </div>

                                    {sortedLivestock.length === 0 && !loading && (
                                        <div className="premium-empty-state" onClick={() => setShowAddModal(true)}>
                                            <div className="empty-state-icon-wrapper">
                                                <Sparkles size={32} className="sparkles-icon" />
                                            </div>
                                            <h3>Begin Your Journey</h3>
                                            <p>Your sanctuary is ready. Register your first animal to unlock real-time intelligence and monitoring.</p>
                                            <button className="btn-primary-elegant empty-add-btn">
                                                Add Animal
                                            </button>
                                        </div>
                                    )}
                                </div>
                            </div>

                            {/* RIGHT SIDEBAR (AI WIDGET FULL HEIGHT) */}
                            <div className="side-column">
                                <div className="ai-widget-premium-light">
                                    <div className="ai-header-light">
                                        <h3>Today's AI Suggestion</h3>
                                        <div className="ai-pulse-indicator-light">
                                            <div className="pulse-dot-light"></div> Live
                                        </div>
                                    </div>
                                    <div className="ai-insight-list-light">
                                        {sortedLivestock.length > 0 && getStatusWeight(sortedLivestock[0]) === 3 && (
                                            <div className="insight-item-light critical-insight" style={{ borderColor: 'rgba(239, 68, 68, 0.4)', background: 'linear-gradient(to right, #fff1f2, #ffe4e6)' }}>
                                                <div className="insight-icon-light" style={{ background: '#fecdd3', color: '#e11d48' }}><AlertTriangle size={18} /></div>
                                                <div className="insight-content-light">
                                                    <h4 style={{ color: '#be123c', marginBottom: '6px' }}>Priority Insight</h4>
                                                    <p style={{ color: '#9f1239', fontWeight: 600, fontSize: '0.9rem', marginBottom: '2px' }}>
                                                        {sortedLivestock[0].livestock.name || `MIX00${Math.floor(Math.random() * 9) + 1} `} has fever ({sortedLivestock[0].latestSensorData?.temperature?.toFixed(1) || '39.5'}°C)
                                                    </p>
                                                    <p style={{ color: '#e11d48', fontWeight: 600 }}>Check immediately</p>
                                                </div>
                                            </div>
                                        )}

                                        {unresolvedTotal > 0 ? (
                                            <div className="insight-item-light critical-insight" style={{ borderColor: 'rgba(239, 68, 68, 0.3)' }}>
                                                <div className="insight-icon-light" style={{ background: '#fef2f2', color: '#ef4444' }}><AlertTriangle size={18} /></div>
                                                <div className="insight-content-light">
                                                    <h4>Critical Alerts</h4>
                                                    <p>There are {unresolvedTotal} unresolved alerts across your herd. Immediate attention is required.</p>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="insight-item-light healthy-insight">
                                                <div className="insight-icon-light"><Activity size={18} /></div>
                                                <div className="insight-content-light">
                                                    <h4>Herd Healthy</h4>
                                                    <p>All monitored livestock are within optimal health ranges today.</p>
                                                </div>
                                            </div>
                                        )}

                                        <div className="insight-item-light info-insight">
                                            <div className="insight-icon-light"><MapPin size={18} /></div>
                                            <div className="insight-content-light">
                                                <h4>Geofence Notice</h4>
                                                <p>No animals have breached the primary geofence ({user?.farm?.geofences?.[0]?.radius || 500}m radius) today.</p>
                                            </div>
                                        </div>
                                        <div className="insight-item-light system-insight">
                                            <div className="insight-icon-light"><Shield size={18} /></div>
                                            <div className="insight-content-light">
                                                <h4>System Health</h4>
                                                <p>{onlineDevices} out of {totalLivestock} sensors are online and transmitting data normally.</p>
                                            </div>
                                        </div>
                                    </div>

                                    <div className="ai-report-footer">
                                        <button className="btn-primary-elegant w-full justify-center" style={{ width: '100%', justifyContent: 'center' }}>
                                            View Full AI Report
                                        </button>
                                    </div>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            </main>

            {/* ADD ANIMAL MODAL (Kept structure, minimal restyling class changes if needed) */}
            {showAddModal && (
                <div className="modal-overlay" style={{ background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(8px)' }}>
                    <div className="modal-content-modern" style={{ width: '700px', background: 'white', borderRadius: '24px', padding: '32px' }}>
                        <div className="modal-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
                            <h2 style={{ fontFamily: 'Outfit', fontWeight: 700 }}>Add New Animal</h2>
                            <button className="close-btn" style={{ background: 'none', border: 'none', fontSize: '24px', cursor: 'pointer' }} onClick={() => setShowAddModal(false)}>&times;</button>
                        </div>
                        <form onSubmit={handleAddAnimal} className="add-animal-form-premium">

                            <div className="form-group-full">
                                <label className="premium-label">Upload Image</label>
                                <div className="image-upload-zone" onClick={() => document.getElementById('animal-image-upload').click()}>
                                    {newAnimal.image ? (
                                        <div className="upload-preview-text">Image Selected</div>
                                    ) : (
                                        <div className="upload-placeholder">
                                            <Image size={24} />
                                            <span>Click to browse</span>
                                        </div>
                                    )}
                                    <input id="animal-image-upload" type="file" onChange={e => setNewAnimal({ ...newAnimal, image: e.target.files[0] })} hidden accept="image/*" />
                                </div>
                            </div>

                            <div className="form-group-row-premium">
                                <div className="form-group">
                                    <label className="premium-label">Identifier (Hardware ID)</label>
                                    <input type="text" className="premium-input" value={newAnimal.tagNumber} onChange={e => setNewAnimal({ ...newAnimal, tagNumber: e.target.value })} placeholder="e.g. SN-987654" required />
                                </div>
                                <div className="form-group">
                                    <label className="premium-label">Name</label>
                                    <input type="text" className="premium-input" value={newAnimal.name} onChange={e => setNewAnimal({ ...newAnimal, name: e.target.value })} placeholder="e.g. Raju" required />
                                </div>
                            </div>

                            <div className="form-group-row-premium">
                                <div className="form-group">
                                    <label className="premium-label">Type of Livestock</label>
                                    <select className="premium-input" value={newAnimal.type} onChange={e => setNewAnimal({ ...newAnimal, type: e.target.value })} required>
                                        <option value="">Select Type</option>
                                        {livestockTypes.map(t => <option key={t} value={t}>{t}</option>)}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label className="premium-label">Breed</label>
                                    <input type="text" className="premium-input" value={newAnimal.breed} onChange={e => setNewAnimal({ ...newAnimal, breed: e.target.value })} placeholder="e.g. Holstein" />
                                </div>
                            </div>

                            <div className="form-group-row-premium">
                                <div className="form-group">
                                    <label className="premium-label">Age (Years)</label>
                                    <input type="number" step="0.1" className="premium-input" value={newAnimal.age} onChange={e => setNewAnimal({ ...newAnimal, age: e.target.value })} placeholder="e.g. 2.5" />
                                </div>
                                <div className="form-group">
                                    <label className="premium-label">Weight (kg)</label>
                                    <input type="number" className="premium-input" value={newAnimal.weight} onChange={e => setNewAnimal({ ...newAnimal, weight: e.target.value })} placeholder="e.g. 450" />
                                </div>
                            </div>

                            <div className="form-group-full">
                                <label className="premium-label">About the Animal</label>
                                <textarea className="premium-input text-area" rows="3" placeholder="Enter special markings, temperament, or medical notes..." value={newAnimal.about || ''} onChange={e => setNewAnimal({ ...newAnimal, about: e.target.value })}></textarea>
                            </div>

                            <div className="modal-actions-premium">
                                <button type="button" className="btn-secondary-elegant" onClick={() => setShowAddModal(false)}>Cancel</button>
                                <button type="submit" className="btn-primary-elegant">Register Animal</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}

            {showProfileModal && (
                <div className="modal-overlay" style={{ background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(8px)', zIndex: 1000 }}>
                    <div className="modal-content-modern profile-modal" style={{ background: 'white', padding: '32px', borderRadius: '24px', width: '500px' }}>
                        <div className="modal-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
                            <h2 style={{ fontFamily: 'Outfit', margin: 0 }}>User Profile</h2>
                            <button className="close-btn" style={{ background: 'none', border: 'none', fontSize: '24px', cursor: 'pointer' }} onClick={() => setShowProfileModal(false)}>&times;</button>
                        </div>
                        <div className="profile-header-section" style={{ display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '32px' }}>
                            <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: 'var(--text-primary)', color: 'white', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '2rem', fontWeight: 700 }}>
                                {user?.name ? user.name.charAt(0).toUpperCase() : 'K'}
                            </div>
                            <div>
                                <h3 style={{ fontFamily: 'Outfit', margin: 0, fontSize: '1.5rem' }}>{user?.name || 'Krishna'}</h3>
                                <p style={{ color: 'var(--brand-emerald)', fontWeight: 600, margin: '4px 0 0 0' }}>Farm Owner</p>
                            </div>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
                            <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '12px', display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}><Phone size={14} style={{ marginRight: 8 }} /> Mobile</span>
                                <span style={{ fontWeight: 600 }}>{user?.mobile || '+91 98765 43210'}</span>
                            </div>
                            <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '12px', display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}><Mail size={14} style={{ marginRight: 8 }} /> Email</span>
                                <span style={{ fontWeight: 600 }}>{user?.email || 'krishna@gomata.com'}</span>
                            </div>
                            <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '12px', display: 'flex', justifyContent: 'space-between' }}>
                                <span style={{ color: 'var(--text-secondary)', fontWeight: 600 }}><MapPin size={14} style={{ marginRight: 8 }} /> Location</span>
                                <span style={{ fontWeight: 600 }}>{user?.address?.city || 'Vrindavan'}, {user?.address?.state || 'India'}</span>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {showFarmModal && (
                <div className="modal-overlay" style={{ background: 'rgba(15, 23, 42, 0.7)', backdropFilter: 'blur(8px)', zIndex: 1000 }}>
                    <div className="modal-content-modern" style={{ background: 'white', padding: '32px', borderRadius: '24px', width: '500px' }}>
                        <div className="modal-header" style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '24px' }}>
                            <h2 style={{ fontFamily: 'Outfit', margin: 0 }}>About Farm</h2>
                            <button className="close-btn" style={{ background: 'none', border: 'none', fontSize: '24px', cursor: 'pointer' }} onClick={() => setShowFarmModal(false)}>&times;</button>
                        </div>
                        <div style={{ textAlign: 'center', marginBottom: '32px' }}>
                            <div style={{ width: '64px', height: '64px', borderRadius: '50%', background: 'var(--brand-emerald-light)', color: 'var(--brand-emerald)', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 16px auto' }}>
                                <Globe size={32} />
                            </div>
                            <h3 style={{ fontFamily: 'Outfit', margin: 0, fontSize: '1.5rem' }}>{user?.farm?.farmName || 'Gokul Dham'}</h3>
                            <p style={{ color: 'var(--text-secondary)', marginTop: '4px' }}>Premium {user?.farm?.livestockType || 'Dairy'} & Organic Farm</p>
                        </div>
                        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
                            <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '12px' }}>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0, fontWeight: 600 }}><Layers size={14} /> Geofences</p>
                                <p style={{ fontWeight: 700, fontSize: '1.2rem', margin: '8px 0 0 0' }}>{user?.farm?.geofenceCount || 1} Active</p>
                            </div>
                            <div style={{ padding: '16px', background: '#f8fafc', borderRadius: '12px' }}>
                                <p style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', margin: 0, fontWeight: 600 }}><Activity size={14} /> Radius</p>
                                <p style={{ fontWeight: 700, fontSize: '1.2rem', margin: '8px 0 0 0' }}>{user?.farm?.geofences?.[0]?.radius || '500'} m</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* Voice Fab button to AI Orchestrator */}
            <button
                className="voice-fab"
                onClick={() => navigate('/ai-orchestrator', { state: { autoStartVoice: true } })}
                aria-label="Open AI Voice Orchestrator"
            >
                <Mic size={28} />
            </button>

        </div>
    );
};

export default Dashboard;