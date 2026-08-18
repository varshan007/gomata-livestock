import React, { useState, useEffect, useCallback, useRef, useContext } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Polyline, Polygon, Circle, Tooltip as LeafletTooltip } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, ResponsiveContainer, Tooltip, BarChart, Bar, LineChart, Line } from 'recharts';
import { livestockAPI, sensorDataAPI, alertsAPI } from '../services/api';
import { getTemperatureStatus } from '../utils/temperatureAdvisory';
import { ArrowLeft, MapPin, Thermometer, CheckCircle, Loader, Edit, FileText, Footprints, ArrowRight, Battery, Signal, Wifi, Share2, MoreHorizontal, Bell, Activity, ChevronRight, Minimize2, Maximize2, MessageSquare, Download, Map, Clock, Navigation, AlertTriangle, Mic, Image, Paperclip, Sparkles, Send } from 'lucide-react';
import LivestockMapMarker from '../components/LivestockMapMarker';
import MapErrorBoundary from '../components/MapErrorBoundary';
import { useLiveTelemetry } from '../context/LiveTelemetryContext';
import AuthContext from '../context/AuthContext';
import './LivestockDetail.css';

const defaultCenter = { lat: 19.0760, lng: 72.8777 };

function LivestockDetail() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [livestock, setLivestock] = useState(null);
    const [latestData, setLatestData] = useState(null);
    const [history, setHistory] = useState([]);
    const [alerts, setAlerts] = useState([]);
    const [stats, setStats] = useState(null);
    const [analytics, setAnalytics] = useState(null);
    const [aiAdvisory, setAiAdvisory] = useState(null);
    const { user } = useContext(AuthContext);

    // Chat State
    const [chatMessages, setChatMessages] = useState([]);
    const [chatInput, setChatInput] = useState("");
    const [isChatLoading, setIsChatLoading] = useState(false);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const chatEndRef = useRef(null);

    // Health Risk Prediction state
    const [healthPrediction, setHealthPrediction] = useState(null);

    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [isMapDarkMode, setIsMapDarkMode] = useState(false);

    const scrollToBottom = () => {
        chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    const [canAddNote, setCanAddNote] = useState(false);
    useEffect(() => {
        if (user) {
            setCanAddNote(user.type !== 'staff' || ['Manager', 'Operator'].includes(user.role));
        }
    }, [user]);

    useEffect(() => {
        scrollToBottom();
    }, [chatMessages]);

    // ── Health Risk Prediction polling (every 60s) ────────────────
    useEffect(() => {
        const fetchHealthPrediction = async () => {
            try {
                const token = localStorage.getItem('token');
                if (!token || !id) return;
                const res = await fetch(`http://localhost:8000/api/health-prediction/${id}`, {
                    headers: { Authorization: `Bearer ${token}` }
                });
                const json = await res.json();
                const data = json.data || json;
                if (data && (data.diseaseProbability > 0 || data.explanation)) {
                    setHealthPrediction(data);
                }
            } catch (err) {
                console.error('Health prediction fetch error:', err);
            }
        };
        fetchHealthPrediction();
        const interval = setInterval(fetchHealthPrediction, 60000); // 60s
        return () => clearInterval(interval);
    }, [id]);

    // Chart Data
    const chartData = React.useMemo(() => history.map(h => ({
        time: new Date(h.timestamp).getHours() + ':00',
        temp: h.temperature
    })), [history]);

    const handleSendMessage = async (e) => {
        e.preventDefault();
        if (!chatInput.trim()) return;

        const userMsg = { role: 'user', content: chatInput };
        setChatMessages(prev => [...prev, userMsg]);
        setChatInput("");
        setIsChatLoading(true);

        try {
            const res = await livestockAPI.chatWithVet(id, userMsg.content, chatMessages);
            const aiMsg = { role: 'ai', content: res.data.reply };
            setChatMessages(prev => [...prev, aiMsg]);
        } catch (err) {
            console.error(err);
            setChatMessages(prev => [...prev, { role: 'ai', content: "⚠️ Connection error. Please try again." }]);
        } finally {
            setIsChatLoading(false);
        }
    };




    const fetchLivestockDetails = useCallback(async () => {
        try {
            const [livestockRes, latestRes, historyRes, alertsRes, statsRes, analyticsRes] = await Promise.allSettled([
                livestockAPI.getById(id),
                sensorDataAPI.getLatest(id),
                sensorDataAPI.getByLivestock(id, 24),
                alertsAPI.getByLivestock(id),
                sensorDataAPI.getStats(id, 24),
                sensorDataAPI.getAnalytics(id, 24)
            ]);

            if (livestockRes.status === 'fulfilled') {
                const dataObj = livestockRes.value.data;
                const mockNames = ['Aman', 'Raju', 'Himanshu', 'Billy', 'Dolly', 'Sheru', 'Kalu', 'Gauri', 'Nandi', 'Bhoori', 'Basanti', 'Shyam', 'Radha', 'Kishan', 'Heera', 'Moti'];

                // Deterministic hash based on ID to pick a consistent mock name
                const hash = Array.from(id).reduce((sum, char) => sum + char.charCodeAt(0), 0);
                const mockIndex = hash % mockNames.length;

                const baseObj = dataObj.livestock || dataObj;
                const activeName = baseObj.name === 'Aniket' ? mockNames[mockIndex] : baseObj.name;

                setLivestock({
                    ...baseObj,
                    latestSensorData: dataObj.latestSensorData || baseObj.latestSensorData,
                    unresolvedAlerts: dataObj.unresolvedAlerts || 0,
                    name: activeName
                });
            } else {
                console.error('Livestock fetch failed reason:', livestockRes.reason);
                throw new Error(livestockRes.reason?.response?.data?.message || livestockRes.reason?.message || 'Failed to load livestock');
            }

            if (latestRes.status === 'fulfilled') setLatestData(latestRes.value.data);
            if (historyRes.status === 'fulfilled' && Array.isArray(historyRes.value.data)) {
                setHistory(historyRes.value.data.reverse());
            } else {
                setHistory([]);
            }
            if (alertsRes.status === 'fulfilled' && Array.isArray(alertsRes.value.data)) {
                setAlerts(alertsRes.value.data);
            } else {
                setAlerts([]);
            }
            if (statsRes.status === 'fulfilled') setStats(statsRes.value.data);
            if (analyticsRes.status === 'fulfilled') {
                setAnalytics(analyticsRes.value.data);
                setAiAdvisory(prev => {
                    if (analyticsRes.value.data.aiAdvisory && !prev) {
                        return analyticsRes.value.data.aiAdvisory;
                    }
                    return prev;
                });
            }

        } catch (err) {
            console.error('Error:', err);
            setError(err.message);
        } finally {
            setLoading(false);
        }
    }, [id]);

    useEffect(() => {
        fetchLivestockDetails();
        const interval = setInterval(fetchLivestockDetails, 10000);
        return () => clearInterval(interval);
    }, [fetchLivestockDetails]);

    // GoMata AI Scroll Ref
    const aiSectionRef = useRef(null);

    const scrollToAI = () => {
        aiSectionRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // 1. Grab Global Live Data (Must be called before any early returns)
    const { liveData } = useLiveTelemetry();

    if (loading) return <div className="loading-screen"><Loader className="spinner" /></div>;
    if (error || !livestock) return <div>Error loading data: {error}</div>;

    // 2. See if this specific animal has real-time WebSocket data currently streaming
    const realTimeStats = liveData[livestock._id] || liveData[livestock.deviceId] || liveData[livestock.tagNumber];

    // Helper to safely parse GeoJSON points into Leaflet Maps format
    const parseLoc = (locObj) => {
        if (!locObj) return null;
        if (typeof locObj.lat !== 'undefined' && typeof locObj.lng !== 'undefined') return locObj;
        if (locObj.coordinates && locObj.coordinates.length >= 2) {
            return { lat: locObj.coordinates[1], lng: locObj.coordinates[0] };
        }
        return null;
    };

    // 3. Override static database metrics with Live Socket metrics if available
    const temp = realTimeStats ? realTimeStats.temperature : (livestock?.latestSensorData?.temperature || 38.0);
    const heartRate = realTimeStats ? realTimeStats.heartRate : (livestock?.latestSensorData?.heartRate || 80);
    const displayLocation = parseLoc(realTimeStats ? realTimeStats.location : livestock?.lastLocation);

    const tempStatus = getTemperatureStatus(temp);
    const isCritical = tempStatus.status === 'Critical' || tempStatus.status === 'Warning';

    // Live Metadata
    const batteryLevel = livestock?.latestSensorData?.battery || 93;
    const signalStrength = typeof livestock?.latestSensorData?.signalStrength !== 'undefined' ? `${livestock.latestSensorData.signalStrength} dBm` : 'Strong';

    // Convert SQL Timestamp to "X mins ago" approximation
    const calcSync = (ts) => {
        if (!ts) return 'Never';
        const diff = Math.floor((new Date() - new Date(ts)) / 60000);
        return diff < 1 ? 'Just now' : `${diff} min${diff === 1 ? '' : 's'} ago`;
    };
    const lastSync = calcSync(livestock?.latestSensorData?.timestamp) || '2 mins ago';
    const farmLocation = 'Farm A';
    const distanceKm = analytics?.totalDistance || 4.7;
    const restingTime = analytics?.activityBreakdown?.resting || 120;
    const stepCount = 12438;

    // 48-Hour Graph Data mapped dynamically
    const tempGraphData = history && history.length > 0
        ? history.map(h => ({
            time: new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
            temp: parseFloat(h.temperature?.toFixed(1) || 38.0)
        }))
        : [
            { time: '48h', temp: 38.0 },
            { time: '36h', temp: 38.2 },
            { time: '24h', temp: 38.5 },
            { time: '12h', temp: 38.4 },
            { time: '6h', temp: 38.6 },
            { time: 'Now', temp: parseFloat(temp.toFixed(1)) }
        ];

    // Mock Path for history mapping (Needs a future endpoint)
    const pathTrail = [
        displayLocation || defaultCenter
    ];

    // Parse Zone Geometry from API
    const farmZone = livestock?.farmGeofence?.type === 'Polygon'
        ? livestock.farmGeofence.coordinates[0].map(c => ({ lat: c[1], lng: c[0] }))
        : [];

    const localZone = livestock?.zoneGeofence?.type === 'Polygon'
        ? livestock.zoneGeofence.coordinates[0].map(c => [c[1], c[0]])
        : [];
    const localZoneCenter = livestock?.zoneGeofence?.type === 'Point'
        ? { lat: livestock.zoneGeofence.coordinates[1], lng: livestock.zoneGeofence.coordinates[0] }
        : null;
    const localZoneRadius = livestock?.zoneGeofence?.radius || 100;

    const darkMapStyle = [
        { elementType: "geometry", stylers: [{ color: "#242f3e" }] },
        { elementType: "labels.text.stroke", stylers: [{ color: "#242f3e" }] },
        { elementType: "labels.text.fill", stylers: [{ color: "#746855" }] },
        { featureType: "administrative.locality", elementType: "labels.text.fill", stylers: [{ color: "#d59563" }] },
        { featureType: "road", elementType: "geometry", stylers: [{ color: "#38414e" }] },
        { featureType: "road", elementType: "geometry.stroke", stylers: [{ color: "#212a37" }] },
        { featureType: "water", elementType: "geometry", stylers: [{ color: "#17263c" }] },
    ];

    return (
        <div className="agri-dashboard-layout">
            {/* Sidebar (Agricultural Assistant Style) */}
            <nav className="agri-sidebar">
                <div className="sidebar-pill">
                    <div className="nav-item active" onClick={() => navigate('/dashboard')} title="Dashboard">
                        <ArrowLeft size={24} />
                    </div>
                    <div className="nav-item">
                        <FileText size={20} />
                    </div>
                    <div className="nav-item">
                        <Activity size={20} />
                    </div>
                    <div className="nav-item">
                        <Map size={20} />
                    </div>
                    <div className="nav-item">
                        <Sparkles size={20} />
                    </div>
                    <div className="nav-item" style={{ marginTop: 'auto' }}>
                        <var className="profile-img-stub" />
                    </div>
                </div>
            </nav>

            {/* Main Content Area */}
            <main className="agri-main-content">
                {/* Header Row */}
                <header className="agri-header">
                    <div className="header-titles">
                        <span className="subtitle"><Sparkles size={18} color="#10B981" style={{ marginRight: '6px' }} /> GoMata AI</span>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '20px', marginTop: '12px' }}>
                            <img
                                src={livestock.photoUrl || "https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"}
                                alt={livestock.name}
                                style={{ width: '64px', height: '64px', borderRadius: '50%', objectFit: 'cover', border: '3px solid white', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                            />
                            <div>
                                <h1 className="main-title" style={{ margin: 0 }}>{livestock.name} — {livestock.breed}</h1>
                                <p className="sub-info" style={{ marginTop: '4px', color: '#64748b' }}>
                                    Age: {livestock.age || 'N/A'} yrs &nbsp;|&nbsp;
                                    Weight: {livestock.weight || 'N/A'} kg &nbsp;|&nbsp;
                                    Farm: {livestock.farmName || livestock.farmId || 'Unassigned'}
                                </p>
                                <div style={{ display: 'flex', gap: '8px', marginTop: '12px' }}>
                                    <span style={{ fontSize: '0.75rem', background: '#f8fafc', padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: '8px', color: '#475569', fontWeight: '700', letterSpacing: '0.05em' }}>
                                        FARM: {livestock.farmId}
                                    </span>
                                    <span style={{ fontSize: '0.75rem', background: '#f8fafc', padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: '8px', color: '#475569', fontWeight: '700', letterSpacing: '0.05em' }}>
                                        ZONE: {livestock.zoneId || 'N/A'}
                                    </span>
                                    <span style={{ fontSize: '0.75rem', background: '#f8fafc', padding: '6px 12px', border: '1px solid #e2e8f0', borderRadius: '8px', color: '#475569', fontWeight: '700', letterSpacing: '0.05em' }}>
                                        ID: {livestock._id}
                                    </span>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="header-center-status">
                        <div className="live-sync-pill">
                            <Activity size={16} /> Live AI Sync
                        </div>
                    </div>

                    <div className="header-actions">
                        <div className="action-pill">
                            {user?.type !== 'staff' && <span className="icon-btn"><Edit size={24} /></span>}
                            <span className="icon-btn notification"><Bell size={24} /><span className="dot"></span></span>
                            <div className="profile-btn">
                                <img src={livestock.photoUrl || "https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"} alt="Animal" />
                            </div>
                        </div>
                    </div>
                </header>

                <div className="agri-grid">
                    {/* Top Row: Health & Activity */}
                    <div className="agri-top-row">
                        {/* Health Card (Like the Weather card) */}
                        <div className="agri-card white-card health-card">
                            <div className="health-main">
                                <div className="health-temp">
                                    <span className="temp-val">{temp.toFixed(1)}°C</span>
                                    <span className="temp-date">Updated {lastSync}</span>
                                </div>
                                <div className="health-icon">
                                    <Thermometer size={48} color={tempStatus.status === 'CRITICAL' ? '#EF4444' : '#10B981'} />
                                </div>
                            </div>
                            <div className={`metric-card-premium ${heartRate > 95 ? 'critical' : ''}`}>
                                <div className="mc-icon pink">
                                    <Activity size={24} />
                                </div>
                                <div className="mc-info">
                                    <span className="mc-label">Heart Rate</span>
                                    <div className="mc-value">
                                        {heartRate} <span className="mc-unit">bpm</span>
                                    </div>
                                    <div className="mc-trend stable">Normal resting rate</div>
                                </div>
                            </div>
                            <div className="health-metrics-row">
                                <div className="metric-pill">
                                    <CheckCircle size={16} color="#10B981" />
                                    <div className="m-data">
                                        <span className="lbl">Status</span>
                                        <span className="val">{tempStatus.status}</span>
                                    </div>
                                </div>
                                <div className="metric-pill">
                                    <Battery size={16} color="#3b82f6" />
                                    <div className="m-data">
                                        <span className="lbl">Battery</span>
                                        <span className="val">{batteryLevel}%</span>
                                    </div>
                                </div>
                            </div>

                            <div className="health-graph-48h" style={{ marginTop: '24px', height: '120px' }}>
                                <h4 style={{ margin: '0 0 8px 0', fontSize: '0.85rem', color: '#64748b', fontWeight: '600' }}>48-Hour Temperature Trend</h4>
                                <ResponsiveContainer width="100%" height="100%">
                                    <AreaChart data={tempGraphData} margin={{ top: 5, right: 0, left: 0, bottom: 0 }}>
                                        <defs>
                                            <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                                                <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                                <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                            </linearGradient>
                                        </defs>
                                        <Tooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                        <Area type="monotone" dataKey="temp" stroke="#10b981" fillOpacity={1} fill="url(#colorTemp)" />
                                    </AreaChart>
                                </ResponsiveContainer>
                            </div>

                            <div className="health-prediction">
                                <div className="trend-visual">
                                    <div className="trend-bar" style={{ height: '30%' }}></div>
                                    <div className="trend-bar" style={{ height: '40%' }}></div>
                                    <div className="trend-bar" style={{ height: '50%' }}></div>
                                    <div className="trend-bar" style={{ height: '60%' }}></div>
                                    <div className="trend-bar warn" style={{ height: '80%' }}></div>
                                    <div className="trend-bar warn" style={{ height: '90%' }}></div>
                                    <div className="trend-bar" style={{ height: '60%' }}></div>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <strong><Sparkles size={16} color="#10b981" /> 7-Day AI Forecast</strong>
                                    <p>Temperatures rising slightly this weekend. Thermal stress risk remains low.</p>
                                </div>
                            </div>
                        </div>

                        {/* Activity Card (Like the green 450 H card) */}
                        <div className="agri-card green-card activity-card">
                            <div className="act-header">
                                <div className="act-title">
                                    <h2>{distanceKm.toFixed(1)} km</h2>
                                    <p>Distance Today</p>
                                </div>
                                <MoreHorizontal size={24} color="rgba(255,255,255,0.7)" />
                            </div>
                            <div className="act-bar-chart">
                                {/* Visual representation of the rounded bars */}
                                {[40, 70, 30, 85, 50, 65, 90].map((h, i) => (
                                    <div className="bar-container" key={i} data-tooltip={`${((h / 100) * 6).toFixed(1)} km traveled`}>
                                        <div className="bar-bg">
                                            <div className="bar-fill" style={{ height: `${h}%` }}></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="act-labels">
                                <span>Mon</span><span>Tue</span><span>Wed</span><span>Thu</span><span>Fri</span><span>Sat</span><span>Sun</span>
                            </div>
                            <div className="activity-prediction">
                                <div className="trend-visual">
                                    <div className="trend-bar" style={{ height: '60%' }}></div>
                                    <div className="trend-bar" style={{ height: '65%' }}></div>
                                    <div className="trend-bar" style={{ height: '70%' }}></div>
                                    <div className="trend-bar" style={{ height: '70%' }}></div>
                                    <div className="trend-bar" style={{ height: '75%' }}></div>
                                    <div className="trend-bar" style={{ height: '75%' }}></div>
                                    <div className="trend-bar" style={{ height: '80%' }}></div>
                                </div>
                                <div style={{ flex: 1 }}>
                                    <strong><Sparkles size={16} color="#bbf7d0" /> 7-Day AI Forecast</strong>
                                    <p>Activity to hold at ~4.5km/day. Grazing behavior is optimal.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* ── Disease Prediction / Health Risk Section ── */}
                    {healthPrediction && healthPrediction.diseaseProbability > 0 && (
                        <div className="agri-card white-card" style={{
                            marginBottom: '24px',
                            padding: '24px',
                            borderLeft: healthPrediction.severity === 'Critical' ? '4px solid #ef4444' :
                                healthPrediction.severity === 'High' ? '4px solid #f59e0b' : '4px solid #3b82f6'
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '16px' }}>
                                <div>
                                    <h3 style={{ margin: '0 0 4px 0', fontSize: '1.1rem', fontWeight: 700, color: '#0f172a', fontFamily: 'Outfit' }}>
                                        <AlertTriangle size={18} color={healthPrediction.severity === 'Critical' ? '#ef4444' : '#f59e0b'} style={{ marginRight: '8px', verticalAlign: 'middle' }} />
                                        Disease Prediction — Health Risk
                                    </h3>
                                    <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b' }}>
                                        ML Model (XGBoost) • Updated every 60s
                                    </p>
                                </div>
                                <span style={{
                                    padding: '4px 12px',
                                    borderRadius: '20px',
                                    fontSize: '0.75rem',
                                    fontWeight: 700,
                                    textTransform: 'uppercase',
                                    letterSpacing: '0.05em',
                                    color: '#fff',
                                    background: healthPrediction.severity === 'Critical' ? '#ef4444' :
                                        healthPrediction.severity === 'High' ? '#f59e0b' : '#3b82f6'
                                }}>
                                    {healthPrediction.severity}
                                </span>
                            </div>

                            {/* Probability Gauge */}
                            <div style={{ display: 'flex', gap: '24px', alignItems: 'center', marginBottom: '16px' }}>
                                <div style={{ textAlign: 'center' }}>
                                    <div style={{
                                        width: '80px', height: '80px', borderRadius: '50%',
                                        background: `conic-gradient(${healthPrediction.diseaseProbability >= 0.9 ? '#ef4444' : healthPrediction.diseaseProbability >= 0.7 ? '#f59e0b' : '#3b82f6'} ${healthPrediction.diseaseProbability * 360}deg, #f1f5f9 0deg)`,
                                        display: 'flex', alignItems: 'center', justifyContent: 'center'
                                    }}>
                                        <div style={{
                                            width: '64px', height: '64px', borderRadius: '50%', background: '#fff',
                                            display: 'flex', alignItems: 'center', justifyContent: 'center',
                                            fontWeight: 800, fontSize: '1rem', color: '#0f172a', fontFamily: 'Outfit'
                                        }}>
                                            {(healthPrediction.diseaseProbability * 100).toFixed(0)}%
                                        </div>
                                    </div>
                                    <span style={{ fontSize: '0.7rem', color: '#64748b', marginTop: '4px', display: 'block' }}>Disease Prob.</span>
                                </div>

                                <div style={{ flex: 1 }}>
                                    {healthPrediction.currentVitals && (
                                        <div style={{ display: 'flex', gap: '16px', marginBottom: '12px', flexWrap: 'wrap' }}>
                                            {healthPrediction.currentVitals.temperature && (
                                                <div style={{ background: '#fef2f2', padding: '6px 12px', borderRadius: '8px', fontSize: '0.8rem' }}>
                                                    <Thermometer size={14} color="#ef4444" style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                                                    {parseFloat(healthPrediction.currentVitals.temperature).toFixed(1)}°C
                                                </div>
                                            )}
                                            {healthPrediction.currentVitals.heartRate && (
                                                <div style={{ background: '#fdf2f8', padding: '6px 12px', borderRadius: '8px', fontSize: '0.8rem' }}>
                                                    <Activity size={14} color="#ec4899" style={{ verticalAlign: 'middle', marginRight: '4px' }} />
                                                    {Math.round(healthPrediction.currentVitals.heartRate)} bpm
                                                </div>
                                            )}
                                        </div>
                                    )}
                                    <p style={{ margin: 0, fontSize: '0.85rem', color: '#475569', lineHeight: '1.5' }}>
                                        {healthPrediction.explanation
                                            ? healthPrediction.explanation.substring(0, 300) + (healthPrediction.explanation.length > 300 ? '...' : '')
                                            : healthPrediction.severity === 'Critical'
                                                ? 'Critical health risk detected. Immediate veterinary attention recommended.'
                                                : 'Elevated health risk detected. Monitor closely.'}
                                    </p>
                                </div>
                            </div>

                            {/* Full explanation expandable */}
                            {healthPrediction.explanation && healthPrediction.explanation.length > 300 && (
                                <details style={{ marginBottom: '12px' }}>
                                    <summary style={{ cursor: 'pointer', fontSize: '0.8rem', color: '#3b82f6', fontWeight: 600 }}>
                                        Read full AI explanation
                                    </summary>
                                    <div style={{
                                        marginTop: '8px', padding: '12px', background: '#f8fafc',
                                        borderRadius: '12px', fontSize: '0.85rem', color: '#475569',
                                        lineHeight: '1.6', whiteSpace: 'pre-wrap'
                                    }}>
                                        {healthPrediction.explanation}
                                    </div>
                                </details>
                            )}

                            <div style={{ display: 'flex', gap: '12px' }}>
                                <button
                                    onClick={() => navigate('/ai-orchestrator', {
                                        state: {
                                            alertId: healthPrediction.alertId,
                                            animalId: id,
                                            animalName: livestock?.name || healthPrediction.animalName,
                                            autoQuery: `Tell me more about the health alert for ${livestock?.name || healthPrediction.animalName}. What should I do?`
                                        }
                                    })}
                                    style={{
                                        display: 'flex', alignItems: 'center', gap: '8px',
                                        padding: '8px 16px', borderRadius: '10px', border: 'none',
                                        background: 'linear-gradient(135deg, #10b981, #059669)',
                                        color: '#fff', fontWeight: 600, fontSize: '0.85rem',
                                        cursor: 'pointer', fontFamily: 'Outfit'
                                    }}
                                >
                                    <Sparkles size={16} /> Chat with GoMata AI
                                </button>
                            </div>
                        </div>
                    )}

                    {/* Bottom Row: Map */}
                    <div className="agri-map-section">
                        <div className="section-header">
                            <h2>Manage Location</h2>
                            <span className="sub-lbl">| View detailed movement history</span>
                        </div>
                        <div className="agri-map-container">
                            <MapContainer
                                center={displayLocation ? [displayLocation.lat, displayLocation.lng] : [defaultCenter.lat, defaultCenter.lng]}
                                zoom={16}
                                style={{ width: '100%', height: '100%', borderRadius: '28px', zIndex: 1 }}
                                zoomControl={false}
                            >
                                <TileLayer
                                    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                                />
                                <Polyline positions={pathTrail} pathOptions={{ color: "#10B981", weight: 4, opacity: 0.8 }} />
                                {farmZone.length > 0 && (
                                    <Polygon positions={farmZone} pathOptions={{ fillColor: "transparent", color: "#3B82F6", weight: 3, dashArray: "6, 6" }}>
                                        <LeafletTooltip permanent direction="center" className="zone-label-tooltip farm-label">
                                            {livestock?.farmName || "Farm Boundary"}
                                        </LeafletTooltip>
                                    </Polygon>
                                )}
                                {livestock?.allZoneGeofences?.length > 0 ? (
                                    livestock.allZoneGeofences.map((z, idx) => {
                                        const isAssigned = z.name === livestock?.zoneName;
                                        const pathOpts = {
                                            fillColor: isAssigned ? "#10B981" : "#94a3b8",
                                            fillOpacity: isAssigned ? 0.2 : 0.05,
                                            color: isAssigned ? "#10B981" : "#94a3b8",
                                            weight: 2
                                        };
                                        const tooltip = (
                                            <LeafletTooltip permanent direction="center" className={`zone-label-tooltip ${isAssigned ? 'active-zone' : 'inactive-zone'}`}>
                                                {z.name}
                                            </LeafletTooltip>
                                        );

                                        if (z.geofence?.type === 'Polygon' && z.geofence.coordinates && z.geofence.coordinates.length > 0) {
                                            const positions = z.geofence.coordinates[0].map(c => ({ lat: c[1], lng: c[0] }));
                                            return (
                                                <Polygon key={idx} positions={positions} pathOptions={pathOpts}>
                                                    {tooltip}
                                                </Polygon>
                                            );
                                        } else if (z.geofence?.type === 'Point' && z.geofence.coordinates && z.geofence.coordinates.length === 2) {
                                            const center = { lat: z.geofence.coordinates[1], lng: z.geofence.coordinates[0] };
                                            const radius = z.geofence.radius || 100;
                                            return (
                                                <Circle key={idx} center={center} radius={radius} pathOptions={pathOpts}>
                                                    {tooltip}
                                                </Circle>
                                            );
                                        }
                                        return null;
                                    })
                                ) : (
                                    <>
                                        {localZone.length > 0 && (
                                            <Polygon positions={localZone} pathOptions={{ fillColor: "#10B981", fillOpacity: 0.2, color: "#10B981", weight: 2 }}>
                                                <LeafletTooltip permanent direction="center" className="zone-label-tooltip active-zone">
                                                    {livestock?.zoneName || "Assigned Zone"}
                                                </LeafletTooltip>
                                            </Polygon>
                                        )}
                                        {localZoneCenter && (
                                            <Circle center={localZoneCenter} radius={localZoneRadius} pathOptions={{ fillColor: "#10B981", fillOpacity: 0.2, color: "#10B981", weight: 2 }}>
                                                <LeafletTooltip permanent direction="center" className="zone-label-tooltip active-zone">
                                                    {livestock?.zoneName || "Assigned Zone"}
                                                </LeafletTooltip>
                                            </Circle>
                                        )}
                                    </>
                                )}
                                {displayLocation && (
                                    <LivestockMapMarker
                                        position={[displayLocation.lat, displayLocation.lng]}
                                        name={livestock.name}
                                        photoUrl={livestock.photoUrl || "https://images.unsplash.com/photo-1570042225831-d98fa7577f1e?ixlib=rb-4.0.3&auto=format&fit=crop&w=1000&q=80"}
                                    />
                                )}
                            </MapContainer>

                            {/* Floating Map Overlays */}
                            <div className="map-overlay dark-overlay bottom-left">
                                <span className="title">Current Zone</span>
                                <span className="val">{livestock?.zoneName || livestock?.zoneId || 'Unassigned Zone'}</span>
                                <div className="overlay-metrics">
                                    <span className="om"><Navigation size={14} /> {displayLocation ? `${displayLocation.lat.toFixed(4)}, ${displayLocation.lng.toFixed(4)}` : 'N/A'}</span>
                                </div>
                            </div>
                            <div className="map-overlay white-overlay bottom-right">
                                <span className="title">Geofence Status</span>
                                <span className="val" style={{ color: '#10B981' }}>Inside Boundary</span>
                                <div className="overlay-metrics">
                                    <span className="om"><Signal size={14} /> GPS Active</span>
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Right Column: Activities Timeline */}
                    <aside className="agri-timeline-col">
                        <div className="timeline-header">
                            <div className="date-picker-mock">
                                <h2>Today's Activities</h2>
                                <div className="calendar-dots">
                                    <span className="cal-dot"></span>
                                    <span className="cal-dot active"></span>
                                    <span className="cal-dot"></span>
                                </div>
                            </div>
                            <MoreHorizontal size={20} color="#64748b" />
                        </div>
                        <div className="timeline-list">
                            <div className="timeline-item green-outline">
                                <div className="time-lbl">08:00</div>
                                <div className="t-card">
                                    <div className="t-icon"><Footprints size={20} /></div>
                                    <div className="t-details">
                                        <p>Started Morning Grazing</p>
                                        <span>North Pasture</span>
                                    </div>
                                    <ArrowRight size={16} className="arr" />
                                </div>
                            </div>

                            <div className="timeline-item dark-fill">
                                <div className="time-lbl">12:30</div>
                                <div className="t-card">
                                    <div className="t-icon"><Thermometer size={20} /></div>
                                    <div className="t-details">
                                        <p>Temperature Alert (Normal)</p>
                                        <span>38.2°C Recorded</span>
                                    </div>
                                    <ArrowRight size={16} className="arr" />
                                </div>
                            </div>

                            <div className="timeline-item green-outline">
                                <div className="time-lbl">14:00</div>
                                <div className="t-card">
                                    <div className="t-icon"><Activity size={20} /></div>
                                    <div className="t-details">
                                        <p>Resting Period Detected</p>
                                        <span>Under Shade Area</span>
                                    </div>
                                    <ArrowRight size={16} className="arr" />
                                </div>
                            </div>

                            {/* GoMata AI Integration inside timeline block */}
                            <div className="timeline-item ai-fill">
                                <div className="time-lbl">Now</div>
                                <div className="t-card">
                                    <div className="t-icon"><Sparkles size={20} color="#10B981" /></div>
                                    <div className="t-details">
                                        <p style={{ color: '#1e293b' }}>AI Insight</p>
                                        <span style={{ color: '#64748b' }}>{aiAdvisory || 'All metrics are optimal.'}</span>
                                    </div>
                                </div>
                            </div>
                        </div>

                        <button className="chat-btn" onClick={() => navigate('/ai-orchestrator', { state: { autoStartVoice: true } })}>
                            <MessageSquare size={18} /> Chat with GoMata AI
                        </button>

                        <div className="actions-grid-bubbly">
                            <button className="action-btn-bubbly primary">
                                <FileText size={20} /> Generate Report
                            </button>
                            {canAddNote && (
                                <button className="action-btn-bubbly" onClick={() => alert("Add Note Modal (Phase 12 feature)")}>
                                    <Edit size={20} /> Add Note
                                </button>
                            )}
                        </div>

                        <div className="device-card-bubbly">
                            <h3><Wifi size={18} color="#3b82f6" /> Device Information</h3>
                            <div className="device-stats-grid">
                                <div className="device-stat-row">
                                    <span className="stat-l"><Activity size={16} /> Device ID</span>
                                    <span className="stat-v">{livestock.deviceId || 'N/A'}</span>
                                </div>
                                <div className="device-stat-row">
                                    <span className="stat-l"><MoreHorizontal size={16} /> Type</span>
                                    <span className="stat-v">{livestock.deviceType || 'N/A'}</span>
                                </div>
                                <div className="device-stat-row">
                                    <span className="stat-l"><Battery size={16} /> Battery</span>
                                    <span className="stat-v">{batteryLevel}%</span>
                                </div>
                                <div className="device-stat-row">
                                    <span className="stat-l"><Signal size={16} /> Signal</span>
                                    <span className="stat-v">{signalStrength}</span>
                                </div>
                                <div className="device-stat-row">
                                    <span className="stat-l"><Activity size={16} /> Sync</span>
                                    <span className="stat-v">{lastSync}</span>
                                </div>
                            </div>
                        </div>

                        <div className="device-card-bubbly" style={{ marginTop: '16px' }}>
                            <h3><FileText size={18} color="#10B981" /> Veterinary Notes</h3>
                            <div className="device-stats-grid">
                                <div className="device-stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', borderBottom: '1px solid #f1f5f9', paddingBottom: '8px' }}>
                                    <span className="stat-l" style={{ marginBottom: '4px' }}><CheckCircle size={14} /> Vaccination History</span>
                                    <span className="stat-v" style={{ fontWeight: 'normal', color: '#475569', fontSize: '0.85rem' }}>{livestock.vaccinationNotes || 'No records added.'}</span>
                                </div>
                                <div className="device-stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', borderBottom: '1px solid #f1f5f9', paddingBottom: '8px', paddingTop: '8px' }}>
                                    <span className="stat-l" style={{ marginBottom: '4px' }}><Activity size={14} /> Breeding Notes</span>
                                    <span className="stat-v" style={{ fontWeight: 'normal', color: '#475569', fontSize: '0.85rem' }}>{livestock.breedingNotes || 'No records added.'}</span>
                                </div>
                                <div className="device-stat-row" style={{ flexDirection: 'column', alignItems: 'flex-start', paddingTop: '8px' }}>
                                    <span className="stat-l" style={{ marginBottom: '4px' }}><FileText size={14} /> Additional Info</span>
                                    <span className="stat-v" style={{ fontWeight: 'normal', color: '#475569', fontSize: '0.85rem' }}>{livestock.additionalNotes || 'N/A'}</span>
                                </div>
                            </div>
                        </div>

                    </aside>
                </div>
            </main>

            {/* GoMata AI Modal */}
            {isChatOpen && (
                <div className="ai-chat-modal-overlay" onClick={() => setIsChatOpen(false)}>
                    <div className="ai-chat-modal-content" onClick={(e) => e.stopPropagation()}>
                        <div className="ai-chat-header">
                            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                                <Sparkles size={20} color="#10b981" />
                                <h3 style={{ margin: 0, fontSize: '1.1rem', color: '#0f172a', fontFamily: 'Outfit' }}>GoMata AI Assistant</h3>
                            </div>
                            <div className="ai-chat-close" onClick={() => setIsChatOpen(false)} style={{ fontSize: '24px', lineHeight: 1 }}>
                                &times;
                            </div>
                        </div>

                        <div className="ai-chat-body">
                            <div className="msg-bubble ai">
                                <p style={{ margin: '0 0 8px 0' }}>Namaste! 🙏 I'm monitoring <strong>{livestock.name}</strong>.</p>
                                <p style={{ margin: 0 }}>Current vitals look stable, but I noticed a slight drop in activity. How can I assist you today?</p>
                            </div>

                            {chatMessages.map((msg, index) => (
                                <div key={index} className={`msg-bubble ${msg.role}`}>
                                    {msg.content}
                                </div>
                            ))}

                            {isChatLoading && (
                                <div className="msg-bubble ai typing">
                                    <span className="dot"></span><span className="dot"></span><span className="dot"></span>
                                </div>
                            )}
                            <div ref={chatEndRef} />
                        </div>

                        <div className="ai-chat-footer">
                            <input
                                type="text"
                                placeholder="Ask GoMata AI anything..."
                                value={chatInput}
                                onChange={(e) => setChatInput(e.target.value)}
                                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage(e)}
                                disabled={isChatLoading}
                            />
                            <button
                                className="ai-send-btn"
                                onClick={handleSendMessage}
                                disabled={!chatInput.trim() || isChatLoading}
                            >
                                <Send size={18} />
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default LivestockDetail;