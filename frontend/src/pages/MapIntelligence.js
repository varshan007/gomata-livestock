import React, { useState, useEffect, useRef, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { MapContainer, TileLayer, Polygon, Circle, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import {
    Activity, Map as MapIcon, Sparkles, FileText, Settings, Download,
    Thermometer, CheckCircle, AlertTriangle, AlertCircle, ChevronDown, ChevronRight, LayoutDashboard,
    List, Cpu, Layers, Globe, Shield, LogOut, Users, TrendingUp, BarChart2, ShieldAlert, Link as LinkIcon, Menu, Search, User, Zap, MapPin, RefreshCw, WifiOff, XCircle
} from 'lucide-react';
import AuthContext from '../context/AuthContext';
import './MapIntelligence.css';
import './Dashboard.css';

// Fix for default Leaflet icon paths
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

// Helper for images
const getPlaceholderImage = (breed) => {
    const defaultCowImage = "https://images.unsplash.com/photo-1546445317-29f4545e9d53?auto=format&fit=crop&q=80&w=400&h=400";
    if (!breed) return defaultCowImage;
    const searchString = breed.toLowerCase();
    if (searchString.includes('goat')) {
        return "https://images.unsplash.com/photo-1524024973425-502ddf82f25e?auto=format&fit=crop&q=80&w=400&h=400";
    } else if (searchString.includes('sheep')) {
        return "https://images.unsplash.com/photo-1484557985045-caf43e5e714b?auto=format&fit=crop&q=80&w=400&h=400";
    }
    return defaultCowImage;
};

// Create Custom Avatar Icons
const createAvatarIcon = (color, imgUrl, id, name) => {
    const colorMap = {
        'healthy': '#10b981',
        'warning': '#f59e0b',
        'critical': '#ef4444',
        'offline': '#94a3b8'
    };

    return L.divIcon({
        className: 'custom-map-marker',
        html: `
            <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; transform: translateY(-12px);">
                <div style="background-color: ${colorMap[color]}; width: 36px; height: 36px; border-radius: 50%; border: 3px solid white; box-shadow: 0 4px 6px rgba(0,0,0,0.3); overflow: hidden; display: flex; align-items: center; justify-content: center; position: relative;">
                    <img src="${imgUrl}" alt="animal" style="width: 100%; height: 100%; object-fit: cover;" />
                </div>
                <div style="width: 0; height: 0; border-left: 6px solid transparent; border-right: 6px solid transparent; border-top: 8px solid white; margin: 0 auto; margin-top: -2px; filter: drop-shadow(0 2px 2px rgba(0,0,0,0.3));"></div>
                <div style="margin-top: 4px; background: rgba(255, 255, 255, 0.95); padding: 4px 8px; border-radius: 6px; font-size: 11px; font-weight: 700; color: #0f172a; box-shadow: 0 4px 6px rgba(0,0,0,0.1); white-space: nowrap; text-align: center; border: 1px solid #e2e8f0; font-family: 'Outfit', sans-serif;">
                    ${name} <br/> <span style="font-size: 9px; color: #64748b; font-weight: 600; font-family: 'Inter', sans-serif;">${id}</span>
                </div>
            </div>
        `,
        iconSize: [80, 80],
        iconAnchor: [40, 70],
        popupAnchor: [0, -70]
    });
};

const mapCenter = [28.634064, 77.161358]; // Default Center

function MapIntelligence() {
    const navigate = useNavigate();
    const { user } = useContext(AuthContext);

    // UI States
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [showProfileMenu, setShowProfileMenu] = useState(false);
    const [showProfileModal, setShowProfileModal] = useState(false);
    const [searchTerm, setSearchTerm] = useState('');
    const profileRef = useRef();

    // Map State
    const [mapType, setMapType] = useState('standard');
    const [mapRef, setMapRef] = useState(null);
    const [mapCenterState, setMapCenterState] = useState([28.634064, 77.161358]);

    // Data States
    const [parentFarms, setParentFarms] = useState([]);
    const [farmZones, setFarmZones] = useState([]);
    const [animals, setAnimals] = useState([]);
    const [stats, setStats] = useState({ healthy: 0, critical: 0, warning: 0, offline: 0 });

    const fetchMapData = async () => {
        if (!user) return;
        const token = localStorage.getItem('token');
        if (!token) return;
        try {
            const headers = { Authorization: `Bearer ${token}` };

            const [lsRes, fRes] = await Promise.all([
                fetch('http://localhost:8000/api/livestock', { headers }),
                fetch('http://localhost:8000/api/farms', { headers })
            ]);

            const lsJson = await lsRes.json();
            const fJson = await fRes.json();

            const lsData = lsJson.data || lsJson;
            const fData = fJson.data || fJson;

            // Format Animals
            const formattedAnimals = (Array.isArray(lsData) ? lsData : []).map(l => {
                let status = 'healthy';
                const temp = l.latestSensorData.temperature || 0;
                const battery = l.latestSensorData.battery || 0;

                if (battery < 20 || l.unresolvedAlerts > 2) status = 'offline';
                else if (temp > 39.0 || temp < 38.0) status = 'warning';
                if (temp > 39.5) status = 'critical';

                let coords = null;
                if (l.livestock.coords && l.livestock.coords.length === 2) {
                    // Mongo is [lng, lat], Leaflet wants [lat, lng]
                    coords = [l.livestock.coords[1], l.livestock.coords[0]];
                }

                return {
                    id: l.livestock._id,
                    name: l.livestock.name,
                    breed: l.livestock.breed,
                    status: status,
                    temp: temp ? temp.toFixed(1) : '--',
                    location: l.livestock.location || 'Unknown',
                    lastUpdate: l.latestSensorData.timestamp ? new Date(l.latestSensorData.timestamp).toLocaleTimeString() : 'N/A',
                    coords: coords || [28.634064, 77.161358],
                    img: getPlaceholderImage(l.livestock.breed)
                };
            });
            setAnimals(formattedAnimals);

            if (formattedAnimals.length > 0 && mapCenterState[0] === 28.634064) {
                setTimeout(() => setMapCenterState(formattedAnimals[0].coords), 100);
            }

            // Aggregate Stats
            setStats({
                healthy: formattedAnimals.filter(a => a.status === 'healthy').length,
                critical: formattedAnimals.filter(a => a.status === 'critical').length,
                warning: formattedAnimals.filter(a => a.status === 'warning').length,
                offline: formattedAnimals.filter(a => a.status === 'offline').length
            });

            // Format Zones and Parent Farms
            const formattedZones = [];
            const pFarms = [];

            fData.forEach(f => {
                // Parent Farm
                if (f.geofence?.type === 'Polygon' && f.geofence.coordinates && f.geofence.coordinates.length > 0) {
                    const fBounds = f.geofence.coordinates[0].map(c => [c[1], c[0]]);
                    if (fBounds.length > 0) {
                        pFarms.push({
                            name: f.name,
                            bounds: fBounds,
                            color: '#3b82f6' // Blue tracer
                        });
                    }
                }

                // Child Zones
                if (f.zones) {
                    f.zones.forEach(z => {
                        let bounds = [];
                        let center = null;
                        let radius = 0;
                        if (z.geofence?.type === 'Polygon' && z.geofence.coordinates && z.geofence.coordinates.length > 0) {
                            bounds = z.geofence.coordinates[0].map(c => [c[1], c[0]]);
                        } else if (z.geofence?.type === 'Point' && z.geofence.coordinates && z.geofence.coordinates.length === 2) {
                            center = [z.geofence.coordinates[1], z.geofence.coordinates[0]];
                            radius = z.geofence.radius || 100;
                        }

                        const animalsInZone = formattedAnimals.filter(a => a.location === z.name);
                        let status = 'Stable';
                        let color = '#10b981';

                        if (animalsInZone.some(a => a.status === 'critical')) {
                            status = 'High Risk';
                            color = '#ef4444';
                        } else if (animalsInZone.some(a => a.status === 'warning')) {
                            status = 'Needs Attention';
                            color = '#f59e0b';
                        }

                        if (bounds.length > 0 || center) {
                            formattedZones.push({
                                name: z.name,
                                type: z.geofence?.type,
                                bounds: bounds,
                                center: center,
                                radius: radius,
                                color: color,
                                status: status
                            });
                        }
                    });
                }
            });
            setFarmZones(formattedZones);
            setParentFarms(pFarms);

        } catch (error) {
            console.error('Error fetching Map Intelligence data:', error);
        }
    };

    useEffect(() => {
        fetchMapData();
        const interval = setInterval(fetchMapData, 10000); // 10s auto refresh for live trackers
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
        fetchMapData();
    };

    const focusOnAnimal = (coords) => {
        if (mapRef) {
            mapRef.flyTo(coords, 18, { duration: 1.5 });
        }
    };

    const tileUrls = {
        standard: "https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png",
        satellite: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    };

    const filteredAnimals = animals.filter(a => a.id.toLowerCase().includes(searchTerm.toLowerCase()));

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
                        <div className="nav-item-premium active" onClick={() => navigate('/map')}>
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
                    <div className="nav-item-premium" onClick={() => setShowProfileModal(true)}>
                        <User className="nav-icon" /> <span>Profile</span>
                    </div>
                    <div className="nav-item-premium logout" onClick={() => navigate('/')}>
                        <LogOut className="nav-icon" /> <span>Logout</span>
                    </div>
                </div>
            </aside>

            {/* MAIN CONTENT */}
            <main className="map-intel-main">
                {/* TOP BAR */}
                <header className="topbar-premium" style={{ marginBottom: '0' }}>
                    <div className="greeting-area" style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <div>
                            <h1 style={{ margin: 0, fontSize: '1.8rem', fontWeight: 700, color: '#0f172a' }}>Map Intelligence</h1>
                            <p style={{ margin: 0, fontSize: '0.9rem', color: '#64748b' }}>Monitor animal locations and detect geographic risks in real-time.</p>
                        </div>
                    </div>
                    <div className="topbar-right">
                        <div className="control-btn dropdown">
                            <span>Farm A</span>
                            <ChevronDown size={16} />
                        </div>
                        <div className="control-btn dropdown" onClick={() => setMapType(prev => prev === 'standard' ? 'satellite' : 'standard')}>
                            <span>{mapType === 'standard' ? 'Map View' : 'Satellite'}</span>
                            <ChevronDown size={16} />
                        </div>
                        <button className="control-btn action" onClick={handleRefresh}>
                            <RefreshCw size={16} /> Refresh
                        </button>
                        <div className="profile-pill" ref={profileRef} onClick={(e) => { e.stopPropagation(); setShowProfileMenu(!showProfileMenu); }}>
                            <div className="profile-avatar">{user?.name ? user.name.charAt(0).toUpperCase() : 'K'}</div>
                            <span>Profile<ChevronDown size={14} style={{ marginLeft: 4, opacity: 0.6 }} /></span>
                        </div>
                    </div>
                </header>

                <div className="intel-grid">
                    {/* Left Column (Map + Table) */}
                    <div className="left-column">
                        {/* Map Section */}
                        <div className="map-wrapper">
                            <MapContainer
                                center={mapCenterState}
                                zoom={16}
                                className="map-container"
                                zoomControl={false}
                                ref={setMapRef}
                            >
                                <TileLayer
                                    url={tileUrls[mapType]}
                                    attribution='&copy; OpenStreetMap contributors'
                                />

                                {/* Parent Farms */}
                                {parentFarms.map((farm, idx) => (
                                    <Polygon
                                        key={`farm-${idx}`}
                                        positions={farm.bounds}
                                        pathOptions={{
                                            color: farm.color,
                                            fillColor: 'transparent',
                                            fillOpacity: 0,
                                            weight: 3,
                                            dashArray: '8, 8'
                                        }}
                                    />
                                ))}

                                {/* Farm Zones */}
                                {farmZones.map((zone, idx) => (
                                    zone.type === 'Polygon' ? (
                                        <Polygon
                                            key={`zone-${idx}`}
                                            positions={zone.bounds}
                                            pathOptions={{
                                                color: zone.color,
                                                fillColor: zone.color,
                                                fillOpacity: 0.1,
                                                weight: 2,
                                                dashArray: '5, 10'
                                            }}
                                        />
                                    ) : (
                                        <Circle
                                            key={`zone-${idx}`}
                                            center={zone.center}
                                            radius={zone.radius}
                                            pathOptions={{
                                                color: zone.color,
                                                fillColor: zone.color,
                                                fillOpacity: 0.1,
                                                weight: 2,
                                                dashArray: '5, 10'
                                            }}
                                        />
                                    )
                                ))}

                                {/* Animal Markers */}
                                {animals.map((animal) => (
                                    <Marker
                                        key={animal.id}
                                        position={animal.coords}
                                        icon={createAvatarIcon(animal.status, animal.img, animal.id, animal.name)}
                                    >
                                        <Popup className="custom-popup" closeButton={false}>
                                            <div className={`popup-header ${animal.status}`}>
                                                <h3 className="popup-title">{animal.id}</h3>
                                                <span className="popup-badge">{animal.status.toUpperCase()}</span>
                                            </div>
                                            <div className="popup-body">
                                                <div className="popup-row">
                                                    <span className="popup-lbl">Temperature</span>
                                                    <span className="popup-val">{animal.temp}{animal.temp !== '--' ? '°C' : ''}</span>
                                                </div>
                                                <div className="popup-row">
                                                    <span className="popup-lbl">Location</span>
                                                    <span className="popup-val">{animal.location}</span>
                                                </div>
                                                <div className="popup-row">
                                                    <span className="popup-lbl">Updated</span>
                                                    <span className="popup-val" style={{ fontWeight: 500, color: '#64748b' }}>{animal.lastUpdate}</span>
                                                </div>
                                                <button className="popup-action" onClick={() => navigate(`/livestock/${animal.id}`)}>
                                                    View Details →
                                                </button>
                                            </div>
                                        </Popup>
                                    </Marker>
                                ))}
                            </MapContainer>

                            <div className="custom-map-legend" style={{ position: 'absolute', bottom: '24px', right: '24px', left: 'auto', zIndex: 1000, background: 'linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%)', padding: '16px', borderRadius: '16px', boxShadow: '0 10px 25px rgba(0,0,0,0.1)', border: '1px solid #e2e8f0' }}>
                                <h4 style={{ margin: '0 0 12px 0', fontSize: '0.9rem', color: '#0f172a', fontWeight: 700, fontFamily: "'Outfit', sans-serif" }}>Map Boundaries</h4>

                                <div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '8px', fontSize: '0.85rem', color: '#334155', fontWeight: 600 }}>
                                        <div style={{ width: '24px', height: '0px', borderTop: '2px dashed #3b82f6' }}></div> Parent Farm
                                    </div>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '10px', fontSize: '0.85rem', color: '#334155', fontWeight: 600 }}>
                                        <div style={{ width: '24px', height: '14px', backgroundColor: 'rgba(16, 185, 129, 0.15)', border: '1px dashed #10b981', borderRadius: '3px' }}></div> Active Zone
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Bottom Table Section (Inside Left Column) */}
                        <div className="bottom-table-container">
                            <div className="table-header">
                                <h3>Detailed Tracking</h3>
                                <div className="search-box">
                                    <Search size={16} color="#94a3b8" />
                                    <input
                                        type="text"
                                        placeholder="Search animal ID..."
                                        value={searchTerm}
                                        onChange={(e) => setSearchTerm(e.target.value)}
                                    />
                                </div>
                            </div>
                            <table className="intel-table">
                                <thead>
                                    <tr>
                                        <th>Animal ID</th>
                                        <th>Status</th>
                                        <th>Location</th>
                                        <th>Temperature</th>
                                        <th>Last Update</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {filteredAnimals.map(animal => (
                                        <tr key={animal.id}>
                                            <td>{animal.id}</td>
                                            <td><span className={`badge-status ${animal.status}`}>{animal.status.charAt(0).toUpperCase() + animal.status.slice(1)}</span></td>
                                            <td>{animal.location}</td>
                                            <td>{animal.temp}{animal.temp !== '--' ? '°C' : ''}</td>
                                            <td>{animal.lastUpdate}</td>
                                        </tr>
                                    ))}
                                    {filteredAnimals.length === 0 && (
                                        <tr>
                                            <td colSpan="5" style={{ textAlign: 'center', color: '#94a3b8' }}>No animals found matching search.</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* Intelligence Panel (30%) - Right Column */}
                    <div className="side-panel">
                        <div className="panel-card">
                            <h3><Activity size={20} color="#3b82f6" /> Live Summary</h3>
                            <div className="summary-grid">
                                <div className="sum-item">
                                    <span className="sum-lbl">Healthy</span>
                                    <span className="sum-val val-green">{stats.healthy}</span>
                                </div>
                                <div className="sum-item">
                                    <span className="sum-lbl">Critical</span>
                                    <span className="sum-val val-red">{stats.critical}</span>
                                </div>
                                <div className="sum-item">
                                    <span className="sum-lbl">Warning</span>
                                    <span className="sum-val val-yellow">{stats.warning}</span>
                                </div>
                                <div className="sum-item">
                                    <span className="sum-lbl">Offline</span>
                                    <span className="sum-val val-grey">{stats.offline}</span>
                                </div>
                            </div>
                        </div>

                        <div className="ai-insight-box">
                            <div className="ai-header">
                                <Sparkles size={20} /> AI Geographic Insight
                            </div>
                            <p><strong>Barn A</strong> is showing increased thermal stress patterns. 2 critical animals concentrated in the NE corner.</p>
                            <div className="ai-action">
                                Recommended: Inspect Barn A ventilation
                            </div>
                        </div>

                        <div className="panel-card">
                            <h3><AlertTriangle size={20} color="#ef4444" /> High Risk Animals</h3>
                            <div className="risk-list">
                                {animals.filter(a => a.status === 'critical' || a.status === 'warning').map(animal => (
                                    <div key={animal.id} className={`risk-row ${animal.status}`} onClick={() => focusOnAnimal(animal.coords)}>
                                        <div>
                                            <div className="r-id">{animal.id}</div>
                                            <div className="r-loc"><MapPin size={12} /> {animal.location}</div>
                                        </div>
                                        <ChevronRight size={16} className="r-arrow" />
                                    </div>
                                ))}
                            </div>
                        </div>

                        <div className="panel-card">
                            <h3><Globe size={20} color="#3b82f6" /> Zone Summary</h3>
                            <div className="risk-list">
                                {farmZones.map((zone, idx) => (
                                    <div key={idx} className={`risk-row ${zone.status === 'High Risk' ? 'critical' : 'healthy'}`}>
                                        <div>
                                            <div className="r-id">{zone.name}</div>
                                            <div className="r-loc">{zone.status}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    );
}

export default MapIntelligence;
