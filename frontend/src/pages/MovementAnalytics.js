import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom'; // Fixed import
import { MapContainer, TileLayer, Polyline } from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';
import { livestockAPI, sensorDataAPI, geofenceAPI } from '../services/api'; // Ensure geofenceAPI is exported
import { ArrowLeft, MapPin, Activity, Calendar, Shield, Play } from 'lucide-react';
import LivestockMapMarker from '../components/LivestockMapMarker';
import './MovementAnalytics.css';

const mapContainerStyle = {
    width: '100%',
    height: '600px',
    borderRadius: '16px',
};

const defaultCenter = { lat: 19.0760, lng: 72.8777 }; // Mumbai

const pathOptions = {
    color: '#3b82f6',
    opacity: 0.8,
    weight: 4,
};

function MovementAnalytics() {
    const { id } = useParams();
    const navigate = useNavigate();
    const [livestock, setLivestock] = useState(null);
    const [path, setPath] = useState([]);
    const [analytics, setAnalytics] = useState(null);
    const [geofences, setGeofences] = useState([]);
    const [timeRange, setTimeRange] = useState(24); // hours
    const [loading, setLoading] = useState(true);



    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                const [livestockRes, pathRes, analyticsRes, geofenceRes] = await Promise.all([
                    livestockAPI.getById(id),
                    sensorDataAPI.getPath(id, timeRange),
                    sensorDataAPI.getAnalytics(id, timeRange),
                    geofenceAPI.getAll()
                ]);

                setLivestock(livestockRes.data);

                // Process path for map
                const pathData = pathRes.data.map(p => ({ lat: p.latitude, lng: p.longitude }));
                setPath(pathData);

                setAnalytics(analyticsRes.data);
                setGeofences(geofenceRes.data);

            } catch (error) {
                console.error("Error loading analytics:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchData();
    }, [id, timeRange]);

    const handleAddGeofence = async () => {
        // Simple stub for now - creates a "Safe Zone" at current location
        if (!analytics || !path.length) return;
        const center = path[path.length - 1];

        try {
            const newFence = {
                name: `Safe Zone ${geofences.length + 1}`,
                type: 'Safe',
                shape: 'Circle',
                center: { latitude: center.lat, longitude: center.lng },
                radius: 100, // 100m default
                livestockId: id
            };
            await geofenceAPI.create(newFence);
            // Refresh list
            const res = await geofenceAPI.getAll();
            setGeofences(res.data);
        } catch (err) {
            console.error("Failed to add geofence:", err);
        }
    };

    if (loading || !livestock) return <div className="analytics-loading">Loading Analytics...</div>;

    const center = path.length > 0 ? path[path.length - 1] : defaultCenter;

    return (
        <div className="analytics-page">
            <div className="analytics-header">
                <button className="back-btn" onClick={() => navigate(`/livestock/${id}`)}>
                    <ArrowLeft size={20} /> Back
                </button>
                <div className="header-title">
                    <h1>Movement Analytics: {livestock.name}</h1>
                    <div className="time-filters">
                        <button className={timeRange === 24 ? 'active' : ''} onClick={() => setTimeRange(24)}>24 Hours</button>
                        <button className={timeRange === 168 ? 'active' : ''} onClick={() => setTimeRange(168)}>7 Days</button>
                        <button className={timeRange === 720 ? 'active' : ''} onClick={() => setTimeRange(720)}>30 Days</button>
                    </div>
                </div>
            </div>

            <div className="analytics-grid">
                {/* Map Section */}
                <div className="card map-section">
                    <h2><MapPin size={24} /> Historical Path</h2>
                    <div className="map-wrapper-large">
                        <MapContainer
                            center={center ? [center.lat, center.lng] : [defaultCenter.lat, defaultCenter.lng]}
                            zoom={15}
                            style={mapContainerStyle}
                            zoomControl={false}
                        >
                            <TileLayer
                                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                            />
                            <LivestockMapMarker
                                position={center ? [center.lat, center.lng] : [defaultCenter.lat, defaultCenter.lng]}
                                name={livestock.name}
                                photoUrl={livestock.photoUrl}
                            />
                            {path.length > 1 && <Polyline positions={path} pathOptions={pathOptions} />}
                        </MapContainer>
                    </div>
                </div>

                {/* Stats Sidebar */}
                <div className="stats-sidebar">
                    <div className="card stat-card">
                        <h3>Total Distance</h3>
                        <div className="big-stat">{analytics?.totalDistance} km</div>
                        <div className="sub-stat">Average Speed: {analytics?.avgSpeed || 0} km/h</div>

                        {/* Mini Bar Chart for Distance (Mock for now, would need daily aggregation API) */}
                        <div style={{ marginTop: '20px', height: '100px' }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={[{ name: 'Today', distance: analytics?.totalDistance }]}>
                                    <Bar dataKey="distance" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                                    <Tooltip cursor={{ fill: 'transparent' }} />
                                </BarChart>
                            </ResponsiveContainer>
                        </div>
                    </div>

                    <div className="card stat-card">
                        <h3>Activity Level</h3>
                        <div className="activity-donut-wrapper">
                            <div className="stat-row">
                                <span className="dot active-dot"></span>
                                <span>Active Points</span>
                                <strong>{analytics?.activityBreakdown?.active}</strong>
                            </div>
                            <div className="stat-row">
                                <span className="dot resting-dot"></span>
                                <span>Resting Points</span>
                                <strong>{analytics?.activityBreakdown?.resting}</strong>
                            </div>

                            {/* Calculated percentage */}
                            <div className="activity-summary">
                                {analytics?.activityBreakdown?.active + analytics?.activityBreakdown?.resting > 0 ? (
                                    <div className="progress-bar">
                                        <div
                                            className="progress-fill"
                                            style={{
                                                width: `${(analytics.activityBreakdown.active / (analytics.activityBreakdown.active + analytics.activityBreakdown.resting) * 100)}%`
                                            }}
                                        />
                                    </div>
                                ) : null}
                                <p>{analytics?.activityBreakdown?.active > 0 ? 'Mostly Active' : 'Mostly Resting'}</p>
                            </div>
                        </div>
                    </div>

                    <div className="card stat-card">
                        <h3>Geofences</h3>
                        <div className="geofence-list">
                            {geofences.length > 0 ? (
                                geofences.map(gf => (
                                    <div key={gf._id} className="geofence-item">
                                        <Shield size={16} /> {gf.name} ({gf.type})
                                    </div>
                                ))
                            ) : (
                                <p className="text-muted">No geofences set.</p>
                            )}
                            <button className="add-geofence-btn" onClick={handleAddGeofence}>+ Add Safe Zone (Current Loc)</button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default MovementAnalytics;
