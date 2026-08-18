import React, { useState, useEffect, useRef } from 'react';
import { MapContainer, TileLayer, Marker, Polygon, Circle, useMapEvents, useMap } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Save, Trash2, X, MapPin, Navigation } from 'lucide-react';

// Custom themed marker icon
const customMarkerIcon = new L.DivIcon({
    className: 'custom-theme-marker',
    html: `<div style="background-color: var(--primary); width: 14px; height: 14px; border-radius: 50%; border: 2px solid white; box-shadow: 0 0 4px rgba(0,0,0,0.4);"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9]
});

// Custom GPS location marker icon
const userLocationIcon = new L.DivIcon({
    className: 'user-location-marker',
    html: `<div style="background-color: var(--accent); width: 16px; height: 16px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 8px rgba(37, 99, 235, 0.6); position: relative;">
            <div style="background-color: var(--accent); opacity: 0.3; width: 30px; height: 30px; border-radius: 50%; position: absolute; top: -10px; left: -10px; animation: pulse 2s infinite;"></div>
          </div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8]
});

// Component to handle map clicks
const MapEvents = ({ locationType, points, setPoints, center, setCenter, radius, setRadius }) => {
    useMapEvents({
        click(e) {
            if (locationType === 'Polygon') {
                setPoints([...points, [e.latlng.lat, e.latlng.lng]]);
            } else if (locationType === 'Point') {
                setCenter([e.latlng.lat, e.latlng.lng]);
            }
        },
        contextmenu(e) { // Right click
            if (locationType === 'Polygon' && points.length > 0) {
                // Remove the last point
                setPoints(points.slice(0, -1));
            }
        }
    });
    return null;
};

// Component to dynamically adjust map view based on geofence or user location
const MapBounds = ({ points, center, userLocation, parentGeofencePoints, parentGeofenceCenter, allZoneGeofencesLists = [] }) => {
    const map = useMap();
    const hasCentered = useRef(false);

    useEffect(() => {
        if (hasCentered.current) return; // Prevent zooming/panning when clicking to draw

        // Find optimal bounds to display
        const allPoints = [];
        if (points && points.length > 0) allPoints.push(...points);
        if (parentGeofencePoints && parentGeofencePoints.length > 0) allPoints.push(...parentGeofencePoints);

        allZoneGeofencesLists.forEach(z => {
            if (z.points) allPoints.push(...z.points);
        });

        if (allPoints.length > 0) {
            const bounds = L.latLngBounds(allPoints);
            map.fitBounds(bounds, { padding: [50, 50] });
            hasCentered.current = true;
        } else if (center) {
            map.flyTo(center, 15);
            hasCentered.current = true;
        } else if (parentGeofenceCenter) {
            map.flyTo(parentGeofenceCenter, 15);
            hasCentered.current = true;
        } else if (userLocation) {
            map.flyTo(userLocation, 16);
            hasCentered.current = true;
        }
    }, [points, center, userLocation, parentGeofencePoints, parentGeofenceCenter, allZoneGeofencesLists, map]);

    // Reset centering flag when unmounting
    useEffect(() => {
        return () => { hasCentered.current = false; };
    }, []);

    return null;
};

const MapSelectionModal = ({ isOpen, onClose, onSave, locationType, initialGeofence, parentGeofence, allZoneGeofences = [], itemName }) => {
    // Polygon state
    const [points, setPoints] = useState([]);
    // Circular state
    const [center, setCenter] = useState(null);
    const [radius, setRadius] = useState(500); // Default 500m

    // GPS State
    const [userLocation, setUserLocation] = useState(null);

    // Parent Geofence specific state mapping for rendering
    const [parentPoints, setParentPoints] = useState([]);
    const [parentCenter, setParentCenter] = useState(null);
    const [parentRadius, setParentRadius] = useState(0);

    // Sibling Zones mappings
    const [siblings, setSiblings] = useState([]);

    // Default center fallback
    const defaultCenter = [20.5937, 78.9629];

    // Stringify array/object props to prevent unnecessary re-renders when parent state updates
    const allZoneGeofencesStr = JSON.stringify(allZoneGeofences || []);
    const parentGeofenceStr = JSON.stringify(parentGeofence || null);
    const initialGeofenceStr = JSON.stringify(initialGeofence || null);

    useEffect(() => {
        if (isOpen) {
            const parsedAllZones = JSON.parse(allZoneGeofencesStr);
            const parsedParent = JSON.parse(parentGeofenceStr);
            const parsedInitial = JSON.parse(initialGeofenceStr);

            // Sibling Zones Parser
            if (parsedAllZones && parsedAllZones.length > 0) {
                const parsedSiblings = parsedAllZones.map(z => {
                    if (z.type === 'Polygon' && z.coordinates && z.coordinates[0]) {
                        return { type: 'Polygon', points: z.coordinates[0].map(c => [c[1], c[0]]) };
                    } else if (z.type === 'Point') {
                        return { type: 'Point', center: [z.coordinates[1], z.coordinates[0]], radius: z.radius || 500 };
                    }
                    return null;
                }).filter(Boolean);
                setSiblings(parsedSiblings);
            } else {
                setSiblings([]);
            }

            // First parse the parentGeofence if provided (used for context mapping Zones inside Farms)
            if (parsedParent) {
                if (parsedParent.type === 'Polygon' && parsedParent.coordinates && parsedParent.coordinates[0]) {
                    const mappedPoly = parsedParent.coordinates[0].map(coord => [coord[1], coord[0]]);
                    setParentPoints(mappedPoly);
                    // Compute polygon centroid for center fallback logic if needed
                    let tLat = 0, tLng = 0;
                    mappedPoly.forEach(c => { tLat += c[0]; tLng += c[1]; });
                    setParentCenter([tLat / mappedPoly.length, tLng / mappedPoly.length]);
                } else if (parsedParent.type === 'Point') {
                    setParentCenter([parsedParent.coordinates[1], parsedParent.coordinates[0]]);
                    setParentRadius(parsedParent.radius || 500);
                }
            } else {
                setParentPoints([]);
                setParentCenter(null);
                setParentRadius(0);
            }

            // Initialize state from existing geofence if present
            if (parsedInitial) {
                if (parsedInitial.type === 'Polygon') {
                    if (parsedInitial.coordinates && parsedInitial.coordinates[0]) {
                        // Remember Leaflet maps use [lat, lng], whereas GeoJSON is [lng, lat]
                        // Exclude the last closing point to allow user to continue editing smoothly
                        const coords = parsedInitial.coordinates[0];
                        const leafletPoints = coords.slice(0, coords.length - 1).map(coord => [coord[1], coord[0]]);
                        setPoints(leafletPoints);
                    }
                } else if (parsedInitial.type === 'Point') {
                    setCenter([parsedInitial.coordinates[1], parsedInitial.coordinates[0]]);
                    setRadius(parsedInitial.radius || 500);
                }
            } else {
                // Brand new mapping
                setPoints([]);
                setCenter(null);
                setRadius(500);

                // Auto geolocate ONLY if we are mapping a brand new top-level Farm (no parent context given)
                if (!parsedParent && navigator.geolocation) {
                    navigator.geolocation.getCurrentPosition(
                        (position) => {
                            setUserLocation([position.coords.latitude, position.coords.longitude]);
                        },
                        (err) => console.log('Geolocation skipped or denied.', err),
                        { enableHighAccuracy: true, timeout: 5000, maximumAge: 0 }
                    );
                }
            }
        }
    }, [isOpen, initialGeofenceStr, parentGeofenceStr, allZoneGeofencesStr]);

    if (!isOpen) return null;

    const handleSave = () => {
        let geofenceData = null;
        if (locationType === 'Polygon') {
            if (points.length < 4) {
                alert("A polygon mapping requires a minimum of 4 points.");
                return;
            }
            // Close the polygon for standard GeoJSON
            const closedPoints = [...points, points[0]];
            const geoJsonCoords = closedPoints.map(p => [p[1], p[0]]);
            geofenceData = {
                type: 'Polygon',
                coordinates: [geoJsonCoords]
            };
        } else if (locationType === 'Point') {
            if (!center) {
                alert("Please click on the map to set the center point.");
                return;
            }
            if (radius <= 0) {
                alert("Radius must be greater than 0.");
                return;
            }
            geofenceData = {
                type: 'Point',
                coordinates: [center[1], center[0]], // [lng, lat]
                radius: Number(radius)
            };
        }
        onSave(geofenceData);
    };

    const handleClear = () => {
        setPoints([]);
        setCenter(null);
    };

    return (
        <div className="modal-overlay" style={{ zIndex: 9999 }} onClick={onClose}>
            <div className="modal-content modal-wizard map-modal" onClick={e => e.stopPropagation()} style={{ width: '90%', maxWidth: '1000px', padding: '0', display: 'flex', flexDirection: 'column', height: '80vh', overflow: 'hidden' }}>
                <div style={{ padding: '20px 30px', borderBottom: '1px solid rgba(255,255,255,0.1)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <div>
                        <h2 className="dribbble-title" style={{ fontSize: '1.5rem', marginBottom: '0.5rem' }}>
                            Map Boundaries: {itemName || 'Area'}
                        </h2>
                        <p className="auth-subtitle" style={{ margin: 0 }}>
                            {locationType === 'Polygon'
                                ? "Click to drop pins and draw edges. Right click on the map to remove the last pin."
                                : "Click to set the center of the circular zone."}
                        </p>
                    </div>
                    <button className="btn-icon" onClick={onClose}><X size={24} color="var(--text-secondary)" /></button>
                </div>

                <div style={{ flex: 1, position: 'relative' }}>
                    <MapContainer center={defaultCenter} zoom={5} style={{ height: '100%', width: '100%' }}>
                        <TileLayer
                            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
                            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                        />
                        <MapEvents
                            locationType={locationType}
                            points={points} setPoints={setPoints}
                            center={center} setCenter={setCenter}
                        />

                        <MapBounds
                            points={points}
                            center={center}
                            userLocation={userLocation}
                            parentGeofencePoints={parentPoints}
                            parentGeofenceCenter={parentCenter}
                        />

                        {/* Parent Context Geofences (View Only - Lighter Opacity) */}
                        {parentPoints && parentPoints.length > 2 && (
                            <Polygon
                                positions={parentPoints}
                                pathOptions={{ color: 'var(--text-secondary)', fillColor: 'var(--bg-light)', fillOpacity: 0.2, dashArray: '5, 10' }}
                            />
                        )}
                        {parentCenter && parentRadius > 0 && (
                            <Circle
                                center={parentCenter}
                                radius={parentRadius}
                                pathOptions={{ color: 'var(--text-secondary)', fillColor: 'var(--bg-light)', fillOpacity: 0.2, dashArray: '5, 10' }}
                            />
                        )}

                        {/* Existing Sibling Zones Context (View Only - Opacity 0.15 Fill / 0.6 Stroke Secondary) */}
                        {siblings.map((sib, i) => (
                            <React.Fragment key={i}>
                                {sib.type === 'Polygon' && sib.points.length > 2 && (
                                    <Polygon positions={sib.points} pathOptions={{ color: '#64748B', fillColor: '#64748B', fillOpacity: 0.15, weight: 2, opacity: 0.6 }} />
                                )}
                                {sib.type === 'Point' && sib.center && sib.radius > 0 && (
                                    <Circle center={sib.center} radius={sib.radius} pathOptions={{ color: '#64748B', fillColor: '#64748B', fillOpacity: 0.15, weight: 2, opacity: 0.6 }} />
                                )}
                            </React.Fragment>
                        ))}

                        {/* Current User Location GPS Marker */}
                        {userLocation && (
                            <Marker position={userLocation} icon={userLocationIcon} />
                        )}

                        {/* Active Mapping Overlays (Opacity 0.25 Fill / 1.0 Stroke Accent/Primary) */}
                        {locationType === 'Polygon' && points.length > 0 && (
                            <>
                                {points.map((p, i) => (
                                    <Marker key={i} position={p} icon={customMarkerIcon} />
                                ))}
                                {points.length > 2 && (
                                    <Polygon positions={points} pathOptions={{ color: 'var(--accent)', fillColor: 'var(--accent)', fillOpacity: 0.25, weight: 3, opacity: 1.0 }} />
                                )}
                            </>
                        )}

                        {locationType === 'Point' && center && (
                            <>
                                <Marker position={center} icon={customMarkerIcon} />
                                <Circle center={center} radius={Number(radius)} pathOptions={{ color: 'var(--accent)', fillColor: 'var(--accent)', fillOpacity: 0.25, weight: 3, opacity: 1.0 }} />
                            </>
                        )}
                    </MapContainer>

                    {/* Controls Overlay */}
                    <div style={{ position: 'absolute', bottom: '20px', left: '20px', right: '20px', zIndex: 1000, display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', pointerEvents: 'none' }}>

                        <div className="card-glass" style={{ pointerEvents: 'auto', padding: '15px', display: 'flex', gap: '15px', alignItems: 'center', margin: 0 }}>
                            {locationType === 'Point' && (
                                <div className="form-group" style={{ margin: 0 }}>
                                    <label style={{ color: 'var(--text-primary)' }}>Radius (meters)</label>
                                    <input
                                        type="number"
                                        value={radius}
                                        onChange={(e) => setRadius(e.target.value)}
                                        style={{ width: '120px' }}
                                        className="wizard-select"
                                        min="1"
                                    />
                                </div>
                            )}
                            <button className="btn-text danger" onClick={handleClear} style={{ display: 'flex', alignItems: 'center', gap: '8px', color: 'var(--semantic-critical)' }}>
                                <Trash2 size={16} /> Clear Map
                            </button>
                        </div>

                        <button className="btn-wizard-next" onClick={handleSave} style={{ pointerEvents: 'auto' }}>
                            <Save size={18} /> Save Geofence
                        </button>

                    </div>
                </div>
            </div>
        </div>
    );
};

export default MapSelectionModal;
