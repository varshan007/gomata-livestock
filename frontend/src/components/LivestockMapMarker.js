import React from 'react';
import { Marker } from 'react-leaflet';
import L from 'leaflet';
import './LivestockMapMarker.css';

const LivestockMapMarker = ({ position, name, photoUrl }) => {
    // Generate HTML string for the Leaflet divIcon
    const iconHtml = `
        <div class="custom-cow-marker" style="transform: translate(-50%, -100%);">
            <div class="marker-photo-wrapper">
                ${photoUrl
            ? `<img src="${photoUrl}" alt="${name}" class="marker-photo" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />`
            : ''}
                <div class="marker-placeholder" style="display: ${photoUrl ? 'none' : 'flex'}">
                    🐄
                </div>
            </div>
            <div class="marker-label">${name}</div>
            <div class="marker-arrow"></div>
        </div>
    `;

    const customIcon = L.divIcon({
        html: iconHtml,
        className: '', // Prevents Leaflet's default white square
        iconSize: [0, 0],
        iconAnchor: [0, 0] // Our transform in the HTML handles the anchoring precisely
    });

    return (
        <Marker position={position} icon={customIcon} />
    );
};

export default LivestockMapMarker;
