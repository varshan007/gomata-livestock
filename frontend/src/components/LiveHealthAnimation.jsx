import React from 'react';
import './CardAnimations.css';

const LiveHealthAnimation = ({ style }) => {
    return (
        <div className="card-animation-container health-bg" style={style}>
            <h3 className="card-overlay-title">Live Health Monitoring</h3>
            <div className="health-grid"></div>

            <svg
                className="health-svg"
                viewBox="0 0 800 400"
                preserveAspectRatio="none"
            >
                <path
                    className="heartbeat-path-bg"
                    d="M 0 200 L 250 200 L 280 120 L 320 320 L 360 80 L 400 250 L 430 200 L 800 200"
                    fill="none"
                    stroke="rgba(16, 185, 129, 0.15)"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
                <path
                    className="heartbeat-path-active"
                    d="M 0 200 L 250 200 L 280 120 L 320 320 L 360 80 L 400 250 L 430 200 L 800 200"
                    fill="none"
                    stroke="#10b981"
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                />
            </svg>
            <div className="health-fade-overlay"></div>
            <div className="health-scanner-line"></div>
        </div>
    );
};

export default LiveHealthAnimation;
