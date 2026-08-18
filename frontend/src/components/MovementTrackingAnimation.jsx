import React from 'react';
import './CardAnimations.css';

const MovementTrackingAnimation = ({ style }) => {
    return (
        <div className="card-animation-container movement-bg" style={style}>
            <h3 className="card-overlay-title">Movement & Location Tracking</h3>

            <div className="fluid-bg"></div>

            <div className="flow-lines-container">
                <div className="flow-line f1"></div>
                <div className="flow-line f2"></div>
                <div className="flow-line f3"></div>
            </div>

            <div className="tracking-nodes">
                <div className="node n1"><div className="node-glow"></div></div>
                <div className="node n2"><div className="node-glow"></div></div>
                <div className="node n3"><div className="node-glow"></div></div>
                <div className="node n4"><div className="node-glow"></div></div>
                <div className="node n5"><div className="node-glow"></div></div>
            </div>

            <svg className="tracking-svg" viewBox="0 0 400 300" preserveAspectRatio="none">
                <path
                    className="tracking-path"
                    d="M 50 150 C 100 50, 200 250, 350 100"
                    fill="none"
                    stroke="rgba(59, 130, 246, 0.5)"
                    strokeWidth="2"
                    strokeDasharray="10 10"
                />
                <path
                    className="tracking-path p2"
                    d="M 80 250 C 150 200, 250 100, 320 200"
                    fill="none"
                    stroke="rgba(139, 92, 246, 0.5)"
                    strokeWidth="2"
                    strokeDasharray="10 10"
                />
            </svg>
        </div>
    );
};

export default MovementTrackingAnimation;
