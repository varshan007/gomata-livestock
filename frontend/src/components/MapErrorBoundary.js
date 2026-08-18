import React from 'react';

class MapErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError(error) {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        console.error("Map Error Boundary Caught:", error, errorInfo);
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="map-error-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#ef4444', background: 'rgba(239, 68, 68, 0.1)', borderRadius: '16px' }}>
                    <span style={{ fontSize: '2rem', marginBottom: '8px' }}>⚠️</span>
                    <span>Map Error: Component Crashed.</span>
                    <span style={{ fontSize: '0.8rem', opacity: 0.8 }}>Unable to render Google Maps.</span>
                </div>
            );
        }

        return this.props.children;
    }
}

export default MapErrorBoundary;
