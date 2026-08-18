import React, { useContext } from 'react';
import { Navigate } from 'react-router-dom';
import AuthContext from '../context/AuthContext';

const PrivateRoute = ({ children, requireAdmin = false }) => {
    const { user, loading } = useContext(AuthContext);

    if (loading) {
        return <div>Loading...</div>; // Or a spinner component
    }

    if (!user) {
        return <Navigate to="/" />;
    }

    if (requireAdmin && user.type === 'staff') {
        return <Navigate to="/dashboard" />;
    }

    return children;
};

export default PrivateRoute;
