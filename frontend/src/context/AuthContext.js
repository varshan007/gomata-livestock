import React, { createContext, useState, useEffect } from 'react';
import api, { authAPI } from '../services/api';

const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const checkLoggedIn = async () => {
            const token = localStorage.getItem('token');
            if (token) {
                try {
                    // Set default header
                    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
                    const { data } = await api.get('/auth/me');
                    setUser(data);
                } catch (error) {
                    console.error("Session expired or invalid token");
                    localStorage.removeItem('token');
                    delete api.defaults.headers.common['Authorization'];
                }
            }
            setLoading(false);
        };

        checkLoggedIn();
    }, []);

    const login = async (email, password, isStaff = false) => {
        let response;
        if (isStaff) {
            response = await authAPI.staffLogin(email, password);
        } else {
            response = await authAPI.login(email, password);
        }

        const { data } = response;
        localStorage.setItem('token', data.token);
        api.defaults.headers.common['Authorization'] = `Bearer ${data.token}`;
        setUser(data);
        return data;
    };

    const register = async (userData) => {
        const { data } = await authAPI.register(userData);
        localStorage.setItem('token', data.token);
        api.defaults.headers.common['Authorization'] = `Bearer ${data.token}`;
        setUser(data);
        return data;
    };

    const logout = () => {
        localStorage.removeItem('token');
        delete api.defaults.headers.common['Authorization'];
        setUser(null);
    };

    const setupStaffPassword = async (userId, phone, newPassword) => {
        const { data } = await authAPI.staffSetupPassword({ userId, phone, newPassword });
        localStorage.setItem('token', data.token);
        api.defaults.headers.common['Authorization'] = `Bearer ${data.token}`;
        setUser(data);
        return data;
    };

    return (
        <AuthContext.Provider value={{ user, login, register, logout, loading, setupStaffPassword }}>
            {children}
        </AuthContext.Provider>
    );
};

export default AuthContext;
