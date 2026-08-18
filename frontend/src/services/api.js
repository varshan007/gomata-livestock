import axios from 'axios';

// Make sure your .env has:
// REACT_APP_API_URL=http://localhost:8000/api

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

const api = axios.create({
    baseURL: API_URL,
    timeout: 60000, // 60s timeout to accommodate Render free tier cold starts
    headers: {
        'Content-Type': 'application/json',
    },
    withCredentials: false, // change to true only if using cookies/sessions
});

// Add a request interceptor to attach the JWT token
api.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(error);
    }
);

// Add a response interceptor to handle the Universal API Response wrapper
api.interceptors.response.use(
    (response) => {
        // Unwrap successResponse: { success: true, data: {...} }
        if (response.data && response.data.success === true && response.data.hasOwnProperty('data')) {
            response.data = response.data.data;
        }
        return response;
    },
    (error) => {
        // Unwrap errorResponse: { success: false, error: { message: '...' } }
        if (error.response && error.response.data && error.response.data.error) {
            error.response.data = {
                ...error.response.data,
                message: error.response.data.error.message || error.response.data.message || 'An error occurred',
                code: error.response.data.error.code
            };
        }
        return Promise.reject(error);
    }
);


// =========================
// Livestock API
// =========================
export const livestockAPI = {
    getAll: () => api.get('/livestock'),
    getById: (id) => api.get(`/livestock/${id}`),
    create: (data) => api.post('/livestock', data),
    update: (id, data) => api.patch(`/livestock/${id}`, data),
    delete: (id) => api.delete(`/livestock/${id}`),
    getLatestData: (id) => api.get(`/livestock/${id}/latest-data`),
    getAiAdvisory: (id) => api.post(`/livestock/${id}/advisory`),
    chatWithVet: (id, message, history) =>
        api.post(`/livestock/${id}/chat`, { message, history }, { timeout: 60000 }),
};


// =========================
// Sensor Data API (Legacy UI Mapped to Single Aggregate Backend)
// =========================
export const sensorDataAPI = {
    getAll: () => api.get('/livestock'),
    getByLivestock: (livestockId, hours = 24) =>
        api.get(`/livestock/${livestockId}/temperature-history`),
    getLatest: (livestockId) =>
        api.get(`/livestock/${livestockId}`),
    getStats: (livestockId, hours = 24) =>
        api.get(`/livestock/${livestockId}/temperature-history`),
    getDashboard: () => api.get('/livestock'),
    getPath: (id, hours = 24) =>
        api.get(`/sensor-data/livestock/${id}/path?hours=${hours}`),
    getAnalytics: (id, hours = 24) =>
        api.get(`/sensor-data/livestock/${id}/analytics?hours=${hours}`),
};

// =========================
// Dashboard Summary API
// =========================
export const dashboardAPI = {
    getSummary: () => api.get('/dashboard/summary')
};


// =========================
// Alerts API
// =========================
export const alertsAPI = {
    getAll: () => api.get('/alerts'),
    getByLivestock: (id) => api.get(`/alerts/livestock/${id}`),
    getUnresolvedCount: () => api.get('/alerts/unresolved/count'),
    resolve: (id) => api.put(`/alerts/${id}/resolve`),
    acknowledge: (id, userId) =>
        api.put(`/alerts/${id}/acknowledge`, { userId }),
};


// =========================
// Geofence API
// =========================
export const geofenceAPI = {
    getAll: () => api.get('/geofences'),
    create: (data) => api.post('/geofences', data),
    delete: (id) => api.delete(`/geofences/${id}`),
};

// =========================
// Farms API
// =========================
export const farmsAPI = {
    getAll: () => api.get('/farms')
};

// =========================
// Staff Management API
// =========================
export const staffAPI = {
    getAll: () => api.get('/staff'),
    create: (data) => api.post('/staff', data),
    update: (id, data) => api.put(`/staff/${id}`, data),
    delete: (id) => api.delete(`/staff/${id}`)
};


// =========================
// AI API
// =========================
export const aiAPI = {
    voiceChat: (query, history, language) =>
        api.post('/ai/chat', { query, history, language }),
};


// =========================
// Auth API
// =========================
export const authAPI = {
    login: (email, password) =>
        api.post('/auth/login', { email, password }),
    staffLogin: (userId, password) =>
        api.post('/staffAuth/login', { userId, password }),
    staffSetupPassword: (data) =>
        api.post('/staffAuth/setup-password', data),
    register: (userData) =>
        api.post('/system/initialize', userData),
    getMe: () =>
        api.get('/auth/me'),
    updateProfile: (data) =>
        api.put('/auth/profile', data),
};


export default api;