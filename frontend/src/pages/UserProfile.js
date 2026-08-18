import React, { useState, useContext, useEffect } from 'react';
import AuthContext from '../context/AuthContext';
import Header from '../components/Header';
import './Profile.css';

const UserProfile = () => {
    const { user } = useContext(AuthContext);
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        phone: '',
        settings: {
            theme: 'light',
            units: { temperature: 'C', distance: 'km' }
        }
    });
    const [message, setMessage] = useState('');
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (user) {
            setFormData({
                name: user.name || '',
                email: user.email || '',
                phone: user.phone || '',
                settings: user.settings || {
                    theme: 'light',
                    units: { temperature: 'C', distance: 'km' }
                }
            });
        }
    }, [user]);

    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setMessage('');
        try {
            // Mock success for now until backend connection is fully tested
            setTimeout(() => {
                setMessage({ type: 'success', text: 'Profile updated successfully' });
                setLoading(false);
            }, 500);
        } catch (error) {
            setMessage({ type: 'error', text: 'Failed to update profile' });
            setLoading(false);
        }
    };

    return (
        <div className="page-wrapper">
            <Header title="My Profile" />
            <div className="profile-container">
                <div className="profile-card">
                    <div className="profile-header">
                        <div className="avatar-circle">
                            {formData.name.charAt(0).toUpperCase()}
                        </div>
                        <h2>My Profile</h2>
                        <p className="member-since">Member since {new Date(user?.createdAt || Date.now()).toLocaleDateString()}</p>
                    </div>

                    {message && (
                        <div className={`message ${message.type}`}>
                            {message.text}
                        </div>
                    )}

                    <form onSubmit={handleSubmit}>
                        <div className="form-section">
                            <h3>Personal Information</h3>
                            <div className="form-grid">
                                <div className="form-group">
                                    <label>Full Name</label>
                                    <input
                                        name="name"
                                        value={formData.name}
                                        onChange={handleChange}
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Email Address</label>
                                    <input
                                        name="email"
                                        value={formData.email}
                                        disabled
                                        title="Email cannot be changed"
                                    />
                                </div>
                                <div className="form-group">
                                    <label>Phone Number</label>
                                    <input
                                        name="phone"
                                        value={formData.phone}
                                        onChange={handleChange}
                                        placeholder="+91..."
                                    />
                                </div>
                            </div>
                        </div>

                        <div className="form-actions">
                            <button type="submit" className="btn-save" disabled={loading}>
                                {loading ? 'Saving...' : 'Save Changes'}
                            </button>
                        </div>
                    </form>
                </div>
            </div>
        </div>
    );
};

export default UserProfile;
