import React, { useState, useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import './Header.css';

const Header = ({ title }) => {
    const { user, logout } = useContext(AuthContext);
    const navigate = useNavigate();
    const [dropdownOpen, setDropdownOpen] = useState(false);

    const handleLogout = () => {
        logout();
        navigate('/login');
    };

    return (
        <header className="app-header">
            <div className="header-left">
                <button className="logo-btn" onClick={() => navigate('/dashboard')}>
                    GoMata
                </button>
                {title && <span className="page-title">/ {title}</span>}
            </div>

            <div className="header-right">
                <div className="user-menu" onClick={() => setDropdownOpen(!dropdownOpen)}>
                    <div className="avatar-small">
                        {user?.name?.charAt(0).toUpperCase() || 'U'}
                    </div>
                    <span className="user-name">{user?.name}</span>
                    <span className="dropdown-arrow">▼</span>

                    {dropdownOpen && (
                        <div className="dropdown-content">
                            <div className="dropdown-item" onClick={() => navigate('/')}>
                                🏠 Home
                            </div>
                            <div className="dropdown-divider"></div>
                            <div className="dropdown-item" onClick={() => navigate('/profile')}>
                                👤 My Profile
                            </div>
                            <div className="dropdown-item" onClick={() => navigate('/farm')}>
                                🏢 Farm Details
                            </div>
                            <div className="dropdown-item" onClick={() => navigate('/settings')}>
                                ⚙️ Settings
                            </div>
                            <div className="dropdown-divider"></div>
                            <div className="dropdown-item danger" onClick={handleLogout}>
                                🚪 Logout
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </header>
    );
};

export default Header;
