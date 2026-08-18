import React, { useState, useContext } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import './Auth.css'; // Shared CSS for auth pages

const Login = () => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [phone, setPhone] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [isStaff, setIsStaff] = useState(false);
    const [isFirstTime, setIsFirstTime] = useState(false);
    const [error, setError] = useState('');
    const { login, setupStaffPassword } = useContext(AuthContext);
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError('');

        if (isStaff && isFirstTime) {
            try {
                await setupStaffPassword(email.trim(), phone.trim(), newPassword.trim());
                navigate('/dashboard');
            } catch (err) {
                console.error("Setup Error:", err);
                setError(err.response?.data?.message || err.message || 'Setup failed. Check console.');
            }
            return;
        }

        try {
            await login(email.trim(), password.trim(), isStaff);
            navigate('/dashboard');
        } catch (err) {
            console.error("Login Error:", err);
            if (isStaff && err.response?.data?.firstTime) {
                setIsFirstTime(true);
                setError('First time login detected. Please setup your password using your registered phone number.');
            } else {
                setError(err.response?.data?.message || err.message || 'Login failed. Check console.');
            }
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-card" style={{ width: '400px', maxWidth: '100%' }}>
                <h2>{isFirstTime ? 'Setup Your Password' : 'Login to GoMata'}</h2>

                {!isFirstTime && (
                    <div style={{ display: 'flex', background: '#f1f5f9', borderRadius: '8px', padding: '4px', marginBottom: '24px' }}>
                        <button
                            type="button"
                            onClick={() => { setIsStaff(false); setError(''); }}
                            style={{ flex: 1, padding: '10px', borderRadius: '6px', border: 'none', background: !isStaff ? '#ffffff' : 'transparent', color: !isStaff ? '#0f172a' : '#64748b', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s', boxShadow: !isStaff ? '0 1px 3px rgba(0,0,0,0.1)' : 'none' }}
                        >
                            Admin Portal
                        </button>
                        <button
                            type="button"
                            onClick={() => { setIsStaff(true); setError(''); }}
                            style={{ flex: 1, padding: '10px', borderRadius: '6px', border: 'none', background: isStaff ? '#ffffff' : 'transparent', color: isStaff ? '#0f172a' : '#64748b', fontWeight: 600, cursor: 'pointer', transition: 'all 0.2s', boxShadow: isStaff ? '0 1px 3px rgba(0,0,0,0.1)' : 'none' }}
                        >
                            Staff Portal
                        </button>
                    </div>
                )}

                {error && <div className="error-message">{error}</div>}

                <form onSubmit={handleSubmit}>
                    {!isFirstTime ? (
                        <>
                            <div className="form-group">
                                <label>{isStaff ? 'Staff ID' : 'Email Address'}</label>
                                <input
                                    type={isStaff ? 'text' : 'email'}
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder={isStaff ? "e.g. name0001@gomata.ai.com" : "admin@domain.com"}
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>Password</label>
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    required
                                />
                            </div>
                            <button type="submit" className="btn-auth">Login Securely</button>

                            {isStaff && (
                                <div style={{ textAlign: 'center', marginTop: '16px' }}>
                                    <button type="button" onClick={() => setIsFirstTime(true)} style={{ background: 'none', border: 'none', color: '#3b82f6', fontWeight: 600, cursor: 'pointer', textDecoration: 'underline' }}>
                                        First time logging in? Setup Password
                                    </button>
                                </div>
                            )}
                        </>
                    ) : (
                        <>
                            <div className="form-group">
                                <label>Staff ID</label>
                                <input type="text" value={email} disabled style={{ background: '#f1f5f9', color: '#64748b' }} />
                            </div>
                            <div className="form-group">
                                <label>Registered Phone Number *</label>
                                <input
                                    type="tel"
                                    value={phone}
                                    onChange={(e) => setPhone(e.target.value)}
                                    placeholder="+1 (555) 000-0000"
                                    required
                                />
                            </div>
                            <div className="form-group">
                                <label>New Password *</label>
                                <input
                                    type="password"
                                    value={newPassword}
                                    onChange={(e) => setNewPassword(e.target.value)}
                                    required
                                />
                            </div>
                            <div style={{ display: 'flex', gap: 10, marginTop: 20 }}>
                                <button type="button" className="btn-auth" style={{ background: '#e2e8f0', color: '#334155' }} onClick={() => { setIsFirstTime(false); setError(''); }}>Cancel</button>
                                <button type="submit" className="btn-auth" style={{ background: '#3b82f6', flex: 2 }}>Secure Account</button>
                            </div>
                        </>
                    )}
                </form>
                {!isFirstTime && (
                    <div className="auth-footer">
                        <p>Don't have an account? <Link to="/register">Sign Up</Link></p>
                    </div>
                )}
            </div>
        </div >
    );
};

export default Login;
