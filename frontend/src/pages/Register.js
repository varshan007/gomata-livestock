import React, { useState, useContext } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import AuthContext from '../context/AuthContext';
import './Auth.css';

const MultiStepRegister = () => {
    const [step, setStep] = useState(1);
    const { register } = useContext(AuthContext);
    const navigate = useNavigate();

    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
        confirmPassword: '',
        phone: '',
        farm: {
            name: '',
            location: { address: '', city: '', state: '', pinCode: '', country: '' },
            size: '',
            type: 'Dairy Farm'
        },
        settings: {
            notifications: { email: true, sms: false }
        }
    });

    const [error, setError] = useState('');

    const handleChange = (e) => {
        const { name, value } = e.target;
        const keys = name.split('.');

        if (keys.length === 3) {
            const [top, mid, bottom] = keys;
            setFormData(prev => ({
                ...prev,
                [top]: {
                    ...prev[top],
                    [mid]: {
                        ...prev[top][mid],
                        [bottom]: value
                    }
                }
            }));
        } else if (keys.length === 2) {
            const [parent, child] = keys;
            setFormData(prev => ({
                ...prev,
                [parent]: {
                    ...prev[parent],
                    [child]: value
                }
            }));
        } else {
            setFormData(prev => ({ ...prev, [name]: value }));
        }
    };

    const nextStep = () => {
        setError(""); // Clear previous errors
        if (step === 1) {
            if (!formData.name.trim()) return setError("Please enter your full name");
            if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(formData.email)) return setError("Please enter a valid email address");
            if (formData.password.length < 6) return setError("Password must be at least 6 characters");
            if (formData.password !== formData.confirmPassword) return setError("Passwords do not match");
        }
        if (step === 2) {
            if (!formData.farm.name.trim()) return setError("Please enter a farm name");
        }

        setStep(prev => prev + 1);
    };

    const prevStep = () => {
        setError("");
        setStep(prev => prev - 1);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError("");

        // Final validation check before submit
        if (formData.password !== formData.confirmPassword) {
            setError("Passwords do not match");
            return;
        }
        try {
            await register(formData);
            navigate('/dashboard');
        } catch (err) {
            console.error("Registration failed:", err);
            const msg = err.response && err.response.data && err.response.data.message
                ? err.response.data.message
                : "Registration failed. Check console for details.";
            setError(msg);
        }
    };

    return (
        <div className="auth-container">
            <div className="auth-card register-card">
                <h2>Create Your Account</h2>
                <div className="progress-bar">
                    <div className={`step ${step >= 1 ? 'active' : ''}`}>1</div>
                    <div className="line"></div>
                    <div className={`step ${step >= 2 ? 'active' : ''}`}>2</div>
                    <div className="line"></div>
                    <div className={`step ${step >= 3 ? 'active' : ''}`}>3</div>
                </div>

                {error && <div className="error-message">{error}</div>}

                <form onSubmit={handleSubmit}>
                    {step === 1 && (
                        <div className="form-step">
                            <h3>Account Basics</h3>
                            <div className="form-group">
                                <label>Full Name</label>
                                <input name="name" value={formData.name} onChange={handleChange} required />
                            </div>
                            <div className="form-group">
                                <label>Email Address</label>
                                <input type="email" name="email" value={formData.email} onChange={handleChange} required />
                            </div>
                            <div className="form-group">
                                <label>Password</label>
                                <input type="password" name="password" value={formData.password} onChange={handleChange} required />
                            </div>
                            <div className="form-group">
                                <label>Confirm Password</label>
                                <input type="password" name="confirmPassword" value={formData.confirmPassword} onChange={handleChange} required />
                            </div>
                            <button type="button" className="btn-auth" onClick={nextStep}>Next: Farm Details</button>
                        </div>
                    )}

                    {step === 2 && (
                        <div className="form-step">
                            <h3>Farm Information</h3>
                            <div className="form-group">
                                <label>Farm Name</label>
                                <input name="farm.name" value={formData.farm.name} onChange={handleChange} required />
                            </div>
                            <div className="form-group">
                                <label>Type</label>
                                <select name="farm.type" value={formData.farm.type} onChange={handleChange}>
                                    <option>Dairy Farm</option>
                                    <option>Cattle Ranch</option>
                                    <option>Mixed Livestock</option>
                                    <option>Other</option>
                                </select>
                            </div>
                            <div className="form-group">
                                <label>City</label>
                                <input name="farm.location.city" value={formData.farm.location.city} onChange={handleChange} />
                            </div>
                            <div className="btn-group">
                                <button type="button" className="btn-secondary" onClick={prevStep}>Back</button>
                                <button type="button" className="btn-auth" onClick={nextStep}>Next: Preferences</button>
                            </div>
                        </div>
                    )}

                    {step === 3 && (
                        <div className="form-step">
                            <h3>Complete Setup</h3>
                            <p>By clicking "Create Account", you agree to our Terms and Privacy Policy.</p>
                            <div className="btn-group">
                                <button type="button" className="btn-secondary" onClick={prevStep}>Back</button>
                                <button type="submit" className="btn-auth btn-success">Create Account</button>
                            </div>
                        </div>
                    )}
                </form>
                <div className="auth-footer">
                    <p>Already have an account? <Link to="/login">Login</Link></p>
                </div>
            </div>
        </div>
    );
};

export default MultiStepRegister;
