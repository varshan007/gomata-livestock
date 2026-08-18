import React, { useState, useContext, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, ShieldCheck, BarChart2, ChevronRight, Smartphone, Map, Zap, ArrowRight, Play, Check, Calendar, AlertTriangle, X, Key } from 'lucide-react';
import LivestockNetworkBackground from '../components/LivestockNetworkBackground';
import AuthContext from '../context/AuthContext';
import DashboardPreview from '../assets/dashboard-preview.png';
import MapSelectionModal from '../components/MapSelectionModal';
import SpiralHero from '../components/SpiralHero';
import PlatformSection from '../components/PlatformSection';
import GoMataDivider from '../components/GoMataDivider';
import LiveHealthAnimation from '../components/LiveHealthAnimation';
import MovementTrackingAnimation from '../components/MovementTrackingAnimation';
import './LandingV3.css'; // Keep V3 for modal logic styling
import './LandingV4.css'; // V4 for the new page layout
import './LandingHeroDribbble.css'; // Specific overrides for Dribbble Hero
import './LandingRedesign.css'; // The new dark aesthetic
import './LandingFooter.css'; // Platform Plans and Footer styles

const heroSlidesData = [
    { tag: "Monitor", val1: "99%", title1: "Predictive accuracy", text2: "1. Monitor health, track movement, and predict risks in real-time using AI-powered livestock intelligence.", btn1: "Learn more", brand: "GOMATA" },
    { tag: "Tracking", val1: "<1m", title1: "GPS Precision", text2: "2. Track exact geographic coordinates with custom geofencing alerts to guarantee absolute herd containment.", btn1: "View Maps", brand: "TRACK" },
    { tag: "Predict", val1: "14d", title1: "Early Warning", text2: "3. Predict illness and heat stress up to two weeks before visual symptoms appear with AI models.", btn1: "See Data", brand: "PREDICT" },
    { tag: "Protect", val1: "24/7", title1: "Uptime Security", text2: "4. Automated anomaly detection ensures your livestock is protected around the clock against all threats.", btn1: "Secure Farm", brand: "PROTECT" }
];

const Landing = () => {
    const navigate = useNavigate();

    // Modal State
    const [isLoginOpen, setIsLoginOpen] = useState(false);
    const [isRegisterOpen, setIsRegisterOpen] = useState(false);
    const { login, register } = useContext(AuthContext);

    // Login State
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [phone, setPhone] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [isStaff, setIsStaff] = useState(false);
    const [isFirstTime, setIsFirstTime] = useState(false);
    const [error, setError] = useState('');

    // Registration Wizard State
    const [step, setStep] = useState(1);
    const [activeSlide, setActiveSlide] = useState(0);

    // Hero Section Animation State
    const [currentHeroSlide, setCurrentHeroSlide] = useState(0);

    useEffect(() => {
        const slideInterval = setInterval(() => {
            setCurrentHeroSlide(prev => (prev + 1) % 4);
        }, 3000);
        return () => clearInterval(slideInterval);
    }, []);
    const [regData, setRegData] = useState({
        name: '', dob: '', address1: '', address2: '', address3: '',
        pincode: '', city: '', state: '', country: '',
        lat: '', lng: '', mobile: '', email: '', password: '', confirmPassword: '',
        farms: [], zones: [], livestock: []
    });

    // UI temporary state for forms
    const [farmCount, setFarmCount] = useState(1);
    const [zoneCount, setZoneCount] = useState(1);
    const [livestockCount, setLivestockCount] = useState(1);

    // Map Modal State
    const [mapModalOpen, setMapModalOpen] = useState(false);
    const [activeMapItem, setActiveMapItem] = useState({ type: null, tempId: null, index: null });

    const [verification, setVerification] = useState({
        mobileVerified: false, emailVerified: false,
        otpSentMobile: false, otpSentEmail: false,
        captchaVerified: false
    });
    const [showSuccessAnim, setShowSuccessAnim] = useState(false);
    const [showFinalSuccess, setShowFinalSuccess] = useState(false);

    // Agent Verification States
    const [isVerifying, setIsVerifying] = useState(false);
    const [verificationLogs, setVerificationLogs] = useState([]);

    // Custom Toast
    const [toastMessage, setToastMessage] = useState(null);
    const showToast = (msg) => {
        setToastMessage(msg);
        setTimeout(() => setToastMessage(null), 5000); // 5 sec lifespan
    };

    // Auto-advance slider
    useEffect(() => {
        if (!isRegisterOpen) return;
        const interval = setInterval(() => {
            setActiveSlide(prev => (prev + 1) % 3);
        }, 5000);
        return () => clearInterval(interval);
    }, [isRegisterOpen]);

    const leftSlides = [
        {
            title: "Let's setup your Farm Monitoring",
            subtitle: "All-in-one solution for your livestock. Form a new smart farm from scratch or onboard your existing herd.",
            cardLabel: "I barely had to do anything",
            cardText: "Love the experience. Got my farm set up and all necessary details in about a day and I barely had to do anything. Definitely recommend!",
            author: "Catherine Johns"
        },
        {
            title: "Real-time AI Livestock Monitoring",
            subtitle: "Track vital signs, movement patterns, and health metrics 24/7 with our advanced sensory network.",
            cardLabel: "24/7 Health Tracking",
            cardText: "GoMata instantly identified a temperature spike in Cow #402, allowing us to treat her before symptoms worsened.",
            author: "David Miller"
        },
        {
            title: "Autonomous Agentic AI Intelligence",
            subtitle: "AI agents analyze your farm's data, predict outcomes, and suggest immediate actions.",
            cardLabel: "Predictive Insights",
            cardText: "The AI agent automatically scheduled a vet visit and optimized our feed schedule based on the latest health reports. Incredible!",
            author: "Sarah Jenkins"
        }
    ];

    const handlePincodeChange = async (e) => {
        const pin = e.target.value.replace(/\D/g, '');
        setRegData(prev => ({ ...prev, pincode: pin }));

        if (pin.length >= 5) {
            try {
                // First try a robust postal service if it's 6 digits (India)
                if (pin.length === 6) {
                    const zipRes = await fetch(`https://api.zippopotam.us/IN/${pin}`);
                    if (zipRes.ok) {
                        const zipData = await zipRes.json();
                        const place = zipData.places[0];
                        setRegData(prev => ({
                            ...prev,
                            city: place["place name"] || place.state,
                            state: place.state,
                            country: "India",
                            lat: parseFloat(place.latitude).toFixed(4),
                            lng: parseFloat(place.longitude).toFixed(4)
                        }));
                        return; // Successfully got it, skip OpenStreetMap
                    }
                }

                // Fallback to OpenStreetMap
                const response = await fetch(`https://nominatim.openstreetmap.org/search?format=json&postalcode=${pin}&addressdetails=1&limit=1`, {
                    headers: { 'User-Agent': 'GomataFarmApp/1.0' }
                });
                const data = await response.json();

                if (data && data.length > 0) {
                    const place = data[0];
                    const addr = place.address || {};
                    const resolvedCity = addr.city || addr.town || addr.municipality || addr.village || addr.county || addr.state_district || '';
                    setRegData(prev => ({
                        ...prev,
                        city: resolvedCity,
                        state: addr.state || addr.region || '',
                        country: addr.country || '',
                        lat: parseFloat(place.lat).toFixed(4),
                        lng: parseFloat(place.lon).toFixed(4)
                    }));
                } else {
                    setRegData(prev => ({ ...prev, city: '', state: '', country: '', lat: '', lng: '' }));
                }
            } catch (error) {
                console.error("Geocoding failed", error);
            }
        }
    };

    const handleRegChange = (e) => {
        setRegData({ ...regData, [e.target.name]: e.target.value });
    };

    // Real OTP Logic via Backend
    const handleSendOTP = async (type) => {
        const identifier = type === 'mobile' ? regData.mobile : regData.email;
        if (type === 'mobile' && identifier.length < 10) return showToast("Enter valid mobile number");
        if (type === 'email' && !identifier.includes('@')) return showToast("Enter valid email address");

        // Optimistically set UI state to show input box
        setVerification(prev => ({ ...prev, [type === 'mobile' ? 'otpSentMobile' : 'otpSentEmail']: true }));

        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL || '/api'}/auth/send-otp`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier, type })
            });
            const data = await response.json();

            if (!response.ok) {
                showToast(data.message || "Failed to send OTP");
                // Revert UI if failed
                setVerification(prev => ({ ...prev, [type === 'mobile' ? 'otpSentMobile' : 'otpSentEmail']: false }));
            }
        } catch (error) {
            console.error("OTP Error", error);
            showToast("Network error. Could not send OTP.");
            setVerification(prev => ({ ...prev, [type === 'mobile' ? 'otpSentMobile' : 'otpSentEmail']: false }));
        }
    };

    const handleVerifySuccess = async (type, code) => {
        const identifier = type === 'mobile' ? regData.mobile : regData.email;
        try {
            const response = await fetch(`${process.env.REACT_APP_API_URL || '/api'}/auth/verify-otp`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ identifier, code })
            });

            if (response.ok) {
                if (type === 'mobile') setVerification(prev => ({ ...prev, mobileVerified: true }));
                if (type === 'email') setVerification(prev => ({ ...prev, emailVerified: true }));
            } else {
                showToast("Incorrect OTP. Please try again.");
            }
        } catch (error) {
            console.error("OTP Verification Error", error);
        }
    };

    const handleArrayChange = (arrayName, index, field, value) => {
        const newArray = [...regData[arrayName]];
        newArray[index][field] = value;
        setRegData({ ...regData, [arrayName]: newArray });
    };

    const handleOpenMap = (type, index, tempId) => {
        setActiveMapItem({ type, index, tempId });
        setMapModalOpen(true);
    };

    const handleMapSave = (geofenceData) => {
        if (!activeMapItem.type) return;
        const { type, index } = activeMapItem;
        handleArrayChange(type, index, 'geofence', geofenceData);
        setMapModalOpen(false);
    };

    const handleStep2Submit = () => {
        // FLOW: Close Modal -> Show Success Anim (5s) -> Open Step 3
        setIsRegisterOpen(false); // 1. Close Modal
        setTimeout(() => {
            setShowSuccessAnim(true); // 2. Show Animation Trigger
        }, 100);

        setTimeout(() => {
            setShowSuccessAnim(false);
            setStep(3);
            setIsRegisterOpen(true); // 3. Re-open Modal at Step 3
        }, 5000); // 5 seconds duration
    };

    const handleFinalRegister = async () => {
        try {
            const resData = await register({
                name: regData.name,
                email: regData.email,
                password: regData.password,
                phone: regData.mobile,
                dob: regData.dob,
                farm: {
                    location: {
                        address: `${regData.address1}, ${regData.address2}, ${regData.address3}`,
                        city: regData.city,
                        state: regData.state,
                        pinCode: regData.pincode,
                        country: regData.country,
                        coordinates: {
                            latitude: parseFloat(regData.lat),
                            longitude: parseFloat(regData.lng)
                        }
                    }
                },
                farms: regData.farms,
                zones: regData.zones,
                livestock: regData.livestock
            });

            // FLOW: Close Modal -> Show Verification Anim -> Redirect
            setIsRegisterOpen(false);
            setIsVerifying(true);
            setVerificationLogs([]);

            // Multi-Agent Verification Sequence
            setVerificationLogs(["Onboarding Agent stores base data..."]);
            setTimeout(() => setVerificationLogs(p => [...p, `Hardware Agent already receiving telemetry...`]), 1500);
            setTimeout(() => setVerificationLogs(p => [...p, `Data Management Agent verifies device match ✓ (${resData.devices_verified || regData.livestock.length} devices)`]), 3000);
            setTimeout(() => setVerificationLogs(p => [...p, `Generating sequence IDs...`]), 4500);
            setTimeout(() => setVerificationLogs(p => [...p, `Farm ID generated: ${resData.generated_ids?.farms[0] || 'FM-001'}`]), 6000);
            setTimeout(() => setVerificationLogs(p => [...p, `Zone ID generated: ${resData.generated_ids?.zones[0] || 'ZN-001'}`]), 7000);
            setTimeout(() => setVerificationLogs(p => [...p, `Livestock IDs mapped: ${resData.generated_ids?.livestock[0] || 'LS-001'}...`]), 8000);
            setTimeout(() => setVerificationLogs(p => [...p, `Initialization Agent prepares dashboard data ✓`]), 9000);

            setTimeout(() => {
                setIsVerifying(false);
                setShowFinalSuccess(true);
            }, 10500);

            setTimeout(() => {
                navigate('/dashboard');
            }, 14500); // 4 seconds final animation before redirect
        } catch (err) {
            console.error("Registration Error:", err.response?.data || err);
            const errMsg = err.response?.data?.message || err.response?.data?.error || 'Registration failed. Check console for details.';
            setError(errMsg);
            showToast(errMsg);
            setIsRegisterOpen(true); // Re-open on error
        }
    };

    const handleLogin = async (e) => {
        e.preventDefault();
        setError('');

        if (isStaff && isFirstTime) {
            try {
                const { authAPI } = require('../services/api');
                const { data } = await authAPI.staffSetupPassword({ userId: email.trim(), phone: phone.trim(), newPassword: newPassword.trim() });
                localStorage.setItem('token', data.token);
                // Also need to set default headers if we are mimicking login, simpler to just reload to let AuthContext initialize
                window.location.href = '/dashboard';
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

    const openLogin = () => { setIsLoginOpen(true); setIsRegisterOpen(false); setError(''); };
    const openRegister = () => { setIsRegisterOpen(true); setIsLoginOpen(false); setError(''); };
    const closeModals = () => {
        setIsLoginOpen(false);
        setIsRegisterOpen(false);
        setEmail('');
        setPassword('');
        setError('');
        setStep(1); setShowSuccessAnim(false); setShowFinalSuccess(false);
        setRegData({
            name: '', dob: '', address1: '', address2: '', address3: '',
            pincode: '', city: '', state: '', country: '',
            lat: '', lng: '', mobile: '', email: '', password: '', confirmPassword: '',
            farms: [], zones: [], livestock: []
        });
        setVerification({
            mobileVerified: false, emailVerified: false,
            otpSentMobile: false, otpSentEmail: false,
            captchaVerified: false
        });
    };

    return (
        <div className="lr-page">
            <header className="lr-header">
                <div className="lr-logo">
                    <Activity size={24} />
                    <span>GoMata</span>
                </div>
                <div className="lr-links">
                    <span className="lr-links-label">Quick Links</span>
                    <a onClick={() => window.scrollTo(0, 0)}>Home</a>
                    <a href="#featured">Gallery</a>
                    <a href="#services">Work</a>
                    <a onClick={openLogin}>Login / Contact</a>
                </div>
            </header>

            <SpiralHero />

            <GoMataDivider />

            <PlatformSection />

            <section id="featured" className="lr-featured">
                <h1 className="lr-featured-header">Platform in Action©</h1>
                <div className="lr-featured-content">
                    <div className="lr-featured-desc">
                        <p>Real-time intelligence powering autonomous livestock monitoring, predictive health alerts, and autonomous operational decision systems.</p>
                        <button className="lr-see-works" onClick={openRegister}>View Live Platform &rarr;</button>
                    </div>
                    <div className="lr-featured-cards">
                        <div className="lr-card" style={{ marginTop: '40px' }}>
                            <LiveHealthAnimation style={{ aspectRatio: '3/4' }} />
                        </div>
                        <div className="lr-card">
                            <MovementTrackingAnimation style={{ aspectRatio: '4/3' }} />
                            <div className="lr-card-info">
                                <span>Predictive Analytics Engine</span>
                                <span>(02)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section id="services" className="lr-services">
                <h1 className="lr-services-header">Platform Services</h1>
                <div className="lr-services-bar">
                    <span>REAL-TIME</span>
                    <span>PREDICTIVE</span>
                    <span>AUTONOMOUS</span>
                    <span>INTELLIGENT</span>
                </div>
                <div className="lr-services-list">
                    <div className="lr-service-item">
                        <div className="lr-service-num">01</div>
                        <div className="lr-service-title">Real-Time<br />Livestock Monitoring</div>
                        <div className="lr-service-desc">Continuously tracks vitals, movement, and telemetry from every animal in real time.</div>
                    </div>
                    <div className="lr-service-item">
                        <div className="lr-service-num">02</div>
                        <div className="lr-service-title">Predictive<br />Health Intelligence</div>
                        <div className="lr-service-desc">Detects disease risk, fever probability, and behavioral anomalies before symptoms appear.</div>
                    </div>
                    <div className="lr-service-item">
                        <div className="lr-service-num">03</div>
                        <div className="lr-service-title">Autonomous Movement<br />& Location Tracking</div>
                        <div className="lr-service-desc">Analyzes grazing behavior and provides real-time location intelligence.</div>
                    </div>
                    <div className="lr-service-item">
                        <div className="lr-service-num">04</div>
                        <div className="lr-service-title">Device & Infrastructure<br />Intelligence</div>
                        <div className="lr-service-desc">Predicts battery failure, signal loss, and ensures uninterrupted monitoring.</div>
                    </div>
                </div>
            </section>



            <div className="lr-practice-top">
                <span>© EXPERIENCE エクスペリエンス</span>
                <span>(WDX® — 05)</span>
                <span>DIGITAL CRAFT</span>
            </div>

            <section className="lr-practice">
                <div className="lr-practice-main">
                    <h2>Platform Deployment.</h2>
                </div>
                <div className="lr-practice-bar">
                    <span>Creative Collabs</span>
                    <span>Studio</span>
                    <span>Creative Partners</span>
                </div>
                <div className="lr-table">
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Farm Intelligence Deployment</div>
                        <div className="lr-table-col2">2024 — Present</div>
                        <div className="lr-table-col3">Autonomous Monitoring System</div>
                        <div className="lr-table-col4">India</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Livestock Health Prediction Engine</div>
                        <div className="lr-table-col2">2023 — Present</div>
                        <div className="lr-table-col3">Predictive AI System</div>
                        <div className="lr-table-col4">Global</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Device Telemetry Infrastructure</div>
                        <div className="lr-table-col2">2023 — Present</div>
                        <div className="lr-table-col3">IoT Monitoring Network</div>
                        <div className="lr-table-col4">Multi-Region</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Movement Intelligence System</div>
                        <div className="lr-table-col2">2022 — Present</div>
                        <div className="lr-table-col3">Behavioral Analytics Engine</div>
                        <div className="lr-table-col4">Global</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Autonomous Farm Operations Platform</div>
                        <div className="lr-table-col2">2022 — Present</div>
                        <div className="lr-table-col3">Integrated Intelligence Platform</div>
                        <div className="lr-table-col4">Global</div>
                    </div>
                </div>
            </section>

            <section id="plans" className="lr-plans">
                <h1 className="lr-plans-header">Platform Plans.</h1>
                <div className="lr-plans-bar">
                    <span>Options &rarr; Transparent Pricing</span>
                    <span>Transparent &rarr; Scalable Plans</span>
                    <span>Design Packages &rarr; Platform Access</span>
                    <span>Price &rarr; Subscription</span>
                </div>
                <div className="lr-plans-cards">
                    {/* Card 1 */}
                    <div className="lr-plan-card">
                        <div className="lr-plan-price">
                            <span className="lr-price-num">$29</span>
                            <span className="lr-price-duration">/Month</span>
                        </div>
                        <h3 className="lr-plan-title">Starter Monitoring</h3>
                        <p className="lr-plan-desc">Best for small farms starting livestock health and location monitoring.</p>
                        <div className="lr-plan-divider"></div>
                        <ul className="lr-plan-features">
                            <li><span className="check">&#10003;</span> Monitor up to 25 animals</li>
                            <li><span className="check">&#10003;</span> Real-time health telemetry</li>
                            <li><span className="check">&#10003;</span> Basic movement tracking</li>
                            <li><span className="check">&#10003;</span> Mobile dashboard access</li>
                            <li><span className="check">&#10003;</span> Alert notifications</li>
                            <li><span className="check">&#10003;</span> Device battery monitoring</li>
                        </ul>
                    </div>

                    {/* Card 2 */}
                    <div className="lr-plan-card highlight">
                        <div className="lr-plan-price">
                            <span className="lr-price-num">$99</span>
                            <span className="lr-price-duration">/Month</span>
                        </div>
                        <h3 className="lr-plan-title">Intelligence Plan</h3>
                        <p className="lr-plan-desc">Advanced predictive intelligence and anomaly detection for growing farms.</p>
                        <div className="lr-plan-divider"></div>
                        <ul className="lr-plan-features">
                            <li><span className="check">&#10003;</span> Monitor up to 150 animals</li>
                            <li><span className="check">&#10003;</span> Predictive health forecasting</li>
                            <li><span className="check">&#10003;</span> Disease risk detection</li>
                            <li><span className="check">&#10003;</span> Movement analytics engine</li>
                            <li><span className="check">&#10003;</span> Device intelligence monitoring</li>
                            <li><span className="check">&#10003;</span> Staff access control</li>
                            <li><span className="check">&#10003;</span> AI-powered alerts</li>
                        </ul>
                    </div>

                    {/* Card 3 */}
                    <div className="lr-plan-card">
                        <div className="lr-plan-price">
                            <span className="lr-price-num">$299</span>
                            <span className="lr-price-duration">/Month</span>
                        </div>
                        <h3 className="lr-plan-title">Autonomous Operations</h3>
                        <p className="lr-plan-desc">Full autonomous livestock intelligence and operational automation.</p>
                        <div className="lr-plan-divider"></div>
                        <ul className="lr-plan-features">
                            <li><span className="check">&#10003;</span> Unlimited animal monitoring</li>
                            <li><span className="check">&#10003;</span> Autonomous AI intelligence engine</li>
                            <li><span className="check">&#10003;</span> Breeding intelligence insights</li>
                            <li><span className="check">&#10003;</span> Production intelligence analytics</li>
                            <li><span className="check">&#10003;</span> Financial intelligence dashboard</li>
                            <li><span className="check">&#10003;</span> Multi-farm management</li>
                            <li><span className="check">&#10003;</span> API and integrations access</li>
                            <li><span className="check">&#10003;</span> Priority support</li>
                        </ul>
                    </div>
                </div>
            </section>

            <section className="lr-footer">
                <div className="lr-footer-gallery-wrapper">
                    <div className="lr-footer-gallery">
                        <img src="/media__1772004409712.jpg" alt="Farm IoT sensor closeup" />
                        <img src="/media__1772004409748.jpg" alt="Livestock movement heatmap visualization" />
                        <img src="/media__1772004409761.jpg" alt="Smart farm environment imagery" />

                        {/* Duplicate for infinite marquee scroll */}
                        <img src="/media__1772004409712.jpg" alt="Farm IoT sensor closeup" />
                        <img src="/media__1772004409748.jpg" alt="Livestock movement heatmap visualization" />
                        <img src="/media__1772004409761.jpg" alt="Smart farm environment imagery" />
                    </div>
                </div>

                <div className="lr-footer-bar">
                    <span>Independent</span>
                    <span>Overview</span>
                    <span>Multidisciplinary</span>
                    <span>Focus</span>
                </div>

                <div className="lr-footer-statement">
                    <p>GoMata provides autonomous intelligence for livestock through real-time monitoring, predictive AI, and operational automation — enabling farms to become intelligent, efficient, and self-optimizing systems.</p>
                    <button className="lr-back-top" onClick={openRegister}>Start Monitoring &rarr;</button>
                </div>

                <div className="lr-footer-links">
                    <div className="links-col">
                        <strong>Quick Links</strong>
                        <p>Home, Platform, Intelligence, Pricing, Contact</p>
                    </div>
                    <div className="links-col right">
                        <strong>Platform</strong>
                        <p>Health Intelligence, Movement Intelligence, Device Intelligence, Production Intelligence, Financial Intelligence</p>
                    </div>
                </div>

                <div className="lr-giant-footer-wrapper">
                    <div className="lr-giant-footer lr-footer-marquee">
                        <span>GoMata AI GoMata AI GoMata AI GoMata AI GoMata AI </span>
                    </div>
                </div>
            </section>

            {/* MODALS */}
            {(isLoginOpen || isRegisterOpen) && (
                <div className="modal-overlay" onClick={closeModals}>
                    <div className={`modal-content ${isRegisterOpen ? 'modal-wizard' : ''}`} onClick={(e) => e.stopPropagation()}>
                        {!isRegisterOpen && <button className="modal-close" onClick={closeModals}>&times;</button>}

                        {isLoginOpen && (
                            <div className="auth-form-container" style={{ padding: '2rem' }}>
                                <h2>{isFirstTime ? 'Setup Your Password' : 'Welcome Back'}</h2>
                                {!isFirstTime && <p className="auth-subtitle">Login to your dashboard</p>}

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

                                <form onSubmit={handleLogin}>
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
                                            <button type="submit" className="btn-primary-glow full-width">Login Securely</button>

                                            {isStaff && (
                                                <div style={{ textAlign: 'center', marginTop: '16px' }}>
                                                    <button type="button" onClick={() => setIsFirstTime(true)} style={{ background: 'none', border: 'none', color: '#10b981', fontWeight: 600, cursor: 'pointer', textDecoration: 'underline' }}>
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
                                                <button type="button" className="btn-primary-glow" style={{ background: '#e2e8f0', color: '#334155' }} onClick={() => { setIsFirstTime(false); setError(''); }}>Cancel</button>
                                                <button type="submit" className="btn-primary-glow" style={{ flex: 2 }}>Secure Account</button>
                                            </div>
                                        </>
                                    )}
                                </form>
                                {!isFirstTime && (
                                    <p className="auth-switch">
                                        Don't have an account? <span onClick={openRegister}>Sign Up</span>
                                    </p>
                                )}
                            </div>
                        )}

                        {isRegisterOpen && (
                            <div className="wizard-split-container">
                                {/* LEFT PANEL - Branding & Testimonial */}
                                <div className={`wizard-left-panel slide-theme-${activeSlide}`}>
                                    <div className="wizard-left-header">
                                        <Activity className="logo-icon theme-text" />
                                        <span className="logo-text theme-text">GOMATA</span>
                                    </div>
                                    <div className="wizard-left-body">
                                        <div className="wizard-slider-container">
                                            {leftSlides.map((slide, idx) => (
                                                <div key={idx} className={`wizard-slide ${idx === activeSlide ? 'active' : ''}`}>
                                                    <h1 className="wizard-left-title">{slide.title}</h1>
                                                    <p className="wizard-left-subtitle">{slide.subtitle}</p>

                                                    <div className="wizard-testimonial-card shadow-glass">
                                                        <h3 className="testimonial-title">{slide.cardLabel}</h3>
                                                        <p className="testimonial-text">{slide.cardText}</p>
                                                        <div className="testimonial-author">
                                                            <div className="author-avatar"><Activity size={16} /></div>
                                                            <span className="author-name">{slide.author}</span>
                                                            <div className="author-stars">★★★★★</div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ))}
                                        </div>

                                        <div className="wizard-left-pagination">
                                            {leftSlides.map((_, idx) => (
                                                <span key={idx} className={`dot ${idx === activeSlide ? 'active' : ''}`} onClick={() => setActiveSlide(idx)}></span>
                                            ))}
                                        </div>
                                    </div>
                                </div>

                                {/* RIGHT PANEL - Form Container */}
                                <div className="wizard-right-panel">
                                    <button className="modal-close right-close" onClick={closeModals}>&times;</button>

                                    <div className="wizard-right-content">
                                        <div className="auth-form-container wizard-container">

                                            {/* Dribbble-style top step indicator */}
                                            <div className="wizard-top-indicator">
                                                <div className={`step-dot ${step >= 1 ? 'active' : ''}`}></div>
                                                <div className={`step-line-top ${step >= 2 ? 'active' : ''}`}></div>
                                                <div className={`step-dot ${step >= 2 ? 'active' : ''}`}></div>
                                                <div className={`step-line-top ${step >= 3 ? 'active' : ''}`}></div>
                                                <div className={`step-dot ${step >= 3 ? 'active' : ''}`}></div>
                                                <div className={`step-line-top ${step >= 4 ? 'active' : ''}`}></div>
                                                <div className={`step-dot ${step >= 4 ? 'active' : ''}`}></div>
                                                <div className={`step-line-top ${step >= 5 ? 'active' : ''}`}></div>
                                                <div className={`step-dot ${step >= 5 ? 'active' : ''}`}></div>
                                            </div>

                                            {step === 1 && (
                                                <>
                                                    <div className="dribbble-title-container">
                                                        <Activity className="dribbble-title-logo" size={32} />
                                                        <h2 className="dribbble-title">Let's get started</h2>
                                                    </div>

                                                    <form onSubmit={(e) => { e.preventDefault(); setStep(2); }}>
                                                        <div className="form-row-2">
                                                            <div className="form-group">
                                                                <label>Full Name</label>
                                                                <input type="text" name="name" value={regData.name} onChange={handleRegChange} required />
                                                            </div>
                                                            <div className="form-group">
                                                                <label>Date of Birth</label>
                                                                <div className="premium-date-wrapper">
                                                                    <div className={`premium-date-display ${!regData.dob ? 'placeholder' : ''}`}>
                                                                        {regData.dob || "Select your birth date"}
                                                                    </div>
                                                                    <input type="date" name="dob" value={regData.dob} onChange={handleRegChange} required className="premium-date-input-hidden" />
                                                                    <Calendar className="date-icon" size={18} />
                                                                </div>
                                                            </div>
                                                        </div>

                                                        <div className="form-group">
                                                            <label>Address Line 1</label>
                                                            <input type="text" name="address1" value={regData.address1} onChange={handleRegChange} required placeholder="Street / House No." />
                                                        </div>
                                                        <div className="form-row-2">
                                                            <div className="form-group">
                                                                <label>Address Line 2 (Optional)</label>
                                                                <input type="text" name="address2" value={regData.address2} onChange={handleRegChange} />
                                                            </div>
                                                            <div className="form-group">
                                                                <label>Address Line 3 (Optional)</label>
                                                                <input type="text" name="address3" value={regData.address3} onChange={handleRegChange} />
                                                            </div>
                                                        </div>

                                                        <div className="form-row-3">
                                                            <div className="form-group">
                                                                <label>Pincode</label>
                                                                <input type="text" name="pincode" value={regData.pincode} onChange={handlePincodeChange} required maxLength="6" />
                                                            </div>
                                                            <div className="form-group">
                                                                <label>City</label>
                                                                <input type="text" name="city" value={regData.city} readOnly className="read-only-input" />
                                                            </div>
                                                            <div className="form-group">
                                                                <label>State</label>
                                                                <input type="text" name="state" value={regData.state} readOnly className="read-only-input" />
                                                            </div>
                                                        </div>

                                                        <div className="form-row-2">
                                                            <div className="form-group">
                                                                <label>Country</label>
                                                                <input type="text" name="country" value={regData.country} readOnly className="read-only-input" />
                                                            </div>
                                                            <div className="form-group">
                                                                <label>Coordinates (Auto-detected)</label>
                                                                <div className="coords-display">
                                                                    <span>{regData.lat || '--'}</span>, <span>{regData.lng || '--'}</span>
                                                                </div>
                                                            </div>
                                                        </div>

                                                        <button type="submit" className="btn-primary-glow full-width">Next Step <ChevronRight size={16} /></button>
                                                    </form>
                                                    <p className="auth-switch">
                                                        Already have an account? <span onClick={openLogin}>Login</span>
                                                    </p>
                                                </>
                                            )}

                                            {step === 2 && (
                                                <>
                                                    <div className="dribbble-title-container">
                                                        <ShieldCheck className="dribbble-title-logo" size={32} />
                                                        <h2 className="dribbble-title">Verification</h2>
                                                    </div>

                                                    <div className="verification-form">
                                                        {/* Mobile Verification */}
                                                        <div className="form-group">
                                                            <label>Mobile Number</label>
                                                            <div className="input-group-verify">
                                                                <input type="tel" name="mobile" value={regData.mobile} onChange={handleRegChange} placeholder="+91 9876543210" disabled={verification.mobileVerified} />
                                                                {!verification.mobileVerified ? (
                                                                    <button type="button" className="btn-verify" onClick={() => handleSendOTP('mobile')}>
                                                                        {verification.otpSentMobile ? 'Resend' : 'Verify'}
                                                                    </button>
                                                                ) : (
                                                                    <span className="verified-badge"><Check size={14} /> Verified</span>
                                                                )}
                                                            </div>
                                                            {verification.otpSentMobile && !verification.mobileVerified && (
                                                                <div className="otp-entry fade-in">
                                                                    <input type="text" placeholder="Enter 6-digit OTP" className="otp-input"
                                                                        onChange={(e) => {
                                                                            const val = e.target.value.replace(/\D/g, '');
                                                                            if (val.length === 6) handleVerifySuccess('mobile', val);
                                                                        }} maxLength="6" />
                                                                </div>
                                                            )}
                                                        </div>

                                                        {/* Email Verification */}
                                                        <div className="form-group">
                                                            <label>Email Address</label>
                                                            <div className="input-group-verify">
                                                                <input type="email" name="email" value={regData.email} onChange={handleRegChange} placeholder="user@example.com" disabled={verification.emailVerified} />
                                                                {!verification.emailVerified ? (
                                                                    <button type="button" className="btn-verify" onClick={() => handleSendOTP('email')}>
                                                                        {verification.otpSentEmail ? 'Resend' : 'Verify'}
                                                                    </button>
                                                                ) : (
                                                                    <span className="verified-badge"><Check size={14} /> Verified</span>
                                                                )}
                                                            </div>
                                                            {verification.otpSentEmail && !verification.emailVerified && (
                                                                <div className="otp-entry fade-in">
                                                                    <input type="text" placeholder="Enter 6-digit OTP" className="otp-input"
                                                                        onChange={(e) => {
                                                                            const val = e.target.value.replace(/\D/g, '');
                                                                            if (val.length === 6) handleVerifySuccess('email', val);
                                                                        }} maxLength="6" />
                                                                </div>
                                                            )}
                                                        </div>

                                                        <div className="form-row-2">
                                                            <div className="form-group">
                                                                <label>Password</label>
                                                                <input type="password" name="password" value={regData.password} onChange={handleRegChange} required />
                                                            </div>
                                                            <div className="form-group">
                                                                <label>Confirm Password</label>
                                                                <input type="password" name="confirmPassword" value={regData.confirmPassword || ''} onChange={handleRegChange} required />
                                                            </div>
                                                        </div>
                                                        {regData.password && regData.confirmPassword && regData.password !== regData.confirmPassword && (
                                                            <div className="error-message" style={{ marginBottom: '1rem' }}>Passwords do not match</div>
                                                        )}

                                                        {/* Mock reCAPTCHA */}
                                                        <div className="recaptcha-mock" onClick={() => setVerification(prev => ({ ...prev, captchaVerified: !prev.captchaVerified }))}>
                                                            <div className={`checkbox-box ${verification.captchaVerified ? 'checked' : ''}`}>
                                                                {verification.captchaVerified && <Check size={18} />}
                                                            </div>
                                                            <span>I'm not a robot</span>
                                                            <img src="https://www.gstatic.com/recaptcha/api2/logo_48.png" alt="reCAPTCHA" className="captcha-logo" />
                                                        </div>

                                                        <div className="wizard-actions">
                                                            <button type="button" className="btn-text" onClick={() => setStep(1)}>Back</button>
                                                            <button type="button" className="btn-wizard-next" onClick={handleStep2Submit}
                                                                disabled={!verification.mobileVerified || !verification.emailVerified || !verification.captchaVerified || !regData.password || regData.password !== regData.confirmPassword}>
                                                                Next Step <ArrowRight size={18} />
                                                            </button>
                                                        </div>
                                                    </div>
                                                </>
                                            )}

                                            {step === 3 && (
                                                <>
                                                    <div className="dribbble-title-container">
                                                        <BarChart2 className="dribbble-title-logo" size={32} />
                                                        <h2 className="dribbble-title">Farm Details</h2>
                                                    </div>

                                                    <form onSubmit={(e) => { e.preventDefault(); setStep(4); }}>
                                                        <p className="auth-subtitle" style={{ marginBottom: '1.5rem' }}>Define your physical spaces.</p>

                                                        <div className="array-manager-group">
                                                            <div className="form-group flex-row-between">
                                                                <label>How many Farms do you operate here?</label>
                                                                <input type="number" min="1" value={farmCount}
                                                                    onChange={(e) => {
                                                                        const c = parseInt(e.target.value) || 1;
                                                                        setFarmCount(c);
                                                                        const existing = regData.farms;
                                                                        if (c > existing.length) {
                                                                            const add = Array.from({ length: c - existing.length }).map((_, i) => ({
                                                                                tempId: `farm_${existing.length + i}_${Date.now()}`,
                                                                                name: '', locationType: 'Polygon', geofence: null
                                                                            }));
                                                                            setRegData(p => ({ ...p, farms: [...existing, ...add] }));
                                                                        }
                                                                        if (c < existing.length) {
                                                                            setRegData(p => ({ ...p, farms: existing.slice(0, c) }));
                                                                        }
                                                                    }}
                                                                    className="small-number-input" />
                                                            </div>

                                                            {regData.farms.map((farm, index) => (
                                                                <div key={farm.tempId} className="card-glass nested-form-block">
                                                                    <h4>Farm #{index + 1}</h4>
                                                                    <div className="form-row-3">
                                                                        <div className="form-group">
                                                                            <label>Farm Name</label>
                                                                            <input type="text" value={farm.name} onChange={(e) => handleArrayChange('farms', index, 'name', e.target.value)} required placeholder={`e.g. North Pasture`} />
                                                                        </div>
                                                                        <div className="form-group">
                                                                            <label>Mapping Method</label>
                                                                            <select value={farm.locationType} onChange={(e) => handleArrayChange('farms', index, 'locationType', e.target.value)} className="wizard-select">
                                                                                <option value="Polygon">Polygon (Draw Edges)</option>
                                                                                <option value="Point">Circular (Center + Radius)</option>
                                                                            </select>
                                                                        </div>
                                                                        <div className="form-group" style={{ display: 'flex', alignItems: 'flex-end' }}>
                                                                            <button type="button" className={`btn-wizard-next ${farm.geofence ? 'success' : ''}`} style={{ width: '100%', ...(farm.geofence ? { backgroundColor: 'var(--semantic-success)' } : {}) }} onClick={() => handleOpenMap('farms', index, farm.tempId)}>
                                                                                <Map size={16} /> {farm.geofence ? 'Boundary Mapped ✓' : 'Mapping Boundary'}
                                                                            </button>
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>

                                                        <div className="array-manager-group" style={{ marginTop: '2rem' }}>
                                                            <div className="form-group flex-row-between">
                                                                <label>How many Sub-Zones exist within these farms?</label>
                                                                <input type="number" min="0" value={zoneCount}
                                                                    onChange={(e) => {
                                                                        const c = parseInt(e.target.value) || 0;
                                                                        setZoneCount(c);
                                                                        const existing = regData.zones;
                                                                        if (c > existing.length) {
                                                                            const add = Array.from({ length: c - existing.length }).map((_, i) => ({
                                                                                tempId: `zone_${existing.length + i}_${Date.now()}`,
                                                                                farmTempId: regData.farms[0]?.tempId || '',
                                                                                name: '', locationType: 'Polygon', geofence: null
                                                                            }));
                                                                            setRegData(p => ({ ...p, zones: [...existing, ...add] }));
                                                                        }
                                                                        if (c < existing.length) {
                                                                            setRegData(p => ({ ...p, zones: existing.slice(0, c) }));
                                                                        }
                                                                    }}
                                                                    className="small-number-input" />
                                                            </div>

                                                            {regData.zones.map((zone, index) => (
                                                                <div key={zone.tempId} className="card-glass nested-form-block theme-blue">
                                                                    <h4>Zone #{index + 1}</h4>
                                                                    <div className="form-row-3">
                                                                        <div className="form-group">
                                                                            <label>Assigned Farm</label>
                                                                            <select value={zone.farmTempId} onChange={(e) => handleArrayChange('zones', index, 'farmTempId', e.target.value)} className="wizard-select">
                                                                                {regData.farms.map(f => <option key={f.tempId} value={f.tempId}>{f.name || 'Unnamed Farm'}</option>)}
                                                                            </select>
                                                                        </div>
                                                                        <div className="form-group">
                                                                            <label>Zone Name</label>
                                                                            <input type="text" value={zone.name} onChange={(e) => handleArrayChange('zones', index, 'name', e.target.value)} required placeholder={`e.g. Grazing Sector A`} />
                                                                        </div>
                                                                    </div>
                                                                    <div className="form-row-2">
                                                                        <div className="form-group">
                                                                            <label>Mapping Method</label>
                                                                            <select value={zone.locationType} onChange={(e) => handleArrayChange('zones', index, 'locationType', e.target.value)} className="wizard-select">
                                                                                <option value="Polygon">Polygon (Draw Edges)</option>
                                                                                <option value="Point">Circular (Center + Radius)</option>
                                                                            </select>
                                                                        </div>
                                                                        <div className="form-group" style={{ display: 'flex', alignItems: 'flex-end' }}>
                                                                            <button type="button" className={`btn-wizard-next ${zone.geofence ? 'success' : ''}`} style={{ width: '100%', ...(zone.geofence ? { backgroundColor: 'var(--semantic-success)' } : {}) }} onClick={() => handleOpenMap('zones', index, zone.tempId)}>
                                                                                <Map size={16} /> {zone.geofence ? 'Boundary Mapped ✓' : 'Mapping Boundary'}
                                                                            </button>
                                                                        </div>
                                                                    </div>
                                                                </div>
                                                            ))}
                                                        </div>

                                                        <div className="wizard-actions" style={{ marginTop: '2rem' }}>
                                                            <button type="button" className="btn-text" onClick={() => setStep(2)}>Back</button>
                                                            <button type="button" className="btn-wizard-next" onClick={() => setStep(4)}
                                                                disabled={regData.farms.length === 0 || regData.farms.some(f => !f.geofence || !f.name)}>
                                                                Next Step <ArrowRight size={18} />
                                                            </button>
                                                        </div>
                                                    </form>
                                                </>
                                            )}

                                            {step === 4 && (
                                                <>
                                                    <div className="dribbble-title-container">
                                                        <Activity className="dribbble-title-logo" size={32} />
                                                        <h2 className="dribbble-title">Livestock Profiles</h2>
                                                    </div>

                                                    <form>
                                                        <p className="auth-subtitle" style={{ marginBottom: '1.5rem' }}>Deploy the Gomata sensor network to your animals.</p>

                                                        <div className="array-manager-group">
                                                            <div className="form-group flex-row-between">
                                                                <label>How many animals are you onboarding today?</label>
                                                                <input type="number" min="1" value={livestockCount}
                                                                    onChange={(e) => {
                                                                        const c = parseInt(e.target.value) || 1;
                                                                        setLivestockCount(c);
                                                                        const existing = regData.livestock;
                                                                        if (c > existing.length) {
                                                                            const add = Array.from({ length: c - existing.length }).map((_, i) => ({
                                                                                tagNumber: '', name: '', breed: '', type: 'Dairy Cattle', age: '', weight: '', deviceId: '',
                                                                                farmTempId: regData.farms[0]?.tempId || '', zoneTempId: regData.zones[0]?.tempId || '',
                                                                                vaccinationNotes: '', breedingNotes: '', additionalNotes: ''
                                                                            }));
                                                                            setRegData(p => ({ ...p, livestock: [...existing, ...add] }));
                                                                        }
                                                                        if (c < existing.length) {
                                                                            setRegData(p => ({ ...p, livestock: existing.slice(0, c) }));
                                                                        }
                                                                    }}
                                                                    className="small-number-input" />
                                                            </div>

                                                            {regData.livestock.map((l, index) => (
                                                                <div key={index} className="card-glass nested-form-block theme-emerald">
                                                                    <h4>Animal #{index + 1}</h4>

                                                                    <div className="form-row-2">
                                                                        <div className="form-group">
                                                                            <label>Assigned Farm</label>
                                                                            <select value={l.farmTempId} onChange={(e) => handleArrayChange('livestock', index, 'farmTempId', e.target.value)} className="wizard-select">
                                                                                {regData.farms.map(f => <option key={f.tempId} value={f.tempId}>{f.name}</option>)}
                                                                            </select>
                                                                        </div>
                                                                        <div className="form-group">
                                                                            <label>Assigned Zone (Optional)</label>
                                                                            <select value={l.zoneTempId} onChange={(e) => handleArrayChange('livestock', index, 'zoneTempId', e.target.value)} className="wizard-select">
                                                                                <option value="">-- No Specific Zone --</option>
                                                                                {regData.zones.filter(z => z.farmTempId === l.farmTempId).map(z => <option key={z.tempId} value={z.tempId}>{z.name}</option>)}
                                                                            </select>
                                                                        </div>
                                                                    </div>

                                                                    <div className="form-row-3">
                                                                        <div className="form-group"><label>Name / ID</label><input type="text" value={l.name} onChange={(e) => handleArrayChange('livestock', index, 'name', e.target.value)} /></div>
                                                                        <div className="form-group">
                                                                            <label>Species Template</label>
                                                                            <select value={l.type} onChange={(e) => handleArrayChange('livestock', index, 'type', e.target.value)} className="wizard-select">
                                                                                <option>Dairy Cattle</option><option>Beef Cattle</option><option>Sheep</option><option>Goats</option><option>Pigs</option>
                                                                            </select>
                                                                        </div>
                                                                        <div className="form-group"><label>Breed</label><input type="text" value={l.breed} onChange={(e) => handleArrayChange('livestock', index, 'breed', e.target.value)} /></div>
                                                                    </div>
                                                                    <div className="form-row-2">
                                                                        <div className="form-group"><label>Age (Months/Years)</label><input type="text" value={l.age || ''} onChange={(e) => handleArrayChange('livestock', index, 'age', e.target.value)} placeholder="e.g. 24 Months" /></div>
                                                                        <div className="form-group"><label>Weight (kg/lbs)</label><input type="text" value={l.weight || ''} onChange={(e) => handleArrayChange('livestock', index, 'weight', e.target.value)} placeholder="e.g. 600 kg" /></div>
                                                                    </div>
                                                                    <div className="form-row-2">
                                                                        <div className="form-group"><label>Vaccination Notes</label><input type="text" value={l.vaccinationNotes || ''} onChange={(e) => handleArrayChange('livestock', index, 'vaccinationNotes', e.target.value)} placeholder="Recent vaccinations, dates, etc." /></div>
                                                                        <div className="form-group"><label>Breeding Notes</label><input type="text" value={l.breedingNotes || ''} onChange={(e) => handleArrayChange('livestock', index, 'breedingNotes', e.target.value)} placeholder="Breeding history" /></div>
                                                                    </div>
                                                                    <div className="form-group"><label>Additional Notes</label><input type="text" value={l.additionalNotes || ''} onChange={(e) => handleArrayChange('livestock', index, 'additionalNotes', e.target.value)} placeholder="Any behavioral or general notes" style={{ width: '100%' }} /></div>
                                                                </div>
                                                            ))}
                                                        </div>

                                                        <div className="wizard-actions" style={{ marginTop: '2rem' }}>
                                                            <button type="button" className="btn-text" onClick={() => setStep(3)}>Back</button>
                                                            <button type="button" className="btn-wizard-next" onClick={() => setStep(5)}
                                                                disabled={regData.livestock.length === 0 || regData.livestock.some(l => !l.farmTempId)}>
                                                                Continue
                                                            </button>
                                                        </div>
                                                    </form>
                                                </>
                                            )}

                                            {step === 5 && (
                                                <>
                                                    <div className="wizard-header">
                                                        <h2 className="dribbble-title">About Devices</h2>
                                                        <p className="auth-subtitle">Assign physical hardware sensors to monitor telemetry data for each animal.</p>
                                                    </div>

                                                    <form onSubmit={(e) => { e.preventDefault(); handleFinalRegister(); }} className="wizard-form">
                                                        <div className="wizard-scroll-container">
                                                            {regData.livestock.map((l, index) => {
                                                                const farmName = regData.farms.find(f => f.tempId === l.farmTempId)?.name || 'Unknown Farm';
                                                                const zoneName = regData.zones.find(z => z.tempId === l.zoneTempId)?.name || 'No specific zone';

                                                                return (
                                                                    <div key={index} className="card-glass nested-form-block theme-emerald">
                                                                        <h4>Device Assignment: {l.name || `Animal #${index + 1}`}</h4>

                                                                        <div className="form-row-2" style={{ marginBottom: '1.5rem', opacity: 0.8 }}>
                                                                            <div className="form-group">
                                                                                <label>Livestock Reference</label>
                                                                                <input type="text" value={`${l.name || `Animal #${index + 1}`} - ${l.type}`} disabled className="read-only-input" />
                                                                            </div>
                                                                            <div className="form-group">
                                                                                <label>Assigned Location</label>
                                                                                <input type="text" value={`${farmName} (${zoneName})`} disabled className="read-only-input" />
                                                                            </div>
                                                                        </div>

                                                                        <div className="form-row-2">
                                                                            <div className="form-group">
                                                                                <label>Device ID</label>
                                                                                <input type="text" value={l.deviceId || ''} onChange={(e) => handleArrayChange('livestock', index, 'deviceId', e.target.value)} required placeholder={`e.g. GM-SN-1001`} />
                                                                            </div>
                                                                            <div className="form-group">
                                                                                <label>Device Type</label>
                                                                                <select value={l.deviceType || 'Other'} onChange={(e) => handleArrayChange('livestock', index, 'deviceType', e.target.value)} className="wizard-select">
                                                                                    <option>Neck Collar</option>
                                                                                    <option>Ear Tag</option>
                                                                                    <option>Leg Band</option>
                                                                                    <option>Implantable Chip</option>
                                                                                    <option>Other</option>
                                                                                </select>
                                                                            </div>
                                                                        </div>
                                                                    </div>
                                                                );
                                                            })}
                                                        </div>

                                                        <div className="wizard-actions" style={{ marginTop: '2rem' }}>
                                                            <button type="button" className="btn-text" onClick={() => setStep(4)}>Back</button>
                                                            <button type="button" className="btn-wizard-next" onClick={handleFinalRegister}
                                                                disabled={regData.livestock.length === 0 || regData.livestock.some(l => !l.deviceId || !l.deviceType)}>
                                                                Deploy & Finish Setup
                                                            </button>
                                                        </div>
                                                    </form>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            </div>
                        )
                        }
                    </div >
                </div >
            )}

            {/* Render the Map Selection Modal if it's open and an item is active */}
            {activeMapItem && activeMapItem.type && activeMapItem.index !== null && (
                <MapSelectionModal
                    isOpen={mapModalOpen}
                    onClose={() => setMapModalOpen(false)}
                    onSave={handleMapSave}
                    locationType={regData[activeMapItem.type][activeMapItem.index]?.locationType || 'Polygon'}
                    initialGeofence={regData[activeMapItem.type][activeMapItem.index]?.geofence || null}
                    parentGeofence={activeMapItem.type === 'zones' ? regData.farms.find(f => f.tempId === regData.zones[activeMapItem.index].farmTempId)?.geofence : null}
                    allZoneGeofences={activeMapItem.type === 'zones' ? regData.zones.filter((z, idx) => z.farmTempId === regData.zones[activeMapItem.index].farmTempId && idx < activeMapItem.index && z.geofence).map(z => z.geofence) : []}
                    itemName={regData[activeMapItem.type][activeMapItem.index]?.name || `${activeMapItem.type === 'farms' ? 'Farm' : 'Zone'} ${activeMapItem.index + 1}`}
                />
            )}
        </div >
    );
};

// Helper Component for Feature Cards
const FeatureCard = ({ icon, color, title, desc }) => (
    <div className={`feature-card-modern ${color}`}>
        <div className={`feature-icon-circle ${color}`}>
            {icon}
        </div>
        <h3>{title}</h3>
        <p>{desc}</p>
    </div>
);

export default Landing;
