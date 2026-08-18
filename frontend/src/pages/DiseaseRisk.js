import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Menu, Eye, Activity, Sparkles, ShieldAlert, Cpu, Globe, Database, Play, AlertTriangle, List, Map as MapIcon, Link as LinkIcon, Settings, LayoutDashboard, Calendar, Navigation, Layers, ChevronRight, CheckCircle2, TrendingUp, TrendingDown, Bell, Search, Info, LogOut, Minus, Maximize2, MoveRight, User, Clock, FileText, CheckCircle, ArrowRight } from 'lucide-react';
import { ComposedChart, Line, Area, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Legend } from 'recharts';
import './DiseaseRisk.css';

import { livestockAPI } from '../services/api';

const UserAvatar = ({ name }) => {
    const initials = name ? name.split(' ').map(n => n[0]).join('') : 'U';
    return (
        <div style={{ width: '32px', height: '32px', borderRadius: '50%', background: '#f0fdf4', color: '#16a34a', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>
            {initials}
        </div>
    );
};

export default function DiseaseRisk() {
    const navigate = useNavigate();
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [horizon, setHorizon] = useState('7D');
    const user = { name: 'Admin Manager' };

    const [hasOutbreak, setHasOutbreak] = useState(false);
    const [exposedCount, setExposedCount] = useState(0);

    React.useEffect(() => {
        const fetchLivestock = async () => {
            try {
                const response = await livestockAPI.getAll();
                if (response && response.data) {
                    const criticalCount = response.data.filter(i => i.latestSensorData?.temperature > 39.5).length;
                    setHasOutbreak(criticalCount > 0);
                    setExposedCount(criticalCount * 3); // Estimate 3 exposed per critical
                }
            } catch (err) {
                console.error("Failed to fetch livestock for disease risk:", err);
            }
        };
        fetchLivestock();
    }, []);

    // Spread Projection Data
    const spreadData = hasOutbreak ? [
        { day: 'Day 1', noAction: 2, partial: 2, full: 2 },
        { day: 'Day 2', noAction: 5, partial: 4, full: 3 },
        { day: 'Day 3', noAction: 11, partial: 8, full: 5 },
        { day: 'Day 4', noAction: 18, partial: 12, full: 6 },
        { day: 'Day 5', noAction: 27, partial: 16, full: 4 },
        { day: 'Day 6', noAction: 38, partial: 21, full: 2 },
        { day: 'Day 7', noAction: 45, partial: 28, full: 0 },
    ] : [
        { day: 'Day 1', noAction: 0, partial: 0, full: 0 },
        { day: 'Day 2', noAction: 0, partial: 0, full: 0 },
        { day: 'Day 3', noAction: 0, partial: 0, full: 0 },
        { day: 'Day 4', noAction: 0, partial: 0, full: 0 },
        { day: 'Day 5', noAction: 0, partial: 0, full: 0 },
        { day: 'Day 6', noAction: 0, partial: 0, full: 0 },
        { day: 'Day 7', noAction: 0, partial: 0, full: 0 },
    ];

    const contactTracing = hasOutbreak ? [
        { exposed: 'MIX005', index: 'INF012', distance: '< 2m', duration: '4.5 hrs', prob: 'High', zone: 'Barn A Water Trough' },
        { exposed: 'MIX009', index: 'INF012', distance: '3-5m', duration: '2.1 hrs', prob: 'High', zone: 'Barn A Feed Line' },
        { exposed: 'MIX014', index: 'INF003', distance: '1-3m', duration: '5.0 hrs', prob: 'High', zone: 'Pasture Transition' },
        { exposed: 'JER102', index: 'INF008', distance: '8-10m', duration: '0.5 hrs', prob: 'Low', zone: 'Milking Parlor Q' }
    ] : [];

    const diseaseDiffDiag = [
        { disease: 'Foot and Mouth Disease (FMD)', prob: 34, notifiable: true },
        { disease: 'Bovine Respiratory Disease (BRD)', prob: 41, notifiable: false, recommended: true },
        { disease: 'Heat Stress Fever', prob: 18, notifiable: false },
        { disease: 'Other / Unknown Vector', prob: 7, notifiable: false }
    ];

    const historicalEpisodes = [];

    return (
        <div className="predictions-layout">
            {/* SIDEBAR */}
            <aside className={`sidebar-premium ${!isSidebarOpen ? 'collapsed' : ''}`}>
                <div className="sidebar-scrollable">
                    <div className="sidebar-logo">
                        <Activity color="#10b981" size={28} strokeWidth={3} />
                        {isSidebarOpen && <h2>GoMata</h2>}
                    </div>

                    <nav className="sidebar-nav">
                        <div className="sidebar-section-title">Main</div>
                        <div className="nav-item-premium" onClick={() => navigate('/dashboard')}><LayoutDashboard className="nav-icon" /> <span>Overview</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/livestock')}><List className="nav-icon" /> <span>Animals</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/map')}><MapIcon className="nav-icon" /> <span>Map Intelligence</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/health-analytics')}><Activity className="nav-icon" /> <span>Health Analytics</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/alerts')}><AlertTriangle className="nav-icon" /> <span>Alerts Center</span></div>

                        <div className="sidebar-section-title">Operations</div>
                        <div className="nav-item-premium" onClick={() => navigate('/devices')}><Cpu className="nav-icon" /> <span>Devices</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/farm')}><Globe className="nav-icon" /> <span>Farms & Locations</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/breeds')}><Database className="nav-icon" /> <span>Breeds</span></div>

                        <div className="sidebar-section-title">Intelligence</div>
                        <div className="nav-item-premium" onClick={() => navigate('/ai-orchestrator')}><Play className="nav-icon" /> <span>AI Orchestrator</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/predictions')}><Sparkles className="nav-icon" /> <span>AI Predictions</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/behaviour')}><Eye className="nav-icon" /> <span>Behaviour Analysis</span></div>
                        <div className="nav-item-premium active" onClick={() => navigate('/disease-risk')}><ShieldAlert className="nav-icon" /> <span>Disease Risk</span></div>
                    </nav>
                </div>
            </aside>

            {/* MAIN CONTENT AREA */}
            <main className="predictions-main b-scroll-container">

                {/* HEADER */}
                <header className="b-header" style={{ marginBottom: 24 }}>
                    <div className="b-header-top">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                            <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                                <Menu size={20} />
                            </button>
                            <div className="b-title-area" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
                                <h1 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}><ShieldAlert strokeWidth={2.5} size={28} color="#ef4444" /> Disease Risk & Epidemiology</h1>
                                <p style={{ margin: 0 }}>Herd-level transmission pathways, contact tracing, and proactive containment.</p>
                            </div>
                        </div>

                        <div className="b-header-controls">
                            <div className="b-horizon-controls">
                                <button className={`b-horizon-btn ${horizon === '24H' ? 'active' : ''}`} onClick={() => setHorizon('24H')}>24 hrs</button>
                                <button className={`b-horizon-btn ${horizon === '7D' ? 'active' : ''}`} onClick={() => setHorizon('7D')}>7 days</button>
                                <button className={`b-horizon-btn ${horizon === '30D' ? 'active' : ''}`} onClick={() => setHorizon('30D')}>30 days</button>
                            </div>
                        </div>
                    </div>

                    <div className="b-quick-stats" style={{ paddingTop: '20px', borderTop: '1px solid #f0f2f1' }}>
                        <div className="b-stat-chip">
                            <span>Active Outbreaks</span>
                            <strong>{hasOutbreak ? '1 (Barn A)' : '0'}</strong>
                        </div>
                        <div className="b-stat-chip">
                            <span>Exposed Population</span>
                            <strong>{exposedCount} Animals</strong>
                        </div>
                        <div className="b-stat-chip" style={{ background: hasOutbreak ? '#fef2f2' : '#f0fdf4', borderColor: hasOutbreak ? '#fecaca' : '#bbf7d0' }}>
                            <span style={{ color: hasOutbreak ? '#b91c1c' : '#166534' }}>Financial Risk Exposure</span>
                            <strong style={{ color: hasOutbreak ? '#991b1b' : '#15803d' }}>{hasOutbreak ? '₹84,000' : '₹0'}</strong>
                        </div>
                    </div>
                </header>

                <div style={{ paddingBottom: '40px' }}>
                    <div className="dr-split-grid">

                        {/* Section 1: Transmission Map */}
                        <div className="dr-card">
                            <div className="dr-card-header">
                                <h3><Globe size={20} color="#3b82f6" /> Static Transmission Map</h3>
                                <button className="dr-icon-btn"><Maximize2 size={16} /></button>
                            </div>
                            <p className="dr-card-desc">Mapping disease vectors based on shared resources and proximity.</p>

                            <div className="dr-map-container" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                {hasOutbreak ? (
                                    <>
                                        <svg width="100%" height="100%" viewBox="0 0 500 300" className="dr-svg-map">
                                            {/* Links (Transmission Vectors) */}
                                            <line x1="150" y1="100" x2="350" y2="100" stroke="#ef4444" strokeWidth="4" strokeDasharray="5,5" className="vector-line" />
                                            <line x1="150" y1="100" x2="250" y2="220" stroke="#f59e0b" strokeWidth="3" className="vector-line" />
                                            <line x1="350" y1="100" x2="250" y2="220" stroke="#e2e8f0" strokeWidth="2" />

                                            {/* Animated Arrow Heads */}
                                            <path d="M 250 100 L 240 95 L 240 105 Z" fill="#ef4444" className="vector-arrow" />
                                            <path d="M 200 160 L 195 150 L 205 150 Z" fill="#f59e0b" className="vector-arrow" transform="rotate(-50 200 160)" />

                                            {/* Nodes */}
                                            <circle cx="150" cy="100" r="35" fill="#fee2e2" stroke="#ef4444" strokeWidth="3" />
                                            <text x="150" y="95" textAnchor="middle" fill="#991b1b" fontSize="12" fontWeight="700">Barn A</text>
                                            <text x="150" y="115" textAnchor="middle" fill="#ef4444" fontSize="10" fontWeight="600">18 Infected</text>
                                            <circle cx="150" cy="100" r="45" fill="none" stroke="#ef4444" strokeWidth="1" opacity="0.5" className="pulse-ring" />

                                            <circle cx="350" cy="100" r="30" fill="#f0fdf4" stroke="#10b981" strokeWidth="2" />
                                            <text x="350" y="95" textAnchor="middle" fill="#166534" fontSize="12" fontWeight="700">Pasture B</text>
                                            <text x="350" y="115" textAnchor="middle" fill="#10b981" fontSize="10" fontWeight="600">40 Healthy</text>

                                            <circle cx="250" cy="220" r="28" fill="#fef3c7" stroke="#f59e0b" strokeWidth="2" />
                                            <text x="250" y="215" textAnchor="middle" fill="#92400e" fontSize="12" fontWeight="700">Water 1</text>
                                            <text x="250" y="235" textAnchor="middle" fill="#f59e0b" fontSize="10" fontWeight="600">Shared Source</text>
                                        </svg>

                                        <div className="dr-map-legend">
                                            <span style={{ color: '#ef4444' }}><MoveRight size={14} /> High Risk Vector</span>
                                            <span style={{ color: '#f59e0b' }}><MoveRight size={14} /> Medium Risk Vector</span>
                                        </div>
                                    </>
                                ) : (
                                    <div style={{ textAlign: 'center', color: '#64748b', padding: '40px' }}>
                                        <CheckCircle2 size={32} style={{ marginBottom: 12, color: '#10b981' }} />
                                        <p>No active outbreaks. Transmission vectors are clear.</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Section 2: Spread Projection */}
                        <div className="dr-card">
                            <div className="dr-card-header">
                                <h3><TrendingUp size={20} color="#f59e0b" /> Spread Projection (7 Days)</h3>
                            </div>
                            <p className="dr-card-desc">Forecasting cases under 3 different containment scenarios.</p>

                            <div className="dr-chart-container">
                                <ResponsiveContainer width="100%" height="100%">
                                    <ComposedChart data={spreadData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                                        <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} dy={10} />
                                        <YAxis axisLine={false} tickLine={false} tick={{ fill: '#94a3b8', fontSize: 12 }} />
                                        <RechartsTooltip contentStyle={{ borderRadius: '12px', border: 'none', boxShadow: '0 4px 15px rgba(0,0,0,0.1)' }} />
                                        <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '20px' }} />
                                        <Area type="monotone" dataKey="noAction" name="No Intervention" fill="#fee2e2" stroke="none" fillOpacity={0.6} />
                                        <Line type="monotone" dataKey="noAction" name="No Action" stroke="#ef4444" strokeWidth={3} strokeDasharray="5,5" dot={false} />
                                        <Line type="monotone" dataKey="partial" name="Partial Isolation" stroke="#f59e0b" strokeWidth={3} dot={{ r: 4, fill: '#f59e0b' }} />
                                        <Line type="monotone" dataKey="full" name="Full Quarantine + Treatment" stroke="#3b82f6" strokeWidth={4} dot={{ r: 4, fill: '#3b82f6' }} />
                                    </ComposedChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                    </div>

                    <div className="dr-split-grid" style={{ marginTop: '32px' }}>

                        {/* Section 5: Quarantine Recommendation Engine */}
                        <div className="dr-card" style={{ background: '#f8fafc', border: '1px solid #e2e8f0', boxShadow: 'none' }}>
                            <div className="dr-card-header">
                                <h3><ShieldAlert size={20} color="#10b981" /> Actionable Quarantine Plan</h3>
                            </div>
                            <p className="dr-card-desc">AI-generated containment protocol based on current vectors.</p>

                            <div className="dr-quarantine-box" style={{ display: 'flex', flexDirection: 'column', alignItems: hasOutbreak ? 'stretch' : 'center', justifyContent: 'center' }}>
                                {hasOutbreak ? (
                                    <>
                                        <ul className="dr-q-steps">
                                            <li><CheckCircle2 size={16} color="#10b981" /> <strong>Isolate Immediately:</strong> MIX005, MIX009, and MIX014.</li>
                                            <li><CheckCircle2 size={16} color="#10b981" /> <strong>Relocate To:</strong> Quarantine Zone C (currently empty).</li>
                                            <li><CheckCircle2 size={16} color="#10b981" /> <strong>Protocol:</strong> Do not share water or feed with Barn A for minimum 7 days.</li>
                                            <li><CheckCircle2 size={16} color="#10b981" /> <strong>Vet Trigger:</strong> Auto-notify Dr. Sarah Jenkins for BRD testing.</li>
                                        </ul>

                                        <div className="dr-q-financials">
                                            <div className="dr-cost-item">
                                                <span>Estimated Cost of Isolation</span>
                                                <strong className="good">₹2,400</strong>
                                            </div>
                                            <div className="dr-cost-icon"><ArrowRight size={20} color="#94a3b8" /></div>
                                            <div className="dr-cost-item">
                                                <span>Cost if Spread Reaches Herd</span>
                                                <strong className="bad">₹84,000</strong>
                                            </div>
                                        </div>

                                        <button className="dr-btn-primary">Execute Containment Protocol</button>
                                    </>
                                ) : (
                                    <div style={{ textAlign: 'center', color: '#64748b', padding: '20px' }}>
                                        <ShieldAlert size={32} style={{ marginBottom: 12, color: '#10b981' }} />
                                        <p style={{ margin: 0 }}>No quarantine protocols required. Herd is healthy.</p>
                                    </div>
                                )}
                            </div>
                        </div>

                        {/* Section 6: Historical Disease Episodes */}
                        <div className="dr-card">
                            <div className="dr-card-header">
                                <h3><Clock size={20} color="#64748b" /> Historical Disease Profile</h3>
                            </div>
                            <p className="dr-card-desc">Past events that train the herd-specific prediction model.</p>

                            <div className="dr-timeline">
                                {historicalEpisodes.length > 0 ? historicalEpisodes.map((ep, i) => (
                                    <div key={i} className="dr-timeline-item">
                                        <div className="dr-tl-node"></div>
                                        <div className="dr-tl-content">
                                            <div className="dr-tl-date">{ep.date}</div>
                                            <div className="dr-tl-title">
                                                <strong>{ep.disease}</strong> <span>• {ep.animals} Animals ({ep.location})</span>
                                            </div>
                                            <p className="dr-tl-desc">{ep.resolution}</p>
                                        </div>
                                    </div>
                                )) : (
                                    <div style={{ textAlign: 'center', color: '#64748b', padding: '40px' }}>
                                        <Clock size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
                                        <p>No historical disease events recorded for this herd.</p>
                                    </div>
                                )}
                            </div>
                        </div>

                    </div>
                </div>

            </main>
        </div>
    );
}
