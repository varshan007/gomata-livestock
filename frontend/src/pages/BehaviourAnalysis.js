import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Activity, AlertTriangle, Menu, Search, Bell, Sparkles,
    Eye, Clock, Navigation, Zap, Network, Thermometer, ChevronDown, CheckCircle2,
    Map as MapIcon, Cpu, Globe, Database, Play
} from 'lucide-react';
import { ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Area } from 'recharts';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css';
import './BehaviourAnalysis.css';
import { livestockAPI } from '../services/api';
// --- MOCK DATA FOR CHARTS AND TABLES ---

// Heatmap Data (7 days x 24 hours). 0=Low(Red), 1=Rest(Amber), 2=Move(LightGreen), 3=Graze(DarkGreen)
const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
const hours = Array.from({ length: 24 }, (_, i) => i);
const generateHeatmap = (hasData) => {
    let map = [];
    for (let d = 0; d < 7; d++) {
        let dayRow = [];
        for (let h = 0; h < 24; h++) {
            dayRow.push(hasData ? 1 : null); // Empty state if no data
        }
        map.push(dayRow);
    }
    return map;
};

const correlationData = [];

function BehaviourAnalysis() {
    const navigate = useNavigate();
    const { user } = React.useContext(AuthContext);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [horizon, setHorizon] = useState('7D');
    const [viewMode, setViewMode] = useState('Herd'); // Herd vs Individual
    const [individuals, setIndividuals] = useState([]);
    const [anomalies, setAnomalies] = useState([]);
    
    // Zero data states
    const [hasHistoricalData, setHasHistoricalData] = useState(false);
    const [heatmapData, setHeatmapData] = useState(generateHeatmap(false));
    const [herdStats, setHerdStats] = useState({ grazing: 0, resting: 0, moving: 0, rumination: 0 });

    React.useEffect(() => {
        const fetchLivestock = async () => {
            try {
                const response = await livestockAPI.getAll();
                if (response && response.data) {
                    const data = response.data;
                    const mappedIndividuals = data.map(item => ({
                        id: item.livestock.name || item.livestock.tagNumber,
                        breed: item.livestock.breed || 'Unknown',
                        badge: item.latestSensorData?.temperature > 39.5 ? 'Anomaly' : (item.latestSensorData?.temperature > 39.0 ? 'Attention' : 'Normal'),
                        segments: ['rest', 'rest', 'rest', 'rest', 'rest', 'rest', 'rest', 'rest'], // Init state
                        g: '0h', d: '0km', r: '0h', base: '0%'
                    }));
                    setIndividuals(mappedIndividuals);
                    
                    const actualAnomalies = mappedIndividuals.filter(i => i.badge !== 'Normal').map(i => ({
                        time: 'Just Now',
                        id: i.id,
                        type: 'Health Alert',
                        desc: 'Behavioral/Vitals anomaly detected by AI.',
                        severity: i.badge === 'Anomaly' ? 'High' : 'Medium'
                    }));
                    setAnomalies(actualAnomalies);
                }
            } catch (err) {
                console.error("Failed to fetch livestock for behaviour:", err);
            }
        };
        fetchLivestock();
    }, []);

    return (
        <div className="predictions-layout">
            {/* Standard Sidebar aligned with premium styling */}
            <aside className={`sidebar-premium ${!isSidebarOpen ? 'collapsed' : ''}`}>
                <div className="sidebar-logo">
                    <Activity size={28} className="brand-icon" />
                    <span>GoMata</span>
                </div>
                <div className="sidebar-scroll-container">
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
                        <div className="nav-item-premium active" onClick={() => navigate('/behaviour')}><Eye className="nav-icon" /> <span>Behaviour Analysis</span></div>
                    </nav>
                </div>
            </aside>

            <main className="predictions-main b-scroll-container">

                {/* HEADER */}
                <header className="b-header" style={{ marginBottom: 24 }}>
                    <div className="b-header-top">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                            <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                                <Menu size={20} />
                            </button>
                            <div className="b-title-area" style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start', gap: '4px' }}>
                                <h1 style={{ margin: 0, display: 'flex', alignItems: 'center', gap: '12px' }}><Eye strokeWidth={2.5} size={28} color="#10b981" /> Behaviour Analysis</h1>
                                <p style={{ margin: 0 }}>Detecting behavioural anomalies before they become health events.</p>
                            </div>
                        </div>

                        <div className="b-header-controls">
                            <div className="b-horizon-controls">
                                <button className={`b-horizon-btn ${horizon === '24H' ? 'active' : ''}`} onClick={() => setHorizon('24H')}>24 hrs</button>
                                <button className={`b-horizon-btn ${horizon === '7D' ? 'active' : ''}`} onClick={() => setHorizon('7D')}>7 days</button>
                                <button className={`b-horizon-btn ${horizon === '30D' ? 'active' : ''}`} onClick={() => setHorizon('30D')}>30 days</button>
                            </div>

                            <div className="b-toggle">
                                <button className={viewMode === 'Herd' ? 'active' : ''} onClick={() => setViewMode('Herd')}>Herd View</button>
                                <button className={viewMode === 'Indiv' ? 'active' : ''} onClick={() => setViewMode('Indiv')}>Individual</button>
                            </div>
                        </div>
                    </div>

                    <div className="b-quick-stats" style={{ paddingTop: '20px', borderTop: '1px solid #f0f2f1' }}>
                        <div className="b-stat-chip">
                            <span>Anomalies Today</span>
                            <strong>7</strong>
                        </div>
                        <div className="b-stat-chip">
                            <span>Behav. Delta vs Yest</span>
                            <strong>4 Animals</strong>
                        </div>
                        <div className="b-stat-chip">
                            <span>Herd Avg Activity</span>
                            <strong>58%</strong>
                        </div>
                    </div>
                </header>

                <div className="b-content">

                    {/* SECTION 1: Herd Activity Heatmap */}
                    <div className="dribbble-card b-section">
                        <div className="b-heat-header">
                            <div>
                                <h3 className="card-title">Herd Activity Heatmap (7-Day Average)</h3>
                                <p className="b-ai-text"><Sparkles size={16} /> <strong>AI Interpretation:</strong> {hasHistoricalData ? "This week's pattern shows normal grazing peaks at 06:00–09:00 and 16:00–19:00." : "Gathering baseline telemetry. Heatmap requires 7 days of historical data to generate reliable behaviour clusters."}</p>
                            </div>
                            {hasHistoricalData && (
                                <div className="b-legend">
                                    <span><div className="l-box d-green"></div> High (Grazing)</span>
                                    <span><div className="l-box l-green"></div> Moderate (Moving)</span>
                                    <span><div className="l-box amber"></div> Low (Resting)</span>
                                    <span><div className="l-box red"></div> Anomalous Low</span>
                                </div>
                            )}
                        </div>

                        <div className="b-heatmap-container" style={{ opacity: hasHistoricalData ? 1 : 0.4, pointerEvents: hasHistoricalData ? 'auto' : 'none' }}>
                            <div className="b-hm-y-labels">
                                {days.map(d => <div key={d}>{d}</div>)}
                            </div>
                            <div className="b-hm-grid">
                                {heatmapData.map((row, rIdx) => (
                                    <div key={rIdx} className="b-hm-row">
                                        {row.map((val, cIdx) => (
                                            <div key={`${rIdx}-${cIdx}`} className={`b-hm-cell ${val !== null ? `v-${val}` : 'v-empty'}`} style={{ background: val === null ? '#f1f5f9' : undefined }}>
                                                {val !== null && <div className="hm-tooltip">{days[rIdx]} {cIdx}:00 - {val === 3 ? 'Grazing' : val === 2 ? 'Moving' : val === 1 ? 'Resting' : 'Anomaly'}</div>}
                                            </div>
                                        ))}
                                    </div>
                                ))}
                                <div className="b-hm-x-labels">
                                    {hours.filter(h => h % 2 === 0).map(h => <div key={h} style={{ left: `${(h / 24) * 100}%` }}>{h}:00</div>)}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* SECTION 2: Behaviour Breakdown Cards */}
                    <div className="b-card-grid">
                        <div className="dribbble-card b-kpi-card">
                            <div className="kpi-top"><span>Grazing</span> <span className="delta eq">-</span></div>
                            <div className="kpi-val">{herdStats.grazing} <span>hrs/day</span></div>
                            <div className="kpi-context">Weekly Avg: Not enough data</div>
                            <div className="kpi-warn">Jersey baseline: 7-9 hrs daily</div>
                        </div>
                        <div className="dribbble-card b-kpi-card">
                            <div className="kpi-top"><span>Resting</span> <span className="delta eq">-</span></div>
                            <div className="kpi-val">{herdStats.resting} <span>hrs/day</span></div>
                            <div className="kpi-context">Weekly Avg: Not enough data</div>
                            <div className="kpi-warn" style={{ color: '#64748b' }}>Monitoring resting patterns...</div>
                        </div>
                        <div className="dribbble-card b-kpi-card">
                            <div className="kpi-top"><span>Moving</span> <span className="delta eq">-</span></div>
                            <div className="kpi-val">{herdStats.moving} <span>km/day</span></div>
                            <div className="kpi-context">Weekly Avg: Not enough data</div>
                            <div className="kpi-list">Tracking herd movement...</div>
                        </div>
                        <div className="dribbble-card b-kpi-card">
                            <div className="kpi-top"><span>Est. Rumination</span> <span className="delta eq">-</span></div>
                            <div className="kpi-val">{herdStats.rumination} <span>hrs/day</span></div>
                            <div className="kpi-context">Weekly Avg: Not enough data</div>
                            <div className="kpi-list">Calibrating rumination sensors...</div>
                        </div>
                    </div>

                    {/* SECTION 3: Individual Profiles */}
                    <h3 className="card-title" style={{ marginTop: 40, marginBottom: 16 }}>Individual Behaviour Profiles</h3>
                    <div className="b-profile-grid">
                        {individuals.map(ind => (
                            <div key={ind.id} className={`dribbble-card b-prof-card border-${ind.badge.toLowerCase()}`}>
                                <div className="prof-top">
                                    <div>
                                        <h4>{ind.id}</h4>
                                        <span>{ind.breed}</span>
                                    </div>
                                    <span className={`pill-status ${ind.badge === 'Anomaly' ? 'high' : ind.badge === 'Attention' ? 'medium' : 'low'}`}>{ind.badge}</span>
                                </div>

                                <div className="prof-flex-timeline">
                                    {ind.segments.map((seg, i) => (
                                        <div key={i} className={`flex-seg ${seg}`}></div>
                                    ))}
                                </div>
                                <div className="prof-time-labels"><span>00:00</span><span>12:00</span><span>23:59</span></div>

                                <div className="prof-stats-row">
                                    <div><span>Grazing</span><strong>{ind.g}</strong></div>
                                    <div><span>Distance</span><strong>{ind.d}</strong></div>
                                    <div><span>Resting</span><strong>{ind.r}</strong></div>
                                    <div><span>Vs Base</span><strong style={{ color: ind.base.includes('-') ? '#ef4444' : '#10b981' }}>{ind.base}</strong></div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* SECTION 4: Anomaly Timeline Feed */}
                    <div className="dribbble-card b-section" style={{ marginTop: 32 }}>
                        <h3 className="card-title">Behaviour Anomaly Feed</h3>
                        <div className="b-feed">
                            {anomalies.length > 0 ? anomalies.map((an, i) => (
                                <div key={i} className="feed-item">
                                    <div className="feed-time">{an.time}</div>
                                    <div className="feed-content">
                                        <div className="feed-head">
                                            <span className="feed-id">{an.id}</span>
                                            <span className="feed-type">{an.type}</span>
                                            <span className={`pill-status ${an.severity === 'High' ? 'high' : 'medium'}`}>{an.severity}</span>
                                        </div>
                                        <p className="feed-desc"><Database size={14} /> <strong>Raw Trigger:</strong> {an.desc}</p>
                                    </div>
                                </div>
                            )) : (
                                <div style={{ textAlign: 'center', padding: '32px', color: '#64748b' }}>
                                    <CheckCircle2 size={32} style={{ marginBottom: 12, color: '#10b981' }} />
                                    <p>No behavioural anomalies detected in your herd.</p>
                                </div>
                            )}
                        </div>
                    </div>

                    {/* SECTION 5 & 6 GRID */}
                    <div className="b-split-grid" style={{ marginTop: 32 }}>

                        {/* SECTION 5: Herd Social Network (Graphic) */}
                        <div className="dribbble-card b-dark-card">
                            <h3 className="card-title" style={{ color: 'white' }}><Network size={24} color="#10b981" /> Herd Social Network</h3>
                            <p style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem', marginBottom: 20 }}>GPS proximity matrix. Thick lines = Frequent companions. Withdrawal signals early illness.</p>

                            <div className="b-network-viz" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#1a362a' }}>
                                {hasHistoricalData ? (
                                    <>
                                        <svg width="100%" height="100%" style={{ position: 'absolute' }}>
                                            <line x1="20%" y1="30%" x2="50%" y2="50%" stroke="rgba(255,255,255,0.2)" strokeWidth="4" />
                                            <line x1="80%" y1="20%" x2="50%" y2="50%" stroke="rgba(255,255,255,0.4)" strokeWidth="6" />
                                            <line x1="50%" y1="50%" x2="70%" y2="80%" stroke="rgba(255,255,255,0.2)" strokeWidth="2" />
                                            <line x1="20%" y1="30%" x2="30%" y2="70%" stroke="rgba(239,68,68,0.5)" strokeWidth="1" strokeDasharray="4" />
                                        </svg>
                                        <div className="b-node" style={{ top: '30%', left: '20%', width: 40, height: 40, background: '#10b981' }}><span>GIR12</span></div>
                                        <div className="b-node" style={{ top: '50%', left: '50%', width: 60, height: 60, background: '#10b981' }}><span>LEAD</span></div>
                                        <div className="b-node" style={{ top: '20%', left: '80%', width: 45, height: 45, background: '#10b981' }}><span>MIX22</span></div>
                                        <div className="b-node" style={{ top: '80%', left: '70%', width: 35, height: 35, background: '#f59e0b' }}><span>SHW88</span></div>
                                        <div className="b-node anomaly" style={{ top: '70%', left: '30%', width: 30, height: 30, background: '#ef4444' }}><span>MIX05</span></div>
                                    </>
                                ) : (
                                    <div style={{ textAlign: 'center', padding: '40px', color: 'rgba(255,255,255,0.6)' }}>
                                        <Network size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
                                        <p>Social network mapping requires at least 48 hours of GPS proximity data.</p>
                                    </div>
                                )}
                            </div>

                            <div className="b-prox-table">
                                <h4 style={{ color: 'white', margin: '0 0 12px 0' }}>Top Proximity Drops (7-Day)</h4>
                                {hasHistoricalData ? (
                                    <>
                                        <div className="prox-row"><span>MIX005 ↔ LEAD</span> <strong style={{ color: '#ef4444' }}>-61%</strong></div>
                                        <div className="prox-row"><span>MIX005 ↔ GIR12</span> <strong style={{ color: '#ef4444' }}>-48%</strong></div>
                                        <div className="prox-row"><span>SHW088 ↔ MIX22</span> <strong style={{ color: '#f59e0b' }}>-22%</strong></div>
                                    </>
                                ) : (
                                    <div style={{ color: 'rgba(255,255,255,0.5)', fontSize: '0.9rem' }}>No proximity drops detected.</div>
                                )}
                            </div>
                        </div>

                        {/* SECTION 6: Correlation Chart */}
                        <div className="dribbble-card">
                            <h3 className="card-title">Behaviour vs Health Correlation</h3>
                            <div className="b-correl-chart" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                                {hasHistoricalData ? (
                                    <ResponsiveContainer width="100%" height="100%">
                                        <ComposedChart data={correlationData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                            <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f2f1" />
                                            <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#94a3b8' }} />
                                            <YAxis yAxisId="left" domain={[37, 40]} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#ef4444' }} />
                                            <YAxis yAxisId="right" orientation="right" domain={[0, 100]} axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#3b82f6' }} />
                                            <RechartsTooltip />
                                            <Bar yAxisId="right" dataKey="act" fill="#e0e7ff" radius={[4, 4, 0, 0]} name="Activity %" />
                                            <Line yAxisId="left" type="monotone" dataKey="temp" stroke="#ef4444" strokeWidth={3} dot={false} name="Body Temp (°C)" />
                                        </ComposedChart>
                                    </ResponsiveContainer>
                                ) : (
                                    <div style={{ textAlign: 'center', color: '#94a3b8' }}>
                                        <Activity size={32} style={{ marginBottom: 12, opacity: 0.5 }} />
                                        <p>Insufficient historical data to plot correlations.</p>
                                    </div>
                                )}
                            </div>
                            <p className="b-correl-text">
                                <strong>MIX005 History:</strong> Activity drops have preceded temperature spikes by an average of 16 hours across 3 recorded illness episodes. Activity is a confirmed leading indicator.
                            </p>
                        </div>
                    </div>

                    {/* SECTION 7: Benchmarks */}
                    <div className="dribbble-card b-section" style={{ marginTop: 32 }}>
                        <h3 className="card-title">Breed Behaviour Benchmarks</h3>
                        <div style={{ overflowX: 'auto' }}>
                            <table className="pred-light-table" style={{ minWidth: 800 }}>
                                <thead>
                                    <tr>
                                        <th>Breed</th>
                                        <th>Exp. Grazing</th>
                                        <th>Exp. Resting</th>
                                        <th>Exp. Distance</th>
                                        <th>Peak Window</th>
                                        <th>THI Ceiling</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {[...new Set(individuals.map(ind => ind.breed))].map((breed, i) => (
                                        <tr key={i}>
                                            <td style={{ fontWeight: 700 }}>{breed}</td>
                                            <td className="bg-amber">8 - 10 hrs (Current: 0.0)</td>
                                            <td>8 - 10 hrs</td>
                                            <td>5 - 8 km</td>
                                            <td>06:00 & 18:00</td>
                                            <td>80</td>
                                        </tr>
                                    ))}
                                    {individuals.length === 0 && (
                                        <tr>
                                            <td colSpan="6" style={{ textAlign: 'center', color: '#94a3b8' }}>Register livestock to view breed benchmarks.</td>
                                        </tr>
                                    )}
                                </tbody>
                            </table>
                        </div>
                    </div>

                    {/* SECTION 8: AI Recommendations */}
                    <div className="b-reco-grid" style={{ marginTop: 32 }}>
                        {hasHistoricalData ? (
                            <>
                                <div className="dribbble-card reco-card">
                                    <span className="opp-priority-tag urgent">Today</span>
                                    <h4>Isolation & Grazing Stop</h4>
                                    <p>MIX005 isolated for 3+ hours. Grazed only 1.2h today vs weekly avg 7.4h.</p>
                                    <div className="reco-actions">
                                        <button className="btn-quick-fix">Inspect Animal</button>
                                        <button className="btn-dismiss">Notify Vet</button>
                                    </div>
                                </div>
                                <div className="dribbble-card reco-card">
                                    <span className="opp-priority-tag plan">Planning</span>
                                    <h4>Herd Grazing Compressed</h4>
                                    <p>Morning grazing fell 34% due to early 34°C ambient heat. THI hit severe early.</p>
                                    <div className="reco-actions">
                                        <button className="btn-quick-fix">Relocate Note</button>
                                        <button className="btn-dismiss">Dismiss</button>
                                    </div>
                                </div>
                                <div className="dribbble-card reco-card">
                                    <span className="opp-priority-tag today">Review</span>
                                    <h4>Nocturnal Pacing</h4>
                                    <p>3 animals showing high activity after 23:00 over the past 4 nights in South Zone.</p>
                                    <div className="reco-actions">
                                        <button className="btn-quick-fix">Check Perimeter</button>
                                        <button className="btn-dismiss">Dismiss</button>
                                    </div>
                                </div>
                            </>
                        ) : (
                            <div className="dribbble-card" style={{ gridColumn: '1 / -1', textAlign: 'center', padding: '32px', color: '#64748b' }}>
                                <CheckCircle2 size={32} style={{ marginBottom: 12, color: '#10b981' }} />
                                <p style={{ margin: 0 }}>No AI behavioural recommendations at this time. Herd is operating within expected baseline limits.</p>
                            </div>
                        )}
                    </div>

                </div>
            </main>
        </div>
    );
}

export default BehaviourAnalysis;
