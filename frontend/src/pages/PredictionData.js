import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    LayoutDashboard, List, Activity, AlertTriangle, Menu, Search, Bell, Sparkles,
    Thermometer, HeartPulse, Clock, Database, CloudRain, Wind, Droplets, Info, Share2, Layers, Cpu, Globe, Users, Map as MapIcon, ChevronLeft, BarChart2
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, Cell, ReferenceLine } from 'recharts';
import AuthContext from '../context/AuthContext';
import './LivestockManagement.css'; // Standard Sidebar
import './PredictionData.css'; // Deep Dive Data Styling

const hourlySparkline = [
    { h: '00', t: 38.0 }, { h: '02', t: 38.1 }, { h: '04', t: 38.1 }, { h: '06', t: 38.3 },
    { h: '08', t: 38.5 }, { h: '10', t: 38.8 }, { h: '12', t: 39.1 }, { h: '14', t: 39.4 } // getting hot
];

const featureWeights = [
    { name: 'Body Temp Trajectory', value: 38, fill: '#ef4444' },
    { name: 'Ambient Temp + THI', value: 24, fill: '#f59e0b' },
    { name: 'Humidity Stress', value: 16, fill: '#fbbf24' },
    { name: 'Activity Drop', value: 12, fill: '#8b5cf6' },
    { name: 'Disease Zone Loc', value: 7, fill: '#3b82f6' },
    { name: 'Heart Rate', value: 3, fill: '#f43f5e' }
];

const timelineData = [
    { time: 'Today 14:00', temp: '39.4°C ↑', hr: '78 bpm', act: 'Low (-40%)', zone: 'Barn A', amb: '35.1°C', hum: '72%', thi: '84.2', src: 'MQTT', status: 'High Risk' },
    { time: 'Today 13:00', temp: '39.1°C', hr: '76 bpm', act: 'Low (-35%)', zone: 'Barn A', amb: '34.8°C', hum: '70%', thi: '83.5', src: 'MQTT', status: 'High Risk' },
    { time: 'Today 12:00', temp: '38.8°C', hr: '74 bpm', act: 'Normal (-10%)', zone: 'Barn A', amb: '34.0°C', hum: '68%', thi: '82.1', src: 'MQTT', status: 'Medium Risk' },
    { time: 'Today 06:00', temp: '--', hr: '--', act: '--', zone: '--', amb: '28.5°C', hum: '60%', thi: '76.0', src: 'API', status: 'API Sync' },
    { time: 'Yest 23:00', temp: '38.0°C', hr: '62 bpm', act: 'Normal (Night)', zone: 'North P.', amb: '--', hum: '--', thi: '--', src: 'AGG', status: 'Stable' },
];

function PredictionData() {
    const navigate = useNavigate();
    const { user } = React.useContext(AuthContext);
    const [isSidebarOpen, setIsSidebarOpen] = useState(true);
    const [activeTab, setActiveTab] = useState('hourly'); // hourly, daily, raw

    return (
        <div className="predictions-layout">
            {/* Standard GoMata Sidebar */}
            <aside className={`sidebar-premium ${!isSidebarOpen ? 'collapsed' : ''}`}>
                <div className="sidebar-logo">
                    <Activity size={28} className="brand-icon" />
                    <span>GoMata AI</span>
                </div>
                <div className="sidebar-scroll-container">
                    <nav className="sidebar-nav">
                        <div className="sidebar-section-title">Main</div>
                        <div className="nav-item-premium" onClick={() => navigate('/dashboard')}><LayoutDashboard className="nav-icon" /> <span>Overview</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/livestock')}><List className="nav-icon" /> <span>Farm & Livestock</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/map')}><MapIcon className="nav-icon" /> <span>Map Intelligence</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/health-analytics')}><Activity className="nav-icon" /> <span>Health Analytics</span></div>
                        <div className="nav-item-premium active" onClick={() => navigate('/predictions')}><Sparkles className="nav-icon" /> <span>AI Predictions</span></div>
                        <div className="sidebar-section-title">Operations</div>
                        <div className="nav-item-premium" onClick={() => navigate('/devices')}><Cpu className="nav-icon" /> <span>Devices</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/farm')}><Globe className="nav-icon" /> <span>Farms & Locations</span></div>
                        <div className="nav-item-premium" onClick={() => navigate('/staff')}><Users className="nav-icon" /> <span>Staff</span></div>
                    </nav>
                </div>
            </aside>

            {/* Main Content */}
            <main className="predictions-main pd-scroll-container">

                {/* Header */}
                <header className="pred-header">
                    <div className="pd-header-left">
                        <button className="icon-btn-premium" onClick={() => setIsSidebarOpen(!isSidebarOpen)}>
                            <Menu size={20} />
                        </button>
                        <button className="pd-back-btn" onClick={() => navigate('/predictions')}>
                            <ChevronLeft size={20} /> Back to Insights
                        </button>
                        <div className="pred-title-area pd-title-adjust">
                            <h1>Data Telemetry & Intelligence</h1>
                            <p>Deep-dive context into the AI Model</p>
                        </div>
                    </div>

                    <div className="pred-user-controls">
                        <div className="pred-control-icon"><Search size={20} /></div>
                        <div className="pred-control-icon"><Bell size={20} /></div>
                        <div className="pred-control-icon" style={{ background: '#10b981', color: 'white' }}>A</div>
                    </div>
                </header>

                <div className="pd-content">

                    {/* SECTION 1: Pipeline Overview Diagram */}
                    <div className="pd-section-block">
                        <h2 className="pd-section-title">How GoMata AI Makes This Prediction</h2>
                        <div className="pd-pipeline">
                            <div className="pd-pipe-step">
                                <div className="pipe-icon-box"><Database size={28} /></div>
                                <h4>Hardware Sensors</h4>
                                <p>IoT collars send body temp, GPS, heart rate every 1 hour via MQTT.</p>
                            </div>
                            <div className="pipe-arrow">→</div>
                            <div className="pd-pipe-step">
                                <div className="pipe-icon-box"><CloudRain size={28} /></div>
                                <h4>Weather & Environment API</h4>
                                <p>Ambient temp, humidity, THI, rainfall pulled once daily at 06:00 AM.</p>
                            </div>
                            <div className="pipe-arrow">→</div>
                            <div className="pd-pipe-step">
                                <div className="pipe-icon-box" style={{ background: '#e0e7ff', color: '#4f46e5' }}><Cpu size={28} /></div>
                                <h4>Data Processing Agent</h4>
                                <p>Synchronizes streams and checks against breed baseline z-scores.</p>
                            </div>
                            <div className="pipe-arrow">→</div>
                            <div className="pd-pipe-step">
                                <div className="pipe-icon-box" style={{ background: '#f0fdf4', color: '#10b981' }}><Sparkles size={28} /></div>
                                <h4>AI Model Synthesis</h4>
                                <p>Evaluates combined 48hr thermal footprint to generate prediction %.</p>
                            </div>
                        </div>
                    </div>

                    {/* SECTION 2: Data Freshness Status Row */}
                    <div className="pd-freshness-strip">
                        <div className="pd-fresh-chip">
                            <span className="dot fresh"></span>
                            <div className="fresh-details">
                                <strong>Body Temperature</strong>
                                <span>Last: 12m ago • Next: 48m</span>
                            </div>
                        </div>
                        <div className="pd-fresh-chip">
                            <span className="dot fresh"></span>
                            <div className="fresh-details">
                                <strong>GPS Location</strong>
                                <span>Last: 12m ago • Next: 48m</span>
                            </div>
                        </div>
                        <div className="pd-fresh-chip">
                            <span className="dot fresh"></span>
                            <div className="fresh-details">
                                <strong>Heart Rate</strong>
                                <span>Last: 12m ago • Next: 48m</span>
                            </div>
                        </div>
                        <div className="pd-fresh-chip">
                            <span className="dot amber"></span>
                            <div className="fresh-details">
                                <strong>Weather API (THI)</strong>
                                <span>Updated: 06:00 AM • Next: Tmrrw</span>
                            </div>
                        </div>
                        <div className="pd-fresh-chip">
                            <span className="dot fresh"></span>
                            <div className="fresh-details">
                                <strong>Activity Accelerator</strong>
                                <span>Continuous sync (Hourly Aggr)</span>
                            </div>
                        </div>
                    </div>

                    {/* SECTION 3: Today's Environmental Parameters (Weather API) */}
                    <div className="pd-weather-grid">
                        <div className="pd-w-card">
                            <div className="pd-w-head"><Thermometer size={20} /> Ambient Temperature</div>
                            <div className="pd-w-val">34.2°C</div>
                            <div className="pd-w-sub" style={{ color: '#ef4444' }}>+3.1°C above seasonal average</div>
                            <p className="pd-w-impact">Exacerbates core body temperature regulation in confined spaces.</p>
                        </div>
                        <div className="pd-w-card">
                            <div className="pd-w-head"><Droplets size={20} /> Relative Humidity</div>
                            <div className="pd-w-val">72%</div>
                            <div className="pd-w-sub" style={{ color: '#f59e0b' }}>High — Evaporative stress risk</div>
                            <p className="pd-w-impact">Humidity > 70% critically limits sweating efficiency for Jersey breeds.</p>
                        </div>
                        <div className="pd-w-card">
                            <div className="pd-w-head"><Wind size={20} /> Wind Vector</div>
                            <div className="pd-w-val">12 km/h SE</div>
                            <div className="pd-w-sub" style={{ color: '#10b981' }}>Moderate airflow</div>
                            <p className="pd-w-impact">Provides minor convective cooling but blocked in Barn A infrastructure.</p>
                        </div>
                        <div className="pd-w-card">
                            <div className="pd-w-head"><Layers size={20} /> THI Index (Computed)</div>
                            <div className="pd-w-val" style={{ color: '#ef4444' }}>84.2</div>
                            <div className="pd-w-sub" style={{ color: '#ef4444' }}>SEVERE STRESS</div>
                            <p className="pd-w-impact">Above 80 THI directly damages herd productivity and immunity.</p>
                        </div>
                    </div>

                    {/* TWO COLUMN DEEP DIVES & AI REASONING */}
                    <div className="pd-split-wrapper">

                        {/* LEFT COLUMN: Hardware Deep Dive & Timeline */}
                        <div className="pd-main-col">

                            {/* Section 4: Hardware Sensors Deep Dive */}
                            <div className="dribbble-card pd-compact-card">
                                <h3 className="card-title">IoT Collar Telemetry (MIX005)</h3>
                                <div className="pd-telemetry-grid">
                                    <div className="tel-box">
                                        <span>Body Temp</span>
                                        <strong>39.4°C</strong>
                                        <div className="tel-bar-wrap"><div className="tel-bar red" style={{ width: '90%' }}></div></div>
                                    </div>
                                    <div className="tel-box">
                                        <span>Heart Rate</span>
                                        <strong>78 bpm</strong>
                                        <div className="tel-bar-wrap"><div className="tel-bar amber" style={{ width: '70%' }}></div></div>
                                    </div>
                                    <div className="tel-box">
                                        <span>Activity</span>
                                        <strong>-40%</strong>
                                        <div className="tel-bar-wrap"><div className="tel-bar blue" style={{ width: '40%' }}></div></div>
                                    </div>
                                    <div className="tel-box">
                                        <span>GPS Loc</span>
                                        <strong>Barn A</strong>
                                        <div className="tel-bar-wrap"><div className="tel-bar green" style={{ width: '100%' }}></div></div>
                                    </div>
                                </div>
                                <div className="tel-sparkline-box">
                                    <h4>24hr Thermal Footprint (Hourly Arrays)</h4>
                                    <div style={{ height: 120, width: '100%' }}>
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={hourlySparkline}>
                                                <XAxis dataKey="h" axisLine={false} tickLine={false} fontSize={10} stroke="#94a3b8" />
                                                <RechartsTooltip />
                                                <ReferenceLine y={39.0} stroke="#ef4444" strokeDasharray="3 3" />
                                                <Bar dataKey="t" radius={[4, 4, 0, 0]}>
                                                    {hourlySparkline.map((entry, index) => (
                                                        <Cell key={`cell-${index}`} fill={entry.t >= 39.0 ? '#ef4444' : entry.t >= 38.5 ? '#f59e0b' : '#10b981'} />
                                                    ))}
                                                </Bar>
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </div>
                            </div>

                            {/* Section 5: Timeline Data Table */}
                            <div className="dribbble-card pd-compact-card" style={{ marginTop: 32 }}>
                                <div className="pd-table-header">
                                    <h3 className="card-title" style={{ margin: 0 }}>Time-Series Data Manifest</h3>
                                    <div className="pd-tabs">
                                        <button className={`pd-tab ${activeTab === 'hourly' ? 'active' : ''}`} onClick={() => setActiveTab('hourly')}>Hourly</button>
                                        <button className={`pd-tab ${activeTab === 'daily' ? 'active' : ''}`} onClick={() => setActiveTab('daily')}>Daily AGG</button>
                                        <button className={`pd-tab ${activeTab === 'raw' ? 'active' : ''}`} onClick={() => setActiveTab('raw')}>Raw Stream</button>
                                    </div>
                                </div>

                                <div className="pd-table-scroll">
                                    <table className="pred-light-table pd-mini-table">
                                        <thead>
                                            <tr>
                                                <th>Timestamp</th>
                                                <th>Temp</th>
                                                <th>HR</th>
                                                <th>Act</th>
                                                <th>Loc</th>
                                                <th>THI</th>
                                                <th>Source</th>
                                                <th>Status</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            {timelineData.map((row, i) => (
                                                <tr key={i}>
                                                    <td style={{ color: '#6b7d73', fontSize: '0.8rem' }}>{row.time}</td>
                                                    <td style={{ fontWeight: row.temp.includes('↑') ? 800 : 500, color: row.temp.includes('↑') ? '#ef4444' : 'inherit' }}>{row.temp}</td>
                                                    <td>{row.hr}</td>
                                                    <td>{row.act}</td>
                                                    <td>{row.zone}</td>
                                                    <td style={{ fontWeight: 700 }}>{row.thi}</td>
                                                    <td><span className="pd-src-badge">{row.src}</span></td>
                                                    <td><span className={`pill-status ${row.status.includes('High') ? 'high' : row.status.includes('API') ? 'low' : 'medium'}`} style={{ fontSize: '0.65rem' }}>{row.status}</span></td>
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                                <div className="pd-table-footer">
                                    <span>168 hourly readings & 7 daily API snapshots utilized.</span>
                                    <button className="pred-action-btn"><Share2 size={14} /> Export CSV</button>
                                </div>
                            </div>

                        </div>


                        {/* RIGHT COLUMN: AI Brain & Explanations */}
                        <div className="pd-side-col">

                            {/* Section 7: Prediction Confidence Breakdown (Visual Eye-catcher) */}
                            <div className="dribbble-card pd-confidence-hero">
                                <div className="pd-hero-head">
                                    <div>
                                        <h2 style={{ margin: 0, color: 'white', fontSize: '1.4rem' }}>MIX005</h2>
                                        <span style={{ color: 'rgba(255,255,255,0.7)', fontSize: '0.9rem' }}>Jersey • 3 Yrs • Barn A</span>
                                    </div>
                                    <div className="huge-conf">91%</div>
                                </div>
                                <p className="pd-hero-explain">
                                    Based on 168 contextual telemetry readings, GoMata AI estimates a 91% probability of Severe Clinical Fever (>39.5°C) within 48 hours absent intervention.
                                </p>

                                <div className="pd-conf-factors">
                                    <div className="pd-fact">
                                        <div className="fact-top"><span>Thermal Trajectory</span><span className="pill-status high">High Cert.</span></div>
                                        <p>Rapid 0.8°C climb unlinked to ambient rise.</p>
                                    </div>
                                    <div className="pd-fact">
                                        <div className="fact-top"><span>Env. Stress (THI)</span><span className="pill-status high">High Cert.</span></div>
                                        <p>Severe 84.2 index suppressing natural cooling.</p>
                                    </div>
                                    <div className="pd-fact">
                                        <div className="fact-top"><span>Breed Vulnerability</span><span className="pill-status medium">Med Cert.</span></div>
                                        <p>Jerseys tolerate heat poorly vs local breeds.</p>
                                    </div>
                                </div>
                            </div>

                            {/* Section 6: Feature Weights */}
                            <div className="dribbble-card pd-compact-card">
                                <h3 className="card-title"><BarChart2 size={24} color="#3b82f6" /> Model Feature Weights</h3>
                                <div className="pd-weights-chart" style={{ height: 200, width: '100%', marginBottom: 16 }}>
                                    <ResponsiveContainer width="100%" height="100%">
                                        <BarChart layout="vertical" data={featureWeights} margin={{ top: 0, right: 30, left: 40, bottom: 0 }}>
                                            <XAxis type="number" hide />
                                            <YAxis dataKey="name" type="category" width={110} tick={{ fontSize: 10, fill: '#6b7d73' }} axisLine={false} tickLine={false} />
                                            <RechartsTooltip cursor={{ fill: '#f4f6f5' }} />
                                            <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={12}>
                                                {featureWeights.map((entry, index) => (
                                                    <Cell key={`cell-${index}`} fill={entry.fill} />
                                                ))}
                                            </Bar>
                                        </BarChart>
                                    </ResponsiveContainer>
                                </div>
                                <p className="pd-weights-text">
                                    <strong>AI Rationalization:</strong> Core body temperature velocity (38%) compounding with severe THI stress (24%) accounts for 62% of the predictive urgency. The animal's behavioral shutdown (-40% activity) confirmed the algorithm's clinical flag.
                                </p>
                            </div>

                        </div>
                    </div>

                    {/* Section 8: Data Dictionary Footer */}
                    <div className="pd-dictionary">
                        <h4>GoMata Data Dictionary</h4>
                        <div className="dict-grid">
                            <div className="dict-item"><strong>MQTT:</strong> Hardware collar data stream synced over cellular/WiFi directly to GoMata backend at 1-hour intervals.</div>
                            <div className="dict-item"><strong>API:</strong> External environmental and meteorological data (OpenWeatherMap) pulled daily resolving local micro-climates.</div>
                            <div className="dict-item"><strong>THI (Temp-Humidity Index):</strong> The critical computed formula merging heat & moisture. >72=Mild Stress, >80=Severe Stress.</div>
                            <div className="dict-item"><strong>AGG:</strong> Aggregated smoothing values (using rolling Z-scores) to eliminate hardware noise from raw streams.</div>
                        </div>
                    </div>

                </div>
            </main>
        </div>
    );
}

export default PredictionData;
