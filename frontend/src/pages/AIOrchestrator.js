import React, { useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    Plus, Settings, Paperclip, Image as ImageIcon, Mic, ArrowUp, PanelLeftClose,
    PanelLeft, Hexagon, Activity, Thermometer, MapPin, BarChart2, Bell, Copy, ThumbsUp, ThumbsDown, Sprout, AlertTriangle, Battery, PlusCircle, Leaf, ArrowRight, CheckCircle2, MoreHorizontal, Sparkles, Layers, Users, Calendar, Search, Edit3, Compass, Clock, Folder, LogOut, MessageSquare, ChevronRight, Menu
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './AIOrchestrator.css';

const UserAvatar = ({ initials }) => (
    <div className="aio-avatar">{initials}</div>
);

const MOCK_HISTORY = [
    { id: 1, group: 'Today', text: 'Aman temperature spike', emoji: '🌡️', time: '10:41 AM', active: true },
    { id: 2, group: 'Today', text: 'Herd density in Barn A', emoji: '🗺️', time: '08:15 AM', active: false },
    { id: 3, group: 'Yesterday', text: 'Weekly feed consumption', emoji: '📊', time: '04:30 PM', active: false },
    { id: 4, group: 'Last 7 Days', text: 'Geofence breach alert', emoji: '🔔', time: 'Tue', active: false },
];

const MOCK_AGENTS = [
    { name: 'Health Agent', status: 'online' },
    { name: 'Data Agent', status: 'online' },
    { name: 'Predictive Agent', status: 'busy' },
    { name: 'Automations', status: 'online' },
    { name: 'Nutrition', status: 'idle' },
    { name: 'Weather', status: 'online' },
];

const SUGGESTIONS = [
    { icon: <AlertTriangle size={16} strokeWidth={2.5} color="#f59e0b" />, title: 'Critical Alerts', text: 'Review the last 24hrs of alerts.', query: 'Show me any critical alerts from the last 24 hours.', span: 'col-span-1', grad: 'tint-alerts' },
    { icon: <PlusCircle size={16} strokeWidth={2.5} color="#3b82f6" />, title: 'Log Event', text: 'Log a calf birth in Barn B.', query: 'I need to log a new calf birth in Barn B.', span: 'col-span-1', grad: 'tint-log' },
    { icon: <Leaf size={16} strokeWidth={2.5} color="#14b8a6" />, title: 'Vaccinations', text: 'Animals due for BRD vax.', query: 'Which animals are due for their BRD vaccinations?', span: 'col-span-1', grad: 'tint-vax' },
    { icon: <MapPin size={16} strokeWidth={2.5} color="#8b5cf6" />, title: 'Geofence Check', text: 'Verify if any animals breached the north boundaries.', query: 'Have any animals breached the north geofence today?', span: 'col-span-1', grad: 'tint-geo' },
];

const renderPWM = (count) => {
    const heights = [16, 28, 40, 24, 32, 20, 36, 18, 26, 38, 22, 34];
    return (
        <div className="pwm-container">
            {Array.from({ length: count }).map((_, i) => (
                <div key={i} className="pwm-bar" style={{
                    height: `${heights[i % heights.length]}px`,
                    animationDuration: `${0.6 + (i % 3) * 0.2}s`,
                    animationDelay: `${i * 0.05}s`
                }}></div>
            ))}
        </div>
    );
};

// Mock response for streaming
const MOCK_AI_RESPONSE_VITALS = "Aman's internal temperature has risen to <span style='color:#ef4444;font-weight:700;'>39.8°C</span> over the last 2 hours. This is <span style='color:#ef4444;font-weight:700;'>+1.2°C</span> above his 30-day baseline.\n\nCombined with a 34% drop in rumination activity, this signature strongly correlates with early-onset Bovine Respiratory Disease (BRD).";

const MOCK_AI_RESPONSE_TRENDS = "Barn A's resting behavior shows a steady normal pattern over the last 24 hours. The forecast indicates optimal activity levels matching recent baselines.\n\nHere is the continuous activity forecast model projecting the next 12 hours.";

const MOCK_CHART_DATA = [
    { time: '12 AM', activity: 30 },
    { time: '4 AM', activity: 25 },
    { time: '8 AM', activity: 45 },
    { time: '12 PM', activity: 60 },
    { time: '4 PM', activity: 75 },
    { time: '8 PM', activity: 40 },
];

export default function AIOrchestrator() {
    const navigate = useNavigate();
    const location = useLocation();
    const [sidebarOpen, setSidebarOpen] = useState(true);
    const [input, setInput] = useState('');
    const [messages, setMessages] = useState([]);
    const [isStreaming, setIsStreaming] = useState(false);
    const [isThinking, setIsThinking] = useState(null);
    const [activeWidgetType, setActiveWidgetType] = useState(null);
    const [streamedWords, setStreamedWords] = useState([]);
    const [showFollowUps, setShowFollowUps] = useState(false);
    const [isDragOver, setIsDragOver] = useState(false);
    const [history, setHistory] = useState(MOCK_HISTORY);
    const [isRecording, setIsRecording] = useState(false);
    const [liveTranscript, setLiveTranscript] = useState('');

    const recognitionRef = useRef(null);
    const silenceTimeoutRef = useRef(null);
    const messagesEndRef = useRef(null);

    const startNewChat = () => {
        if (messages.length > 0) {
            const snippet = messages[0].content.substring(0, 30) + (messages[0].content.length > 30 ? '...' : '');
            const newItem = {
                id: Date.now(),
                group: 'Today',
                text: snippet,
                emoji: '💬',
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                active: true,
                savedMessages: [...messages]
            };
            setHistory(prev => [newItem, ...prev.map(h => ({ ...h, active: false }))]);
        }
        setMessages([]);
        setIsStreaming(false);
        setIsThinking(null);
        setActiveWidgetType(null);
        setStreamedWords([]);
        setShowFollowUps(false);
    };

    // Auto-start voice or auto-query if navigated with alert context
    useEffect(() => {
        if (location.state?.autoStartVoice) {
            navigate(location.pathname, { replace: true, state: {} });
            setTimeout(() => {
                if (!isRecording) {
                    toggleVoice();
                }
            }, 500);
        }
        if (location.state?.autoQuery) {
            const autoQuery = location.state.autoQuery;
            // Clear state so we don't trigger again on re-renders
            navigate(location.pathname, { replace: true, state: {} });
            setTimeout(() => {
                handleSend(autoQuery);
            }, 600);
        }
    }, []);

    const loadHistoryChat = (hItem) => {
        if (hItem.savedMessages) {
            setMessages(hItem.savedMessages);
        } else {
            // Restore a mock starting point for hardcoded original history items
            setMessages([{ role: 'user', content: hItem.text, time: hItem.time }]);
        }
        setHistory(prev => prev.map(h => ({ ...h, active: h.id === hItem.id })));
    };

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    // Voice Access Logic
    const startSilenceTimer = () => {
        if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);
        silenceTimeoutRef.current = setTimeout(() => {
            stopVoiceAndSubmit();
        }, 2500); // 2.5 seconds of silence logic
    };

    const stopVoiceAndSubmit = () => {
        if (recognitionRef.current) {
            recognitionRef.current.stop();
        }
        setIsRecording(false);
        if (silenceTimeoutRef.current) clearTimeout(silenceTimeoutRef.current);

        // Auto-submit if transcript is present
        setLiveTranscript(prev => {
            if (prev.trim()) {
                handleSend(prev.trim());
            }
            return '';
        });
    };

    const toggleVoice = () => {
        if (isRecording) {
            stopVoiceAndSubmit();
            return;
        }

        if (!('webkitSpeechRecognition' in window)) {
            alert('Web Speech API is not supported in this browser. Try Chrome.');
            return;
        }

        const recognition = new window.webkitSpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;
        recognition.lang = 'en-US';

        recognition.onstart = () => {
            setIsRecording(true);
            setLiveTranscript('Listening...');
            startSilenceTimer();
        };

        recognition.onresult = (event) => {
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {
                if (event.results[i].isFinal) {
                    interimTranscript += event.results[i][0].transcript;
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            setLiveTranscript(interimTranscript);
            startSilenceTimer(); // Reset silence clock on every word
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error', event.error);
            stopVoiceAndSubmit();
        };

        recognitionRef.current = recognition;
        recognition.start();
    };

    const renderWidget = (type) => {
        if (type === 'vitals') {
            return (
                <div className="aio-inline-card" style={{ animation: 'fadeInWord 0.5s ease forwards' }}>
                    <div className="aio-inline-grid">
                        <div className="aio-inline-stat">
                            <span className="aio-inline-label">Current Temp</span>
                            <span className="aio-inline-val" style={{ color: '#ef4444' }}>39.8°C</span>
                        </div>
                        <div className="aio-inline-stat">
                            <span className="aio-inline-label">Activity Level</span>
                            <span className="aio-inline-val" style={{ color: '#f59e0b' }}>-34%</span>
                        </div>
                    </div>
                </div>
            );
        }
        if (type === 'chart') {
            return (
                <div className="aio-inline-card" style={{ animation: 'fadeInWord 0.5s ease forwards', padding: '16px' }}>
                    <div style={{ marginBottom: '16px', fontSize: '13px', fontWeight: 600, color: '#4b5563' }}>Activity Trend Forecast</div>
                    <div style={{ width: '100%', height: 160 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={MOCK_CHART_DATA}>
                                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e5e7eb" />
                                <XAxis dataKey="time" axisLine={false} tickLine={false} tick={{ fontSize: 10, fill: '#9ca3af' }} dy={10} />
                                <Tooltip contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }} />
                                <Line type="monotone" dataKey="activity" stroke="#10b981" strokeWidth={3} dot={{ r: 4, fill: '#10b981', strokeWidth: 2, stroke: '#fff' }} activeDot={{ r: 6 }} />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            );
        }
        return null;
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages, streamedWords]);

    const handleSend = async (text) => {
        const query = text || input;
        if (!query.trim()) return;

        // Reset follow ups
        setShowFollowUps(false);

        setActiveWidgetType('vitals');

        // Add user message
        const newMsgs = [...messages, { role: 'user', content: query, time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }];
        setMessages(newMsgs);
        setInput('');

        // Start streaming sequence
        setIsStreaming(true);
        setIsThinking('Health Agent is analysing your query...');
        setStreamedWords([]);

        // Phase 1: Thinking transitions
        setTimeout(() => {
            setIsThinking('Data Agent is fetching telemetry...');
        }, 1500);

        setTimeout(() => {
            setIsThinking('Composing response...');
        }, 3000);

        // Phase 2: Call real API
        let responseText = '';
        try {
            const token = localStorage.getItem('token');
            const res = await fetch('http://localhost:8000/api/ai/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    Authorization: `Bearer ${token}`
                },
                body: JSON.stringify({
                    query,
                    history: messages.slice(-6).map(m => ({ role: m.role, content: m.content }))
                })
            });
            const json = await res.json();
            responseText = json.data?.response || json.response || json.data || 'I couldn\'t process that request. Please try again.';
        } catch (err) {
            console.error('AI chat error:', err);
            responseText = MOCK_AI_RESPONSE_VITALS; // Fallback to mock
        }

        // Phase 3: Stream the response word-by-word
        setTimeout(() => {
            setIsThinking(null);
            const words = responseText.split(/(\s+)/); // Split by whitespace while keeping it
            let currentWordIndex = 0;

            const interval = setInterval(() => {
                if (currentWordIndex < words.length) {
                    setStreamedWords(prev => [...prev, words[currentWordIndex]]);
                    currentWordIndex++;
                } else {
                    clearInterval(interval);

                    // Finalize the streamed message into the main chat history
                    setMessages(prev => [...prev, {
                        role: 'ai',
                        content: words.join(''),
                        time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                        widgetType: null
                    }]);

                    setIsStreaming(false);
                    // Phase 4: Show follow-ups
                    setTimeout(() => {
                        setShowFollowUps(true);
                        scrollToBottom();
                    }, 3000);
                }
            }, 30); // Fast word-by-word reveal

        }, 4000);
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    // Drag and drop handlers
    const handleDragOver = (e) => { e.preventDefault(); setIsDragOver(true); };
    const handleDragLeave = (e) => { e.preventDefault(); setIsDragOver(false); };
    const handleDrop = (e) => { e.preventDefault(); setIsDragOver(false); }; // Mock file drop

    return (
        <div className="aio-layout">
            {/* Shared SVG Gradients */}
            <svg width="0" height="0" style={{ position: 'absolute' }}>
                <defs>
                    <linearGradient id="gomataGrad" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#3b82f6" />
                        <stop offset="100%" stopColor="#10b981" />
                    </linearGradient>
                </defs>
            </svg>

            {/* Morva Style Sidebar with Nixtio Colors */}
            <aside className="aio-nixtio-sidebar">
                <div className="n-sidebar-brand-row">
                    <div className="n-brand-left">
                        <Activity size={24} color="#10b981" strokeWidth={3} />
                        <span className="n-brand-text">GoMata</span>
                    </div>
                    <button className="n-sidebar-collapse"><PanelLeftClose size={18} /></button>
                </div>

                <div className="n-sidebar-new-chat">
                    <button className="n-new-chat-btn" onClick={startNewChat}>
                        <MessageSquare size={16} /> New Chat
                    </button>
                </div>

                <div className="n-sidebar-workspace">
                    <div className="n-workspace-title">Workspace</div>

                    <button className="n-workspace-item">
                        <Folder size={16} /> New Project
                    </button>

                    <button className="n-workspace-item">
                        <Folder size={16} /> Pricing Section
                    </button>

                    <button className="n-workspace-item">
                        <Folder size={16} /> Design Guidelines
                    </button>

                    <button className="n-workspace-item">
                        <Folder size={16} /> Design Brief
                    </button>

                    <button className="n-workspace-item">
                        <Folder size={16} /> Marketing
                    </button>
                </div>

                <div className="n-sidebar-history">
                    <div className="n-history-title">Chat History</div>
                    <div className="n-history-scroll-area">
                        {history.map((item) => (
                            <button key={item.id} className={`n-history-item ${item.active ? 'active' : ''}`} onClick={() => loadHistoryChat(item)}>
                                <span className="n-history-text">{item.text}</span>
                            </button>
                        ))}
                    </div>
                </div>

                <div className="n-sidebar-bottom">
                    <button className="n-sidebar-settings">
                        <Settings size={16} /> Settings
                    </button>

                    <div className="n-sidebar-profile">
                        <UserAvatar initials="K" />
                        <span className="n-profile-name">Kanna</span>
                        <button className="n-profile-logout"><LogOut size={16} /></button>
                    </div>
                </div>
            </aside>

            {/* Center Area */}
            <main className={`aio-main ${messages.length === 0 && !isStreaming ? 'is-empty' : 'is-chat'}`}>

                <div className="aio-nixtio-topbar">
                    <div className="n-top-left">
                        <Activity size={16} color="#10b981" strokeWidth={3} />
                        <span>GoMata Assistant v2.6</span>
                    </div>
                    <div className="n-top-center">
                        <span>Daily GoMata</span>
                    </div>
                    <div className="n-top-right">
                        <button className="n-upgrade-btn"><Sparkles size={14} fill="currentColor" /> Upgrade</button>
                    </div>
                </div>

                <div className="aio-main-content">

                    {/* Centered Empty State Greeting */}
                    {messages.length === 0 && !isStreaming && (
                        <div className="aio-nixtio-stage">

                            <div className="aio-nixtio-header">
                                <h1 className="aio-nixtio-title">
                                    Hi Kanna, <span>Ready to</span><br />
                                    Achieve Great Things?
                                </h1>
                                <div className="aio-nixtio-hero-graphics">
                                    <div className="aio-robot-floating">
                                        <img src="/images/robot2.png" alt="A friendly robot assistant" />
                                    </div>
                                    <div className="aio-nixtio-floating-tag">
                                        <span style={{ fontSize: '14px', marginRight: '4px' }}>👋</span>
                                        <div>
                                            <span style={{ fontWeight: 600, color: '#111827' }}>Hey there!</span><br />
                                            Need a boost?
                                        </div>
                                    </div>
                                </div>
                            </div>

                            {isRecording ? (
                                <div className="aio-voice-canvas">
                                    <div className="voice-pulse-wrapper">
                                        {renderPWM(12)}
                                        <div className="voice-pulse-rings">
                                            <div className="v-ring r1"></div>
                                            <div className="v-ring r2"></div>
                                            <div className="v-ring r3"></div>
                                            <div className="v-ring center-core"><Mic size={32} color="white" /></div>
                                        </div>
                                        {renderPWM(12)}
                                    </div>
                                    <div className="voice-transcript">
                                        {liveTranscript}
                                    </div>
                                </div>
                            ) : (
                                <div className="aio-nixtio-cards-container">
                                    <div className="aio-n-card" onClick={() => handleSend("Experience Agentic AI Automations")}>
                                        <div className="n-card-icon-wrap orange">
                                            <Sparkles size={24} color="#f97316" />
                                        </div>
                                        <div style={{ fontWeight: 600, color: '#111827', marginBottom: '8px' }}>Experience Agentic AI Automations</div>
                                        <p style={{ fontWeight: 400, color: '#4b5563', lineHeight: '1.5', margin: '0 0 16px 0', fontSize: '14px' }}>Let AI add records, monitor livestock, generate reports, and execute tasks — automatically.</p>
                                        <span className="n-card-sub">Autonomous Actions</span>
                                    </div>

                                    <div className="aio-n-card" onClick={() => handleSend("Livestock Intelligence & Management")}>
                                        <div className="n-card-icon-wrap multi">
                                            <Activity size={24} color="#ec4899" />
                                        </div>
                                        <div style={{ fontWeight: 600, color: '#111827', marginBottom: '8px' }}>Livestock Intelligence & Management</div>
                                        <p style={{ fontWeight: 400, color: '#4b5563', lineHeight: '1.5', margin: '0 0 16px 0', fontSize: '14px' }}>Monitor health, track behavior, and manage your herd with real-time AI insights.</p>
                                        <span className="n-card-sub">Real-Time Insights</span>
                                    </div>

                                    <div className="aio-n-card" onClick={() => handleSend("Smart Farm Operations")}>
                                        <div className="n-card-icon-wrap blue">
                                            <Calendar size={24} color="#3b82f6" />
                                        </div>
                                        <div style={{ fontWeight: 600, color: '#111827', marginBottom: '8px' }}>Smart Farm Operations</div>
                                        <p style={{ fontWeight: 400, color: '#4b5563', lineHeight: '1.5', margin: '0 0 16px 0', fontSize: '14px' }}>Plan tasks, manage schedules, and keep your livestock operations organized effortlessly.</p>
                                        <span className="n-card-sub">Operational Control</span>
                                    </div>
                                </div>
                            )}
                        </div>
                    )}

                    {/* Chat Feed */}
                    {messages.length > 0 && (
                        <div className="aio-chat-feed">
                            {messages.map((m, i) => (
                                <div key={i} className={`aio-msg-row ${m.role}`}>
                                    <div className={`aio-msg-layout ${m.role}`}>
                                        {m.role === 'ai' && (
                                            <div className="aio-msg-avatar">
                                                <Activity size={16} color="white" strokeWidth={3} />
                                            </div>
                                        )}
                                        <div className="aio-msg-content">
                                            <div className="aio-msg-meta">{m.role === 'user' ? `You · ${m.time}` : ''}</div>

                                            <div className={`aio-msg-text-bubble ${m.role}`}>
                                                <div dangerouslySetInnerHTML={{ __html: m.content }} />
                                            </div>
                                            {m.role === 'ai' && m.widgetType && renderWidget(m.widgetType)}
                                            {m.role === 'ai' && (
                                                <div className="aio-msg-actions" style={{ animation: 'fadeInWord 0.5s ease forwards' }}>
                                                    <button className="aio-action-btn"><Copy size={14} /></button>
                                                    <button className="aio-action-btn"><ThumbsUp size={14} /></button>
                                                    <button className="aio-action-btn"><ThumbsDown size={14} /></button>
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            ))}

                            {/* Active Streaming Message */}
                            {isStreaming && (
                                <div className="aio-msg-row ai">
                                    {isThinking ? (
                                        <div className="aio-heartbeat-loader" style={{ marginLeft: '40px' }}>
                                            <div className="aio-multi-ecg">
                                                <Activity stroke="url(#gomataGrad)" strokeWidth={3} className="ecg-beat p1" />
                                                <Activity stroke="url(#gomataGrad)" strokeWidth={3} className="ecg-beat p2" />
                                                <Activity stroke="url(#gomataGrad)" strokeWidth={3} className="ecg-beat p3" />
                                                <Activity stroke="url(#gomataGrad)" strokeWidth={3} className="ecg-beat p4" />
                                            </div>
                                        </div>
                                    ) : (
                                        <div className="aio-msg-layout ai">
                                            <div className="aio-msg-avatar">
                                                <Activity size={16} color="white" strokeWidth={3} />
                                            </div>
                                            <div className="aio-msg-content">
                                                <div className="aio-msg-text-bubble ai">
                                                    {streamedWords.map((word, i) => (
                                                        word ? <span key={i} className={word.startsWith('<') ? '' : 'stream-word'} dangerouslySetInnerHTML={{ __html: word }} /> : null
                                                    ))}
                                                    <span className="stream-cursor" />
                                                </div>
                                                {streamedWords.length > 20 && activeWidgetType && renderWidget(activeWidgetType)}
                                            </div>
                                        </div>
                                    )}
                                </div>
                            )}

                            {/* Intelligent Follow-ups */}
                            {showFollowUps && (
                                <div className="aio-msg-row ai" style={{ paddingTop: 0 }}>
                                    <div className="aio-msg-layout ai">
                                        <div style={{ width: '28px', flexShrink: 0 }} /> {/* Spacer to align with text */}
                                        <div className="aio-msg-content">
                                            <div className="aio-followup-tray">
                                                <button className="aio-followup-chip" onClick={() => handleSend("Schedule vet visit for Aman")}>Schedule vet visit for Aman</button>
                                                <button className="aio-followup-chip" style={{ animationDelay: '0.1s' }} onClick={() => handleSend("View Aman's 7-day forecast")}>View Aman's 7-day forecast</button>
                                                <button className="aio-followup-chip" style={{ animationDelay: '0.2s' }} onClick={() => handleSend("Which other animals are at risk?")}>Which other animals are at risk?</button>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}

                            <div ref={messagesEndRef} />
                        </div>
                    )}
                </div>

                {/* Advanced Nixtio Input Area */}
                <div className="aio-input-area">
                    <div className="aio-nixtio-input-wrapper">

                        <div className="aio-nixtio-input-topbar">
                            <span className="n-top-left"><Sparkles size={14} /> Unlock more with Pro Plan</span>
                            <span className="n-top-right"><Settings size={14} /> Powered by GoMata v2.6</span>
                        </div>

                        <div className="aio-nixtio-input-box">
                            <button className="aio-icon-btn"><Plus size={18} /></button>
                            {isRecording && messages.length > 0 ? (
                                <div className="aio-input-pwm-wrap">
                                    {renderPWM(14)}
                                    <span className="aio-input-live-transcript">
                                        {liveTranscript || "Listening..."}
                                    </span>
                                </div>
                            ) : (
                                <textarea
                                    className="aio-textarea"
                                    placeholder='Example : "Analyze the resting behaviour trends for Barn A"'
                                    value={input}
                                    onChange={(e) => setInput(e.target.value)}
                                    onKeyDown={handleKeyDown}
                                    rows={1}
                                    style={{ display: (isRecording && messages.length === 0) ? 'none' : 'block' }}
                                />
                            )}
                            <div className="aio-nixtio-input-actions">
                                <button
                                    className={`aio-icon-btn ${isRecording ? 'recording-active' : ''}`}
                                    onClick={toggleVoice}
                                >
                                    <Mic size={18} color={isRecording ? '#ef4444' : 'currentColor'} />
                                </button>
                                <button
                                    className="aio-nixtio-send-btn"
                                    disabled={!input.trim()}
                                    onClick={() => handleSend()}
                                >
                                    <ArrowUp size={16} color="white" strokeWidth={3} />
                                </button>
                            </div>
                        </div>

                    </div>
                </div>

            </main>
        </div>
    );
}
