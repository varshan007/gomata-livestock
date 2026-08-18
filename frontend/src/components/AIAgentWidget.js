import React, { useState, useEffect, useRef } from 'react';
import { Mic, Send, Activity, MapPin, Plus, AlertTriangle, BarChart2, Settings, Sparkles, Zap, DollarSign, Move } from 'lucide-react';
import './AIAgentWidget.css';
import { aiAPI } from '../services/api';

const AIAgentWidget = ({ onQuickAction }) => {
    const [activeAgent, setActiveAgent] = useState('Orchestrator'); // Orchestrator, Health, Location, Movement, Production
    const [state, setState] = useState('idle'); // idle, listening, thinking, success, warning
    const [query, setQuery] = useState('');
    const [responseMsg, setResponseMsg] = useState('');
    const [isListening, setIsListening] = useState(false);
    const recognitionRef = useRef(null);

    // Initialize Speech Recognition
    useEffect(() => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = false;
            recognitionRef.current.interimResults = false;
            recognitionRef.current.lang = 'en-IN';

            recognitionRef.current.onresult = (event) => {
                const text = event.results[0][0].transcript;
                setQuery(text);
                handleQuery(text);
            };

            recognitionRef.current.onend = () => {
                setIsListening(false);
                if (state === 'listening') setState('thinking');
            };
        }
    }, []);

    const startListening = () => {
        setIsListening(true);
        setState('listening');
        recognitionRef.current?.start();
    };

    // Text-to-Speech Function
    const speak = (text) => {
        if (!window.speechSynthesis) return;

        // Cancel any ongoing speech
        window.speechSynthesis.cancel();

        const utterance = new SpeechSynthesisUtterance(text);
        utterance.lang = 'en-IN'; // Indian English
        utterance.rate = 1.0;
        utterance.pitch = 1.1; // Slightly higher pitch for a "cleaner" tone

        // Try to select a "catchy" voice (Female preferred)
        const voices = window.speechSynthesis.getVoices();
        const preferredVoice = voices.find(v => v.name.includes('Google') && v.name.includes('Female'))
            || voices.find(v => v.name.includes('Samantha'))
            || voices.find(v => v.lang === 'en-IN');

        if (preferredVoice) utterance.voice = preferredVoice;

        window.speechSynthesis.speak(utterance);
    };

    const handleQuery = async (text) => {
        if (!text.trim()) return;
        setState('thinking');
        setResponseMsg('');
        setActiveAgent('Orchestrator');

        // Speak processing status
        speak("Analyzing your request...");

        try {
            // Simulate agent routing visualization
            setTimeout(() => {
                if (text.toLowerCase().includes('health') || text.toLowerCase().includes('fever') || text.toLowerCase().includes('sick')) {
                    setActiveAgent('Health');
                } else if (text.toLowerCase().includes('where') || text.toLowerCase().includes('location') || text.toLowerCase().includes('map')) {
                    setActiveAgent('Location');
                } else if (text.toLowerCase().includes('walk') || text.toLowerCase().includes('move') || text.toLowerCase().includes('grazing')) {
                    setActiveAgent('Movement');
                } else if (text.toLowerCase().includes('profit') || text.toLowerCase().includes('milk') || text.toLowerCase().includes('weight')) {
                    setActiveAgent('Production');
                }
            }, 1000);

            const history = [];
            const response = await aiAPI.voiceChat(text, history, 'en-IN');

            console.log("AI Response:", response.data.response);
            setResponseMsg(response.data.response);
            setState('success');

            // Speak the final response
            speak(response.data.response);

            setQuery('');

        } catch (error) {
            console.error("AI Error", error);
            setState('warning');
            const errorMsg = "I'm having trouble connecting to the farm network.";
            setResponseMsg(errorMsg);
            speak(errorMsg);
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter') {
            handleQuery(query);
        }
    };

    const handleQuickActionClick = (action) => {
        // Trigger specific queries based on action
        let prompt = "";
        switch (action) {
            case 'health': prompt = "Check overall herd health status."; break;
            case 'locations': prompt = "Where are my animals right now?"; break;
            case 'add': prompt = "I need to register a new animal."; break;
            case 'alerts': prompt = "Show me critical alerts."; break;
            case 'analytics': prompt = "Show production profitability report."; break;
            case 'settings': prompt = "Open settings."; break;
            default: prompt = "";
        }
        setQuery(prompt);
        handleQuery(prompt);
        if (onQuickAction) onQuickAction(action);
    };

    // Agent Configs
    const agents = {
        Orchestrator: { color: '#8B5CF6', icon: <Sparkles size={16} />, label: "Orchestrator" },
        Health: { color: '#10B981', icon: <Activity size={16} />, label: "Health Agent" },
        Location: { color: '#06B6D4', icon: <MapPin size={16} />, label: "Location Agent" },
        Movement: { color: '#F59E0B', icon: <Move size={16} />, label: "Movement Agent" },
        Production: { color: '#3B82F6', icon: <DollarSign size={16} />, label: "Production Agent" },
    };

    const currentAgent = agents[activeAgent] || agents.Orchestrator;

    return (
        <div className="ai-agent-widget">
            {/* Header */}
            <div className="agent-header">
                <div className="agent-title" style={{ color: currentAgent.color }}>
                    {currentAgent.icon}
                    <span>{currentAgent.label}</span>
                </div>
                <div className="agent-status">
                    <div className={`status-dot-pulse ${state}`} style={{ background: state === 'idle' ? '#ccc' : currentAgent.color }}></div>
                    <span>{state === 'idle' ? 'Ready' : state.charAt(0).toUpperCase() + state.slice(1)}</span>
                </div>
            </div>

            {/* Orb Visualization */}
            <div className="orb-container">
                <div className={`ai-orb ${state} ${activeAgent.toLowerCase()}`}>
                    <div className="orb-core" style={{ boxShadow: `0 0 30px ${currentAgent.color}40` }}></div>
                    <div className="orb-glow" style={{ background: `radial-gradient(circle, ${currentAgent.color}20 0%, transparent 70%)` }}></div>
                    <div className="orb-particles"></div>
                </div>
            </div>

            {/* Response Area */}
            <div className="agent-response-area">
                {state === 'thinking' && <p className="typing-text">Analyzing data...</p>}
                {state === 'success' && responseMsg && (
                    <div className="response-fade-in">
                        "{responseMsg}"
                    </div>
                )}
                {state === 'idle' && !responseMsg && (
                    <p className="placeholder-text">"How can I help you regarding Health, Location, or Production?"</p>
                )}
            </div>

            {/* Input Area */}
            <div className="agent-input-area">
                <div className="input-capsule">
                    <button
                        className={`mic-btn ${isListening ? 'active' : ''}`}
                        onClick={startListening}
                    >
                        <Mic size={18} />
                    </button>
                    <input
                        type="text"
                        placeholder="Ask anything..."
                        value={query}
                        onChange={(e) => setQuery(e.target.value)}
                        onKeyDown={handleKeyDown}
                        disabled={state === 'thinking'}
                    />
                    <button
                        className={`send-btn ${query.trim() ? 'active' : ''}`}
                        onClick={() => handleQuery(query)}
                        disabled={!query.trim()}
                    >
                        <Send size={16} />
                    </button>
                </div>
            </div>

            {/* Quick Actions Pills */}
            <div className="quick-actions-grid">
                <button className="q-pill" onClick={() => handleQuickActionClick('health')}>
                    <Activity size={14} className="text-emerald" /> <span>Health Status</span>
                </button>
                <button className="q-pill" onClick={() => handleQuickActionClick('locations')}>
                    <MapPin size={14} className="text-cyan" /> <span>Locations</span>
                </button>
                <button className="q-pill" onClick={() => handleQuickActionClick('add')}>
                    <Plus size={14} className="text-violet" /> <span>Add Animal</span>
                </button>
                <button className="q-pill" onClick={() => handleQuickActionClick('alerts')}>
                    <AlertTriangle size={14} className="text-amber" /> <span>Alerts</span>
                </button>
                <button className="q-pill" onClick={() => handleQuickActionClick('analytics')}>
                    <DollarSign size={14} className="text-blue" /> <span>Production</span>
                </button>
                <button className="q-pill" onClick={() => handleQuickActionClick('settings')}>
                    <Move size={14} className="text-slate" /> <span>Movement</span>
                </button>
            </div>
        </div>
    );
};

export default AIAgentWidget;
