import React, { useState, useEffect, useRef } from 'react';
import { Mic, X, Activity, Loader, MicOff, Settings, Send, RefreshCcw } from 'lucide-react';
import './VoiceAssistant.css';
import { aiAPI } from '../services/api';

const VoiceAssistant = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [isListening, setIsListening] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [messages, setMessages] = useState([]);
    const [language, setLanguage] = useState('en-IN');
    const [processing, setProcessing] = useState(false);
    const [availableVoices, setAvailableVoices] = useState([]);
    const [transcript, setTranscript] = useState('');

    const recognitionRef = useRef(null);
    const synthRef = useRef(window.speechSynthesis);
    const messagesEndRef = useRef(null);

    // Language Configuration
    const languages = {
        'en-IN': {
            label: 'ENG',
            title: "Voice Assistant",
            hero: (<span>How can I help<br />your farm today?</span>),
            greeting: "Namaste! I'm listening.",
            suggestions: [
                "Health of Bessie?",
                "Critical alerts?",
                "Where is Raju?"
            ]
        },
        'hi-IN': {
            label: 'हिंदी',
            title: "वॉइस असिस्टेंट",
            hero: (<span>आज मैं आपकी<br />क्या मदद करूँ?</span>),
            greeting: "नमस्ते! मैं सुन रही हूँ।",
            suggestions: [
                "Bessie की तबीयत?",
                "कोई खतरा है?",
                "Raju कहाँ है?"
            ]
        }
    };

    const currentLang = languages[language];

    useEffect(() => {
        // Scroll to bottom of chat
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messages]);

    useEffect(() => {
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognitionRef.current = new SpeechRecognition();
            recognitionRef.current.continuous = false;
            recognitionRef.current.interimResults = true; // Enable interim to show "typing" effect

            recognitionRef.current.onresult = (event) => {
                const isFinal = event.results[0].isFinal;
                const text = event.results[0][0].transcript;

                if (isFinal) {
                    setTranscript('');
                    handleUserQuery(text);
                } else {
                    setTranscript(text);
                }
            };

            recognitionRef.current.onend = () => {
                setIsListening(false);
                setTranscript('');
            };

            // Error handling
            recognitionRef.current.onerror = (e) => {
                console.error("Mic Error", e);
                setIsListening(false);
            };
        }

        const loadVoices = () => setAvailableVoices(window.speechSynthesis.getVoices());
        window.speechSynthesis.onvoiceschanged = loadVoices;
        loadVoices();

        return () => { window.speechSynthesis.onvoiceschanged = null; };
    }, [language]); // Re-init if language changes (though usually just setting lang property is enough)

    const toggleAssistant = () => {
        setIsOpen(!isOpen);
    };

    const resetConversation = () => {
        setMessages([]);
        setTranscript('');
        stopListening();
        if (synthRef.current) synthRef.current.cancel();
    };

    const startListening = () => {
        if (isSpeaking) {
            synthRef.current.cancel();
            setIsSpeaking(false);
        }
        if (recognitionRef.current) {
            recognitionRef.current.lang = language;
            recognitionRef.current.start();
            setIsListening(true);
        }
    };

    const stopListening = () => {
        recognitionRef.current?.stop();
        setIsListening(false);
    };

    const handleUserQuery = async (text) => {
        setMessages(prev => [...prev, { role: 'user', content: text }]);
        setProcessing(true);

        try {
            // Pass language explicitly to backend
            const history = messages.map(m => ({ role: m.role, content: m.content }));
            const { data } = await aiAPI.voiceChat(text, history, language);

            const aiResponse = data.response;
            setMessages(prev => [...prev, { role: 'ai', content: aiResponse }]);
            speak(aiResponse);

        } catch (error) {
            console.error("AI Error:", error);
            const errorMsg = language === 'hi-IN' ? "कनेक्शन में समस्या है।" : "Connection error.";
            setMessages(prev => [...prev, { role: 'ai', content: errorMsg }]);
            speak(errorMsg);
        } finally {
            setProcessing(false);
        }
    };

    const speak = (text) => {
        if (!synthRef.current) return;
        synthRef.current.cancel();

        const utterance = new SpeechSynthesisUtterance(text);

        // Robust Voice Selection
        let preferredVoice = availableVoices.find(v =>
            v.lang === language || (language === 'en-IN' && v.lang.includes('India'))
        );

        // Strict fallback for Hindi
        if (language === 'hi-IN' && !preferredVoice) {
            preferredVoice = availableVoices.find(v => v.lang.includes('hi'));
        }

        if (preferredVoice) utterance.voice = preferredVoice;
        utterance.rate = 0.9;
        utterance.onstart = () => setIsSpeaking(true);
        utterance.onend = () => setIsSpeaking(false);

        synthRef.current.speak(utterance);
    };

    return (
        <>
            {/* Launcher Button (Hidden when open) */}
            <button
                className={`voice-fab ${isOpen ? 'active' : ''}`}
                onClick={toggleAssistant}
                aria-label="Open Voice Assistant"
            >
                <Mic size={28} />
            </button>

            {/* Full Overlay Container */}
            <div className={`voice-overlay ${isOpen ? 'open' : ''}`}>
                <div className={`voice-panel ${isListening || isSpeaking ? 'listening' : ''}`}>

                    {/* Header */}
                    <div className="voice-header">
                        <div className="status-badge">
                            <div className={`status-dot ${processing ? 'active' : ''}`}></div>
                            {processing ? 'GOMATA AI Thinking...' : currentLang.title}
                        </div>
                        <div className="header-right">
                            <span className="app-title-small">GoMata AI Voice</span>
                            <button className="icon-btn" onClick={resetConversation} title="Reset Chat">
                                <RefreshCcw size={18} />
                            </button>
                            <button className="close-btn-ghost" onClick={() => setIsOpen(false)}>
                                <X size={24} />
                            </button>
                        </div>
                    </div>

                    {/* Chat Area */}
                    <div className={`chat-area ${messages.length > 0 ? 'scrollable' : 'centered'}`}>
                        {messages.length === 0 ? (
                            <div className="hero-text">
                                {currentLang.hero}
                            </div>
                        ) : (
                            <div className="bubble-container">
                                {messages.map((msg, idx) => (
                                    <div key={idx} className={`bubble ${msg.role}`}>
                                        {msg.content}
                                    </div>
                                ))}
                                {transcript && (
                                    <div className="bubble user preview">
                                        {transcript}...
                                    </div>
                                )}
                                {processing && (
                                    <div className="bubble ai typing">
                                        <Loader size={16} className="spin" />
                                    </div>
                                )}
                                <div ref={messagesEndRef} />
                            </div>
                        )}
                    </div>

                    {/* Footer Controls */}
                    <div className="voice-footer">

                        {/* Suggestions (Only if chat empty) */}
                        {messages.length === 0 && (
                            <div className="suggestions-row">
                                {currentLang.suggestions.map((s, i) => (
                                    <div key={i} className="suggestion-chip" onClick={() => handleUserQuery(s)}>
                                        {s}
                                    </div>
                                ))}
                            </div>
                        )}

                        {/* Visualizer */}
                        <div className={`visualizer ${isListening || isSpeaking ? 'active' : ''}`}>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                            <div className="bar"></div>
                        </div>

                        {/* Main Mic & Toggles */}
                        <div style={{ display: 'flex', alignItems: 'center', gap: '20px' }}>
                            <div className="lang-toggles">
                                <button
                                    className={`lang-btn ${language === 'en-IN' ? 'active' : ''}`}
                                    onClick={() => setLanguage('en-IN')}
                                >
                                    ENG
                                </button>
                                <button
                                    className={`lang-btn ${language === 'hi-IN' ? 'active' : ''}`}
                                    onClick={() => setLanguage('hi-IN')}
                                >
                                    हिंदी
                                </button>
                            </div>

                            <button
                                className={`main-mic ${isListening ? 'listening' : ''}`}
                                onClick={isListening ? stopListening : startListening}
                            >
                                {isListening ? <MicOff size={32} /> : <Mic size={32} />}
                            </button>
                        </div>

                    </div>
                </div>
            </div>
        </>
    );
};

export default VoiceAssistant;
