import React, { useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './SpiralHero.css';

const SpiralHero = () => {
    const navigate = useNavigate();
    const heroRef = useRef(null);
    const letterRefs = useRef([]);

    const [isLoaded, setIsLoaded] = useState(false);

    // Headline broken into 4 lines for staggered animation
    const lines = [
        "AGENTIC AI",
        "AUTONOMOUS",
        "LIVESTOCK",
        "INTELLIGENCE"
    ];

    useEffect(() => {
        // Trigger initial fade up animation reliably
        const timer = setTimeout(() => {
            setIsLoaded(true);
        }, 100);

        let ticking = false;

        const handleScroll = () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    if (!heroRef.current) return;

                    const rect = heroRef.current.getBoundingClientRect();
                    const scrolled = Math.max(0, -rect.top);
                    const scrollableDistance = rect.height - window.innerHeight;

                    let progress = 0;
                    if (scrollableDistance > 0) {
                        // Complete the fade-in at 90% of the sticky scroll distance
                        progress = Math.min(Math.max(scrolled / (scrollableDistance * 0.9), 0), 1);
                    }

                    const totalLetters = letterRefs.current.length;
                    const activeCount = Math.min(
                        Math.floor(progress * (totalLetters + 0.99)),
                        totalLetters
                    );

                    letterRefs.current.forEach((el, index) => {
                        if (el) {
                            if (index < activeCount) {
                                el.style.color = '#FFFFFF';
                            } else {
                                el.style.color = '#5A5A5A';
                            }
                        }
                    });

                    ticking = false;
                });
                ticking = true;
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll();

        return () => {
            window.removeEventListener('scroll', handleScroll);
            clearTimeout(timer);
        };
    }, []);

    let letterIndex = 0;

    const cards = [
        "/media__1772005322247.png",
        "/media__1772005322264.png",
        "/media__1772005343675.png",
        "/media__1772005343692.png",
        "/media__1772005363324.png"
    ];

    return (
        <div className="spiral-hero-wrapper" ref={heroRef}>
            <div className="spiral-hero-container">
                {/* Left Section - Text */}
                <div className="spiral-text-section">
                    <h1 className="spiral-headline">
                        {lines.map((line, lIdx) => (
                            <span
                                key={lIdx}
                                className={`spiral-line ${isLoaded ? 'loaded' : ''}`}
                                style={{ transitionDelay: `${lIdx * 150}ms` }}
                            >
                                {line.split('').map((char, cIdx) => {
                                    const currentIndex = letterIndex++;
                                    // preserve spaces mapping
                                    return (
                                        <span
                                            key={currentIndex}
                                            className="spiral-letter"
                                            ref={(el) => (letterRefs.current[currentIndex] = el)}
                                            style={char === ' ' ? { paddingRight: '0.25em' } : {}}
                                        >
                                            {char}
                                        </span>
                                    );
                                })}
                            </span>
                        ))}
                    </h1>
                    <p className="spiral-subtitle">Predicts, Protects, and Optimizes.</p>
                    <button className="spiral-cta" onClick={() => navigate('/get-started')}>
                        Get Started &rarr;
                    </button>
                </div>

                {/* Right Section - Spiral */}
                <div className="spiral-visual-section">
                    <div className="spiral-glow"></div>
                    <div className="spiral-container">
                        <div className="spiral-float-wrapper">
                            <div className="spiral-axis">
                                {cards.map((src, i) => (
                                    <div className="spiral-card" key={i}>
                                        <img src={src} alt={`GoMata UI Card ${i + 1}`} loading="lazy" />
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default SpiralHero;
