import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './PlatformSection.css';

const PlatformSection = () => {
    const navigate = useNavigate();
    const sectionRef = useRef(null);
    const imageRef = useRef(null);

    useEffect(() => {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    // We only want the animation to occur once as they scroll down
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.15, // Trigger when 15% of element is in view
            rootMargin: '0px 0px -50px 0px'
        });

        const elementsToAnimate = sectionRef.current?.querySelectorAll('.animate-on-scroll');
        elementsToAnimate?.forEach(el => observer.observe(el));

        return () => {
            elementsToAnimate?.forEach(el => observer.unobserve(el));
        };
    }, []);

    useEffect(() => {
        let ticking = false;

        const handleScroll = () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    const scrollY = window.scrollY;
                    if (imageRef.current) {
                        // Apply the requested parallax effect
                        imageRef.current.style.transform = `translateY(${scrollY * 0.08}px)`;
                    }
                    ticking = false;
                });
                ticking = true;
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll(); // Initial call

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const gridItems = [
        "Health Intelligence",
        "Movement Intelligence",
        "Device Intelligence",
        "Breeding Intelligence",
        "Production Intelligence",
        "Financial Intelligence"
    ];

    return (
        <section className="platform-section" ref={sectionRef}>
            <div className="platform-bg-text">GoMaTA</div>

            <div className="platform-container">
                <div className="platform-left animate-on-scroll fade-left">
                    <img
                        ref={imageRef}
                        src="/media__1772008277259.jpg"
                        alt="Herd of livestock"
                        className="platform-image"
                        style={{ objectPosition: 'center 20%' }}
                    />
                </div>

                <div className="platform-right">
                    <div className="cinematic-glow"></div>

                    <div className="platform-content-wrapper">
                        <div className="platform-label animate-on-scroll fade-up">
                            AUTONOMOUS LIVESTOCK INTELLIGENCE PLATFORM
                        </div>

                        <h2 className="platform-headline animate-on-scroll fade-up" style={{ animationDelay: '0.1s' }}>
                            Building the Intelligence Layer<br />
                            for Autonomous Livestock Operations.
                        </h2>

                        <button
                            className="platform-cta animate-on-scroll fade-up"
                            style={{ animationDelay: '0.2s' }}
                            onClick={() => navigate('/get-started')}
                        >
                            Start Monitoring &rarr;
                        </button>

                        <div className="platform-grid">
                            {gridItems.map((item, index) => (
                                <div
                                    key={index}
                                    className="platform-grid-item animate-on-scroll fade-up"
                                    style={{ animationDelay: `${0.3 + (index * 0.1)}s` }}
                                >
                                    {item}
                                </div>
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default PlatformSection;
