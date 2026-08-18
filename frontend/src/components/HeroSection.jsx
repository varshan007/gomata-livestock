import React, { useEffect, useRef, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import './HeroSection.css';

const HeroSection = () => {
    const navigate = useNavigate();
    const heroWrapperRef = useRef(null);
    const heroStickyRef = useRef(null);
    const textRef = useRef(null);
    const letterRefs = useRef([]);
    const imageRef = useRef(null);

    const headlineText = "Autonomous Intelligence\nfor Livestock that\nPredicts, Protects, and\nOptimizes.";
    const characters = useMemo(() => headlineText.split(''), [headlineText]);
    const numLetters = characters.filter(c => c !== '\n' && c !== ' ').length; // Or include spaces? We can just filter out \n

    useEffect(() => {
        let ticking = false;

        const handleScroll = () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    if (!heroWrapperRef.current || !heroStickyRef.current) return;

                    const wrapperRect = heroWrapperRef.current.getBoundingClientRect();
                    const stickyHeight = heroStickyRef.current.offsetHeight;

                    // The distance the wrapper can be scrolled while pinned
                    const scrollableDistance = wrapperRect.height - stickyHeight;

                    let progress = 0;
                    if (scrollableDistance > 0) {
                        const scrolledIntoWrapper = Math.max(0, -wrapperRect.top);
                        // Complete the animation slightly before the end of the sticky wrapper
                        const rawProgress = scrolledIntoWrapper / (scrollableDistance * 0.85);
                        progress = Math.min(Math.max(rawProgress, 0), 1);
                    }

                    const scrollY = window.scrollY;

                    // How many span elements actually correspond to valid letters/spaces
                    // Since letterRefs captures all actual span elements
                    let totalValidSpans = 0;
                    letterRefs.current.forEach(el => { if (el) totalValidSpans++; });

                    const activeCount = Math.floor(progress * totalValidSpans);
                    let validIndex = 0;

                    letterRefs.current.forEach((el) => {
                        if (el) {
                            if (validIndex < activeCount) {
                                el.classList.add('active');
                            } else {
                                el.classList.remove('active');
                            }
                            validIndex++;
                        }
                    });

                    // Transform resistance: up to -40px
                    if (textRef.current) {
                        textRef.current.style.transform = `translateY(${progress * -40}px)`;
                    }

                    // Parallax for image: translate down based on global scroll
                    if (imageRef.current) {
                        imageRef.current.style.transform = `translateY(${scrollY * 0.1}px)`;
                    }

                    ticking = false;
                });
                ticking = true;
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        // Trigger once to set initial state
        handleScroll();

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <div className="premium-hero-wrapper" ref={heroWrapperRef}>
            <section className="premium-hero" ref={heroStickyRef}>
                <div className="premium-hero-container">
                    <div className="premium-hero-text" ref={textRef}>
                        <h1 className="hero-headline">
                            {characters.map((char, i) => {
                                if (char === '\n') return <br key={`br-${i}`} />;
                                return (
                                    <span
                                        key={i}
                                        className="letter"
                                        ref={el => {
                                            if (el) {
                                                letterRefs.current[i] = el;
                                            }
                                        }}
                                    >
                                        {char}
                                    </span>
                                );
                            })}
                        </h1>

                        <button
                            className="premium-cta-link"
                            onClick={() => navigate('/get-started')}
                        >
                            Start Monitoring &rarr;
                        </button>
                    </div>

                    <div className="premium-hero-image-wrapper">
                        <img
                            ref={imageRef}
                            src="https://images.unsplash.com/photo-1542272201-b1ca555f8505?q=80&w=1200&auto=format&fit=crop"
                            alt="Dynamic livestock intelligence"
                            className="premium-hero-img"
                        />
                    </div>
                </div>
            </section>
        </div>
    );
};

export default HeroSection;
