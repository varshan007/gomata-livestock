import React, { useEffect, useRef, useMemo } from 'react';
import './GoMataDivider.css';

const GoMataDivider = () => {
    const sectionRef = useRef(null);
    const textRef = useRef(null);
    const letterRefs = useRef([]);
    const text = "GoMata AI";
    const characters = useMemo(() => text.split(''), [text]);

    useEffect(() => {
        let ticking = false;

        const handleScroll = () => {
            if (!ticking) {
                window.requestAnimationFrame(() => {
                    if (!sectionRef.current) return;

                    const rect = sectionRef.current.getBoundingClientRect();
                    const windowHeight = window.innerHeight;

                    // Calculate scroll progress based on element entering the viewport
                    const distFromBottom = windowHeight - rect.top;

                    // Animate between the time it enters the bottom and reaches 75% of the screen
                    const totalDistForAnimation = windowHeight * 0.75;

                    let progress = 0;
                    if (distFromBottom > 0) {
                        progress = Math.min(Math.max(distFromBottom / totalDistForAnimation, 0), 1);
                    }

                    // Count total valid letter spans
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

                    // Add subtle parallax: text moves "down" while scrolling ("and then go down..")
                    if (textRef.current) {
                        // Ranges from roughly -0.5 to 0.5 as it scrolls across the center
                        const scrollCenterProgress = (windowHeight / 2 - rect.top) / windowHeight;
                        // Moves the text up and down by ~60px
                        textRef.current.style.transform = `translateY(${scrollCenterProgress * 80}px)`;
                    }

                    ticking = false;
                });
                ticking = true;
            }
        };

        window.addEventListener('scroll', handleScroll, { passive: true });
        handleScroll(); // set initial state

        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <section className="gomata-divider-section" ref={sectionRef}>
            <div className="gomata-divider-line"></div>
            <div className="gomata-divider-content" ref={textRef}>
                <h1 className="gomata-divider-headline">
                    {characters.map((char, i) => (
                        <span
                            key={i}
                            className="divider-char"
                            ref={el => {
                                if (el) {
                                    letterRefs.current[i] = el;
                                }
                            }}
                        >
                            {char === ' ' ? '\u00A0' : char}
                        </span>
                    ))}
                </h1>
            </div>
            <div className="gomata-divider-line"></div>
        </section>
    );
};

export default GoMataDivider;
