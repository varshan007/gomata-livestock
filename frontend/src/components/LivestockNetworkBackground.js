import React, { useEffect, useRef } from 'react';

const LivestockNetworkBackground = () => {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];
        // Adjust density based on screen size in resize function
        let particleCount = 40;
        const connectionDistance = 140;
        const mouseDistance = 150;

        const emojis = ['🐄', '🐑', '🐖', '🐓', '🐐', '📡', '🧬'];

        let mouse = { x: null, y: null };

        const resize = () => {
            // Check if parent exists
            if (canvas.parentElement) {
                width = canvas.width = canvas.parentElement.offsetWidth;
                height = canvas.height = canvas.parentElement.offsetHeight;

                // Adjust count for smaller screens
                if (width < 768) {
                    particleCount = 20;
                } else {
                    particleCount = 50;
                }
                init(); // Re-init particles on resize to avoid clustering
            }
        };

        class Particle {
            constructor() {
                this.x = Math.random() * width;
                this.y = Math.random() * height;
                this.vx = (Math.random() - 0.5) * 0.8;
                this.vy = (Math.random() - 0.5) * 0.8;
                this.baseSize = width < 768 ? 20 : 30; // Smaller size
                this.size = this.baseSize;
                this.emoji = emojis[Math.floor(Math.random() * emojis.length)];
            }

            update() {
                this.x += this.vx;
                this.y += this.vy;

                // Wrap around edges
                if (this.x < 0) this.x = width;
                if (this.x > width) this.x = 0;
                if (this.y < 0) this.y = height;
                if (this.y > height) this.y = 0;

                // Mouse interaction (Repel slightly for interactivity)
                if (mouse.x != null) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < mouseDistance) {
                        const forceDirectionX = dx / distance;
                        const forceDirectionY = dy / distance;
                        const force = (mouseDistance - distance) / mouseDistance;

                        // Push away
                        const directionX = forceDirectionX * force * 1.5;
                        const directionY = forceDirectionY * force * 1.5;

                        this.x -= directionX;
                        this.y -= directionY;
                    }
                }

                // Dynamic Size Scaling on Hover
                let targetSize = this.baseSize;
                if (mouse.x != null) {
                    let dx = mouse.x - this.x;
                    let dy = mouse.y - this.y;
                    let distance = Math.sqrt(dx * dx + dy * dy);
                    if (distance < mouseDistance) {
                        targetSize = this.baseSize * 1.5; // Grow 50%
                    }
                }
                // Smooth transition
                this.size += (targetSize - this.size) * 0.1;
            }

            draw() {
                ctx.font = `${this.size}px Arial`;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                // Increased transparency for background feel
                ctx.globalAlpha = 0.3; // Much more transparent
                ctx.fillStyle = '#ffffff'; // White text
                ctx.fillText(this.emoji, this.x, this.y);
            }
        }

        const init = () => {
            particles = [];
            for (let i = 0; i < particleCount; i++) {
                particles.push(new Particle());
            }
        };

        const animate = () => {
            ctx.clearRect(0, 0, width, height);

            // Draw connections
            ctx.globalAlpha = 1; // Keep lines fully visible (relative to their own opacity)
            for (let a = 0; a < particles.length; a++) {
                for (let b = a; b < particles.length; b++) {
                    let dx = particles[a].x - particles[b].x;
                    let dy = particles[a].y - particles[b].y;
                    let distance = Math.sqrt(dx * dx + dy * dy);

                    if (distance < connectionDistance) {
                        // Interactive edges: Stronger if near mouse
                        let isNearMouse = false;
                        if (mouse.x != null) {
                            // Check distance of edge midpoint to mouse
                            const midX = (particles[a].x + particles[b].x) / 2;
                            const midY = (particles[a].y + particles[b].y) / 2;
                            const distToMouse = Math.sqrt((mouse.x - midX) ** 2 + (mouse.y - midY) ** 2);
                            if (distToMouse < 100) isNearMouse = true;
                        }

                        const opacity = 1 - (distance / connectionDistance);

                        ctx.beginPath();
                        if (isNearMouse) {
                            ctx.strokeStyle = `rgba(100, 180, 255, ${opacity})`; // Brighter Blue highlight
                            ctx.lineWidth = 1.5;
                        } else {
                            ctx.strokeStyle = `rgba(50, 255, 150, ${opacity * 0.5})`; // Brighter/Visible Green
                            ctx.lineWidth = 0.5;
                        }

                        ctx.moveTo(particles[a].x, particles[a].y);
                        ctx.lineTo(particles[b].x, particles[b].y);
                        ctx.stroke();
                    }
                }
            }

            // Draw particles
            particles.forEach(p => {
                p.update();
                p.draw();
            });

            requestAnimationFrame(animate);
        };

        // Resize observer for more robust resizing
        const resizeObserver = new ResizeObserver(() => {
            // Defers execution to avoid "ResizeObserver loop completed with undelivered notifications"
            window.requestAnimationFrame(() => {
                resize();
            });
        });

        if (canvas.parentElement) {
            resizeObserver.observe(canvas.parentElement);
        }

        const handleMouseMove = (e) => {
            const rect = canvas.getBoundingClientRect();
            mouse.x = e.clientX - rect.left;
            mouse.y = e.clientY - rect.top;
        };

        const handleMouseLeave = () => {
            mouse.x = null;
            mouse.y = null;
        }

        // Attach listeners to window if overlay covers it, or canvas itself
        // Canvas is safer if z-index is handled
        window.addEventListener('mousemove', handleMouseMove); // Tracking window for smooth effect even if text is over
        // But we need to offset relative to canvas
        // Actually, attaching to canvas paretn or window is tricky if we want exact coordinates relative to canvas
        // Best: attach to window, but map to canvas coordinates

        // Let's stick to canvas listener but ensure pointer-events: none is used on overlay text
        // NO wait, if I put pointer-events: none on text, they can't select text. 
        // Better: Listen on the container of the hero section.

        resize();
        init();
        animate();

        return () => {
            resizeObserver.disconnect();
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    return <canvas ref={canvasRef} className="livestock-network-canvas" />;
};

export default LivestockNetworkBackground;
