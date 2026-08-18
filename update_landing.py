import sys

target_file = '/Users/googledoodle/Downloads/livestock_monitoring/frontend/src/pages/Landing.js'

with open(target_file, 'r') as f:
    lines = f.readlines()

new_ui = """        <div className="lr-page">
            <header className="lr-header">
                <div className="lr-logo">Palmer®</div>
                <div className="lr-links">
                    <span>Quick Links</span>
                    <a onClick={() => window.scrollTo(0,0)}>Home</a>
                    <a href="#featured">Gallery</a>
                    <a href="#services">Work</a>
                    <a onClick={openLogin}>Login / Contact</a>
                </div>
                <div className="lr-location">
                    <span>Based in Tokyo 東京</span>
                    <p>Art Director + Framer Developer</p>
                </div>
            </header>

            <section className="lr-hero">
                <div className="lr-hero-text">
                    <h1>Pattern Dimensions<br/>and Moments that<br/>Connect and Leave a<br/>Bold イメージ.</h1>
                </div>
                <div className="lr-hero-image">
                    <img src="https://images.unsplash.com/photo-1542272201-b1ca555f8505?q=80&w=1200&auto=format&fit=crop" alt="Dynamic pose" />
                    <div className="lr-hero-badge">X</div>
                </div>
            </section>

            <section className="lr-marquee">
                <h1>Akihiko™</h1>
                <button className="lr-use-btn" onClick={openRegister}>
                    ↓ Use for Free
                </button>
            </section>

            <div className="lr-portrait-top">
                <span>© CURATED INTERFACES ビジュアル</span>
                <span>(WDX® — 02)</span>
                <span>DIGITAL DESIGNER</span>
            </div>

            <section className="lr-portrait-section">
                <div className="lr-portrait-img">
                    <img src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?q=80&w=800&auto=format&fit=crop" alt="Portrait" />
                </div>
                <div className="lr-portrait-content">
                    <div className="lr-portrait-text">
                        <h2>13+ years™ of digital form,<br/>sharp interactions, and<br/>relentless creative discipline<br/>and effort.</h2>
                        <button className="lr-contact-btn" onClick={openRegister}>CONTACT</button>
                    </div>
                    <div className="lr-practice-bar" style={{borderBottom: '1px solid var(--border-color)', borderTop: 'none'}}>
                        <span>Digital Nomad</span>
                        <span>Creative Developer</span>
                    </div>
                    <div className="lr-logos-grid">
                        <div className="lr-logo-box">Cairo</div>
                        <div className="lr-logo-box large">oslo.</div>
                        <div className="lr-logo-box">Chain</div>
                        <div className="lr-logo-box large">Manila.</div>
                        <div className="lr-logo-box" style={{fontFamily: 'cursive', textTransform: 'lowercase'}}>ther</div>
                    </div>
                </div>
            </section>

            <section id="featured" className="lr-featured">
                <h1 className="lr-featured-header">Featured Works©</h1>
                <div className="lr-featured-content">
                    <div className="lr-featured-desc">
                        <p>Every project is a chance to blend design and development, shaping bold interactive ideas into <strong>sleek digital realities — built with</strong> intent, speed, and visual clarity that attracts lot of peoples.</p>
                        <button className="lr-see-works" onClick={openRegister}>SEE WORKS</button>
                    </div>
                    <div className="lr-featured-cards">
                        <div className="lr-card" style={{marginTop: '40px'}}>
                            <img src="https://images.unsplash.com/photo-1505740420928-5e560c06d30e?q=80&w=800&auto=format&fit=crop" alt="Headphones" style={{aspectRatio: '3/4'}}/>
                        </div>
                        <div className="lr-card">
                            <img src="https://images.unsplash.com/photo-1512413914856-17b5f2597284?q=80&w=800&auto=format&fit=crop" alt="Halo Wear" />
                            <div className="lr-card-info">
                                <span>Halo Wear</span>
                                <span>(02)</span>
                            </div>
                        </div>
                    </div>
                </div>
            </section>

            <section id="services" className="lr-services">
                <h1 className="lr-services-header">Services<sup>(6)</sup></h1>
                <div className="lr-services-bar">
                    <span>Precise</span>
                    <span>Structured</span>
                    <span>Focused</span>
                    <span>Visual Language</span>
                </div>
                <div className="lr-services-list">
                    <div className="lr-service-item">
                        <div className="lr-service-num">01</div>
                        <div className="lr-service-title">Art<br/>Direction</div>
                        <div className="lr-service-desc">We guide every visual decision from start to finish, ensuring clarity, emotion, and impact across every touchpoint.</div>
                    </div>
                    <div className="lr-service-item">
                        <div className="lr-service-num">02</div>
                        <div className="lr-service-title">Brand<br/>Identity</div>
                        <div className="lr-service-desc">From strategy to execution, we shape consistent brand systems that speak clearly and feel uniquely ownable.</div>
                    </div>
                    <div className="lr-service-item">
                        <div className="lr-service-num">03</div>
                        <div className="lr-service-title">Motion<br/>Direction</div>
                        <div className="lr-service-desc">We use motion as a design tool — adding clarity, rhythm, and energy to digital experiences with intention.</div>
                    </div>
                    <div className="lr-service-item">
                        <div className="lr-service-num">04</div>
                        <div className="lr-service-title">Framer<br/>Sites</div>
                        <div className="lr-service-desc">Design meets execution with real-time, scalable websites — all crafted natively inside Framer for speed and precision.</div>
                    </div>
                </div>
            </section>

            <div className="lr-practice-top">
                <span>© EXPERIENCE エクスペリエンス</span>
                <span>(WDX® — 05)</span>
                <span>DIGITAL CRAFT</span>
            </div>

            <section className="lr-practice">
                <div className="lr-practice-main">
                    <h2>Practice.</h2>
                    <div className="lr-practice-img">
                        <img src="https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?q=80&w=600&auto=format&fit=crop" alt="Small Portrait" />
                    </div>
                </div>
                <div className="lr-practice-bar">
                    <span>Creative Collabs</span>
                    <span>Studio</span>
                    <span>Creative Partners</span>
                </div>
                <div className="lr-table">
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Clavmen<br/>Studio</div>
                        <div className="lr-table-col2">2022 - present</div>
                        <div className="lr-table-col3">Art Director &<br/>Designer</div>
                        <div className="lr-table-col4">Tokyo</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Modular Eight</div>
                        <div className="lr-table-col2">2020 - 2022</div>
                        <div className="lr-table-col3">Senior<br/>Developer</div>
                        <div className="lr-table-col4">Osaka</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Haus of Signal</div>
                        <div className="lr-table-col2">2018 - 2020</div>
                        <div className="lr-table-col3">Creative<br/>Technologist</div>
                        <div className="lr-table-col4">Berlin</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Studio Orbit</div>
                        <div className="lr-table-col2">2016 - 2018</div>
                        <div className="lr-table-col3">UI/UX Designer</div>
                        <div className="lr-table-col4">Dallas</div>
                    </div>
                    <div className="lr-table-row">
                        <div className="lr-table-col1">Novaform Labs</div>
                        <div className="lr-table-col2">2014 - 2016</div>
                        <div className="lr-table-col3">Junior<br/>Designer</div>
                        <div className="lr-table-col4">Kyoto</div>
                    </div>
                </div>
            </section>
"""

start_idx = -1
end_idx = -1

for i, line in enumerate(lines):
    if '<div className="landing-page">' in line:
        start_idx = i
        break

for i in range(start_idx, len(lines)):
    if '{/* MODALS */}' in lines[i]:
        end_idx = i
        break

if start_idx != -1 and end_idx != -1:
    lines[start_idx:end_idx] = [new_ui + "\n"]
    with open(target_file, 'w') as f:
        f.writelines(lines)
    print("Successfully replaced main content.")
else:
    print("Could not find start or end block")
