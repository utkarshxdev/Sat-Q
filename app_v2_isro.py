"""
SatQuery AI - High Fidelity ISRO UI
Based on the provided design mockup.
"""
import streamlit as st
from textwrap import dedent

st.set_page_config(
    page_title="SatQuery AI | ISRO Command Center",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── GLOBAL CSS OVERRIDES ──────────────────────────────────────────────────
st.html("""
<style>
    /* Base Theme & Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #050A14 !important; /* Deepest space blue */
        color: #F8FAFC !important;
    }
    
    /* Hide top padding and streamline */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0 !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #070F1E !important;
        border-right: 1px solid #14233A;
        width: 280px !important;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: transparent !important;
        border: 1px solid #14233A !important;
        color: #F8FAFC !important;
        border-radius: 4px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        border-color: #FF671F !important;
        color: #FF671F !important;
    }
    
    /* Primary Orange Button */
    .primary-btn>div>button {
        background: linear-gradient(135deg, #FF7B00, #FF5500) !important;
        border: none !important;
        color: #FFF !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 12px rgba(255, 103, 31, 0.3) !important;
    }
    
    /* Custom Headers */
    .section-title {
        color: #64748B;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-top: 24px;
        margin-bottom: 12px;
    }
    
    /* Sidebar Menu Items */
    .menu-item {
        display: flex;
        align-items: center;
        padding: 10px 12px;
        margin-bottom: 4px;
        border-radius: 4px;
        cursor: pointer;
        color: #94A3B8;
        font-size: 0.9rem;
    }
    .menu-item.active {
        background-color: rgba(255, 103, 31, 0.1);
        color: #F8FAFC;
        border: 1px solid #1E3150;
    }
    .menu-item:hover:not(.active) {
        background-color: #0A162B;
    }
    
    /* Custom Map Container */
    .map-container {
        background: url('https://images.unsplash.com/photo-1541873676-a18131494184?q=80&w=2000&auto=format&fit=crop') center/cover;
        position: relative;
        height: 600px;
        border-radius: 6px;
        border: 1px solid #14233A;
        overflow: hidden;
        margin-bottom: 20px;
        box-shadow: inset 0 0 100px rgba(5,10,20,0.8);
    }
    
    .map-overlay-grid {
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-image: 
            linear-gradient(rgba(20, 35, 58, 0.3) 1px, transparent 1px),
            linear-gradient(90deg, rgba(20, 35, 58, 0.3) 1px, transparent 1px);
        background-size: 40px 40px;
    }
    
    /* Split Line & Handle */
    .split-line {
        position: absolute;
        left: 50%;
        top: 0;
        bottom: 0;
        width: 2px;
        background-color: #FF671F;
        z-index: 10;
        box-shadow: 0 0 10px rgba(255, 103, 31, 0.8);
    }
    .split-handle {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background-color: #FF671F;
        color: white;
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
        z-index: 11;
        cursor: ew-resize;
    }

    /* Floating Panels on Map */
    .glass-panel {
        background: rgba(7, 15, 30, 0.85);
        backdrop-filter: blur(10px);
        border: 1px solid #1E3150;
        border-radius: 6px;
        padding: 16px;
        position: absolute;
        z-index: 20;
    }
    
    .panel-layer-stack { top: 20px; right: 60px; width: 280px; }
    .panel-change-region { bottom: 20px; left: 20px; }
    .panel-classification { bottom: 20px; right: 60px; }
    
    /* Map Toolbar */
    .map-toolbar {
        position: absolute;
        right: 15px;
        top: 250px;
        background: rgba(7, 15, 30, 0.9);
        border: 1px solid #1E3150;
        border-radius: 6px;
        display: flex;
        flex-direction: column;
        z-index: 20;
    }
    .map-toolbar div {
        padding: 8px 12px;
        border-bottom: 1px solid #14233A;
        cursor: pointer;
        color: #94A3B8;
        text-align: center;
    }
    .map-toolbar div:hover { color: #FFF; }
    .map-toolbar div:last-child { border-bottom: none; }
    
    /* Bottom Stats Row */
    .stat-card {
        background-color: #070F1E;
        border: 1px solid #14233A;
        border-radius: 6px;
        padding: 16px;
        text-align: left;
    }
    .stat-label { color: #64748B; font-size: 0.75rem; font-weight: 600; text-transform: uppercase; margin-bottom: 4px; }
    .stat-value { color: #F8FAFC; font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
    .stat-sub { font-size: 0.75rem; display: flex; align-items: center; gap: 4px; }
    .text-green { color: #10B981; }
    .text-cyan { color: #00B4D8; }

    /* Chat Section */
    .chat-container {
        background-color: #070F1E;
        border: 1px solid #14233A;
        border-radius: 6px;
        height: 600px;
        display: flex;
        flex-direction: column;
    }
    .chat-header {
        display: flex;
        border-bottom: 1px solid #14233A;
        background-color: #091325;
        border-radius: 6px 6px 0 0;
    }
    .chat-tab {
        flex: 1;
        text-align: center;
        padding: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        cursor: pointer;
    }
    .chat-tab.active {
        color: #FFF;
        border-bottom: 2px solid #FF671F;
        background-color: #0B172D;
    }
    .chat-body {
        flex: 1;
        padding: 16px;
        overflow-y: auto;
    }
    
    /* Chat Bubbles */
    .chat-msg {
        display: flex;
        gap: 12px;
        margin-bottom: 20px;
    }
    .avatar {
        width: 28px;
        height: 28px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.7rem;
        font-weight: bold;
        flex-shrink: 0;
    }
    .avatar.sq { background-color: #FF671F; color: white; }
    .avatar.user { background-color: #1E3150; color: #94A3B8; }
    
    .msg-content {
        background-color: #0B172D;
        border: 1px solid #14233A;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.85rem;
        line-height: 1.5;
        color: #E2E8F0;
        width: 100%;
    }
    .msg-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.7rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
    }
    
    .highlight-pill {
        color: #00B4D8;
        cursor: pointer;
        font-weight: 500;
        transition: opacity 0.2s;
    }
    .highlight-pill:hover { opacity: 0.8; text-decoration: underline; }
    
    /* Chat Input */
    .chat-input-wrapper {
        padding: 16px;
        border-top: 1px solid #14233A;
        background-color: #091325;
    }
    .suggestion-chips {
        display: flex;
        gap: 8px;
        margin-bottom: 12px;
        flex-wrap: wrap;
    }
    .chip {
        background: transparent;
        border: 1px solid #1E3150;
        color: #94A3B8;
        font-size: 0.7rem;
        padding: 4px 10px;
        border-radius: 12px;
        cursor: pointer;
    }
    .chip:hover { border-color: #00B4D8; color: #00B4D8; }
</style>
""")


# ─── LAYOUT: Sidebar + Main + Chat ─────────────────────────────────────────
# Hack to hide native sidebar gap if needed, but we use native for ease
with st.sidebar:
    st.html("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom: 30px;">
        <h2 style="margin:0; color:#FF671F;">🛰️ SATQUERY AI</h2>
    </div>
    """)
    
    st.html('<div class="section-title">WORKSPACE</div>')
    st.html('<div class="menu-item active">⬡ Command Center</div>')
    st.html('<div class="menu-item">∿ Change Detection</div>')
    st.html('<div class="menu-item">◩ Optical + SAR Fusion</div>')
    st.html('<div class="menu-item">⚲ VQA Explorer</div>')
    
    st.html('<div class="section-title">DATA INGESTION</div>')
    st.html("""
    <div style="font-size:0.8rem; margin-bottom:8px; display:flex; justify-content:space-between;">
        <span><span style="color:#10B981;">●</span> Sentinel-2</span> <span style="color:#10B981; font-weight:600;">READY</span>
    </div>
    <div style="font-size:0.8rem; margin-bottom:16px; display:flex; justify-content:space-between;">
        <span><span style="color:#00B4D8;">●</span> Sentinel-1 SAR</span> <span style="color:#10B981; font-weight:600;">READY</span>
    </div>
    """)
    st.button("+ Upload GeoTIFF", use_container_width=True)
    
    st.html('<div class="section-title" style="margin-top:40px;">MODEL SETTINGS</div>')
    st.caption("Inference model")
    st.selectbox("Model", ["SatQuery-VQA v1.4", "Optical-SAR Base"], label_visibility="collapsed")
    
    st.caption("Confidence threshold (0.70)")
    st.slider("Threshold", 0.0, 1.0, 0.70, label_visibility="collapsed")
    
    st.html("""
    <div style="position: absolute; bottom: 20px; font-size: 0.7rem; color: #64748B;">
        <span style="color:#10B981;">●</span> SYSTEM ONLINE<br>
        v0.9.7 • LOCAL INFERENCE
    </div>
    """)


# Main Content Columns
col_main, col_chat = st.columns([2.8, 1.2], gap="large")

with col_main:
    # ─── HEADER ───
    st.html("""
    <div style="display:flex; justify-content:space-between; align-items:flex-end; margin-bottom:16px;">
        <div>
            <div style="color:#94A3B8; font-size:0.75rem; letter-spacing:1px; text-transform:uppercase; margin-bottom:4px;">
                GEOSPATIAL WORKSPACE / REGION 28.6139° N, 77.2090° E
            </div>
            <h2 style="margin:0; font-size:1.8rem; font-weight:600;">Delhi Urban Change Analysis</h2>
        </div>
        <div style="display:flex; gap:12px; align-items:center;">
            <div style="border: 1px solid #1E3150; padding: 6px 16px; border-radius:4px; font-size:0.85rem; font-weight:500;">
                ⚲ T1 / T2
            </div>
            <div class="primary-btn">
                <button style="padding: 6px 20px !important;">▶ RUN ANALYSIS</button>
            </div>
        </div>
    </div>
    """)
    
    # ─── MAP WORKSPACE ───
    st.html("""
    <div class="map-container">
        <div class="map-overlay-grid"></div>
        
        <!-- Synthetic Geo Overlay representing change -->
        <div style="position:absolute; top:30%; left:20%; width:200px; height:300px; background:rgba(16, 185, 129, 0.2); border:1px solid rgba(16, 185, 129, 0.5);"></div>
        <div style="position:absolute; top:20%; right:30%; width:250px; height:200px; background:rgba(239, 68, 68, 0.2); border:1px solid rgba(239, 68, 68, 0.5);"></div>
        
        <!-- Legend Top Left -->
        <div style="position:absolute; top:20px; left:20px; display:flex; gap:10px;">
            <div style="background:#070F1E; border:1px solid #14233A; padding:6px 12px; border-radius:4px; font-size:0.75rem; color:#94A3B8;">
                T1 • 2025-01-14
            </div>
            <div style="background:#070F1E; border:1px solid #14233A; padding:6px 12px; border-radius:4px; font-size:0.75rem; color:#94A3B8;">
                T2 • 2026-07-18
            </div>
        </div>
        
        <!-- Split Slider -->
        <div class="split-line"></div>
        <div class="split-handle">T1 ‹ › T2</div>
        
        <!-- Layer Stack Panel -->
        <div class="glass-panel panel-layer-stack">
            <div style="font-size:0.75rem; font-weight:600; color:#F8FAFC; margin-bottom:12px; letter-spacing:1px;">LAYER STACK <span>^</span></div>
            
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:0.85rem;">
                <input type="checkbox" checked style="accent-color:#FF671F;"> <span style="color:#10B981;">■</span> Optical / Sentinel-2
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px; font-size:0.85rem;">
                <input type="checkbox" checked style="accent-color:#FF671F;"> <span style="color:#00B4D8;">■</span> SAR / Sentinel-1
            </div>
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:16px; font-size:0.85rem;">
                <input type="checkbox" checked style="accent-color:#FF671F;"> <span style="color:#FF671F;">■</span> Cross-Attention
            </div>
            
            <div style="font-size:0.7rem; color:#94A3B8; display:flex; justify-content:space-between; margin-bottom:4px;">
                <span>RADAR OPACITY</span><span>55%</span>
            </div>
            <input type="range" min="1" max="100" value="55" style="width:100%; accent-color:#00B4D8;">
        </div>
        
        <!-- Toolbar -->
        <div class="map-toolbar">
            <div>⌖</div>
            <div>+</div>
            <div>-</div>
            <div>↺</div>
            <div>❖</div>
        </div>
        
        <!-- Change Region Bottom Left -->
        <div class="glass-panel panel-change-region" style="width: 140px;">
            <div style="font-size:0.65rem; color:#94A3B8; text-transform:uppercase; margin-bottom:4px;">CHANGE REGION</div>
            <div style="font-size:1.5rem; font-weight:700; color:#FFF; margin-bottom:4px;">94.0%</div>
            <div class="text-green" style="font-size:0.75rem;">▲ 3.8%</div>
        </div>
        
        <!-- Classification Bottom Right -->
        <div class="glass-panel panel-classification" style="width: 300px; display:flex; justify-content:space-between; align-items:center;">
            <div>
                <div style="font-size:0.65rem; color:#94A3B8; text-transform:uppercase; margin-bottom:4px;">BIGEARTHNET-19</div>
                <div style="font-size:1.1rem; font-weight:700; color:#FFF; margin-bottom:2px;">INDUSTRIAL ZONE</div>
                <div style="font-size:0.65rem; color:#64748B;">Growth / Change / Damage / Loss Focus</div>
            </div>
            <div class="text-green" style="font-size:1.5rem; font-weight:700;">0.91</div>
        </div>
    </div>
    """)
    
    # ─── STATS ROW ───
    st.html("""
    <div style="display:grid; grid-template-columns: repeat(5, 1fr); gap: 16px;">
        <div class="stat-card">
            <div class="stat-label">TOTAL AREA CHANGED</div>
            <div class="stat-value">14.2%</div>
            <div class="stat-sub text-green">▲ 3.8%</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">MODEL CONFIDENCE</div>
            <div class="stat-value">94.0%</div>
            <div class="stat-sub text-green">HIGH</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">PIXELS ANALYZED</div>
            <div class="stat-value">2.84M</div>
            <div class="stat-sub" style="color:#94A3B8;">10 m RES.</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">CHANGE DETECTION</div>
            <div class="stat-value" style="font-size:1rem; margin-top:10px;">COMPLETE</div>
            <div class="stat-sub text-green" style="margin-top:8px;">● 42ms</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">RADAR QUALITY</div>
            <div class="stat-value">55%</div>
            <div class="stat-sub text-green">GOOD</div>
        </div>
    </div>
    """)


with col_chat:
    # ─── RIGHT CHAT PANE ───
    st.html("""
    <div class="chat-container">
        <!-- Tabs -->
        <div class="chat-header">
            <div class="chat-tab active">VQA CHAT</div>
            <div class="chat-tab">ARCHITECT LOG <span style="background:#FF671F; color:#FFF; padding:2px 6px; border-radius:4px; font-size:0.6rem; margin-left:4px;">7</span></div>
        </div>
        
        <!-- Chat Body -->
        <div class="chat-body">
            <!-- Context Header -->
            <div style="display:flex; gap:12px; align-items:center; margin-bottom:24px;">
                <div style="width:16px; height:16px; border-radius:50%; border:2px solid #00B4D8; box-shadow:0 0 8px #00B4D8;"></div>
                <div>
                    <div style="color:#F8FAFC; font-size:0.85rem; font-weight:500;">Analysis context locked</div>
                    <div style="color:#64748B; font-size:0.75rem;">Delhi • T1 → T2 • Multimodal</div>
                </div>
            </div>
            
            <!-- User Msg -->
            <div class="chat-msg">
                <div class="msg-content" style="background:transparent; border:none; padding:0; border-bottom:1px solid #14233A; padding-bottom:16px; border-radius:0;">
                    What significant changes are visible in the northern industrial area?
                </div>
            </div>
            
            <!-- AI Msg 1 -->
            <div class="chat-msg">
                <div class="avatar sq">SQ</div>
                <div class="msg-content">
                    <div class="msg-header">
                        <span>SATQUERY AI</span>
                        <span class="text-green">94% CONFIDENCE</span>
                    </div>
                    <div style="margin-bottom:12px;">
                        The imagery indicates <span class="highlight-pill">flooding</span> and newly developed built-up structures in the northern sector.
                    </div>
                    <div style="margin-bottom:12px;">
                        The strongest change cluster covers approximately <span class="highlight-pill">18.6 ha</span>.
                    </div>
                    <div style="display:flex; gap:8px;">
                        <span style="border:1px solid #1E3150; border-radius:4px; padding:2px 6px; font-size:0.65rem; color:#94A3B8;">□ T1 / T2</span>
                        <span style="border:1px solid #1E3150; border-radius:4px; padding:2px 6px; font-size:0.65rem; color:#94A3B8;">~ SAR</span>
                        <span style="border:1px solid #1E3150; border-radius:4px; padding:2px 6px; font-size:0.65rem; color:#94A3B8;">⚲ VQA</span>
                    </div>
                </div>
            </div>
            
            <!-- User Msg 2 -->
            <div class="chat-msg" style="flex-direction:row-reverse;">
                <div class="avatar user">AN</div>
                <div class="msg-content" style="background:#091325;">
                    Is the change likely permanent?
                </div>
            </div>
            
            <!-- AI Msg 2 -->
            <div class="chat-msg">
                <div class="avatar sq">SQ</div>
                <div class="msg-content">
                    <div class="msg-header">
                        <span>SATQUERY AI</span>
                        <span class="text-green">91% CONFIDENCE</span>
                    </div>
                    <div>
                        Yes. SAR backscatter and optical texture jointly suggest <span class="highlight-pill">persistent construction</span>, rather than a temporary surface anomaly.
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Input Area -->
        <div class="chat-input-wrapper">
            <div class="suggestion-chips">
                <button class="chip">Compare vegetation</button>
                <button class="chip">Show SAR evidence</button>
                <button class="chip">Quantify built-up change</button>
            </div>
            <div style="display:flex; gap:8px;">
                <input type="text" placeholder="Ask about this imagery..." style="flex:1; background:#050A14; border:1px solid #1E3150; color:#FFF; border-radius:4px; padding:10px 12px; outline:none; font-family:'Inter',sans-serif; font-size:0.85rem;">
                <button style="background:#FF671F; border:none; border-radius:4px; width:40px; color:#FFF; cursor:pointer;">↑</button>
            </div>
            <div style="text-align:center; color:#475569; font-size:0.6rem; margin-top:8px;">
                AI-generated analysis • Verify against source imagery before operational use.
            </div>
        </div>
    </div>
    """)
