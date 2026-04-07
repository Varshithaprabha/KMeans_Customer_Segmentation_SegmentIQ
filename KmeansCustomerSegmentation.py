"""
K-MEANS CUSTOMER SEGMENTATION DASHBOARD
Retail Analytics Suite

Dataset : Mall Customer Segmentation (Kaggle)
          https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python

Install :  pip install pandas numpy scikit-learn

Usage   :  python kmeans_customer_segmentation.py
           python kmeans_customer_segmentation.py --data Mall_Customers.csv
           python kmeans_customer_segmentation.py --data Mall_Customers.csv --k 5 --port 9000
"""

import argparse, json, os, warnings, webbrowser, http.server, threading, datetime
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

BUILD_DT = datetime.datetime.now().strftime("%d %b %Y, %H:%M")

COLORS = ["#C0533A","#3D7A5F","#C4963A","#7B5EA7",
          "#4A8FA8","#B85C8A","#5F8A3D","#A05C3A"]

SEG_NAMES = [
    "High Rollers", "Affluent Savers", "Impulsive Buyers", "Budget Shoppers",
    "Mainstream Crowd", "Young Trendsetters", "Mature Elites", "Deal Hunters",
]
SEG_RECS = [
    "🏆 VIP priority — exclusive early access, premium loyalty rewards, white-glove service.",
    "💡 Untapped potential — aspirational upsell campaigns, prestige product showcases.",
    "⚡ Impulse-driven — limited-time offers, BNPL options, scarcity messaging.",
    "💰 Price-sensitive — value bundles, cashback programs, referral incentives.",
    "👥 Core backbone — personalised recommendations, loyalty points, retention campaigns.",
    "✨ Trend-driven — social proof, influencer content, fast-moving fashion drops.",
    "🎯 Quality-focused — premium memberships, concierge service, heritage brands.",
    "🔖 Deal-seekers — flash sales, price-match guarantees, comparison widgets.",
]
SEG_ICONS = ["💎","🏦","🛍️","🪙","👥","✨","🎯","🔖"]


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_or_generate(path=None):
    if path and os.path.exists(path):
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        rename = {}
        for c in df.columns:
            cl = c.lower().replace(" ", "_")
            if   "income"  in cl: rename[c] = "Annual_Income"
            elif "spend"   in cl: rename[c] = "Spending_Score"
            elif "age"     in cl: rename[c] = "Age"
            elif "gender"  in cl: rename[c] = "Gender"
        df.rename(columns=rename, inplace=True)
        print(f"  ✓  Loaded {len(df)} customers from '{path}'")
        return df
    print("  ℹ  No CSV — generating synthetic mall dataset (n=200).")
    np.random.seed(42)
    groups = [(85,80,28,40),(25,75,32,35),(85,15,45,40),(25,20,50,30),(55,50,35,55)]
    rows = []
    for inc_mu, sp_mu, age_mu, n in groups:
        inc  = np.clip(np.random.normal(inc_mu, 10, n), 15, 137)
        sp   = np.clip(np.random.normal(sp_mu,  12, n), 1, 100)
        age  = np.clip(np.random.normal(age_mu,  8, n), 18, 70)
        gend = np.random.choice(["Male","Female"], n)
        for i in range(n):
            rows.append([gend[i], int(age[i]), round(float(inc[i]),1), int(sp[i])])
    return pd.DataFrame(rows, columns=["Gender","Age","Annual_Income","Spending_Score"])


# ─────────────────────────────────────────────────────────────────────────────
# MATH
# ─────────────────────────────────────────────────────────────────────────────
def elbow_data(X_sc, k_max=10):
    ks, wcss, sil = [], [], []
    for k in range(2, k_max+1):
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        lb = km.fit_predict(X_sc)
        ks.append(k)
        wcss.append(round(float(km.inertia_), 2))
        sil.append(round(float(silhouette_score(X_sc, lb)), 4))
    return ks, wcss, sil

def auto_k(wcss):
    d2 = np.diff(wcss, 2)
    return int(np.argmax(d2) + 3)

def run_kmeans(X_sc, k):
    km = KMeans(n_clusters=k, init="k-means++", n_init=15, max_iter=300, random_state=42)
    return km, km.fit_predict(X_sc)

def make_profiles(df, labels, inc_med, sp_med):
    d = df.copy(); d["Cluster"] = labels
    result = []
    for c, g in d.groupby("Cluster"):
        inc = float(g["Annual_Income"].mean())
        sp  = float(g["Spending_Score"].mean())
        age = float(g["Age"].mean())
        hi = inc >= inc_med; hs = sp >= sp_med
        ni = 0 if hi and hs else 1 if hi else 2 if hs else 3
        result.append({
            "name":  SEG_NAMES[ni], "rec": SEG_RECS[ni],
            "icon":  SEG_ICONS[ni], "size": int(len(g)),
            "pct":   round(len(g)/len(df)*100, 1),
            "income":round(inc,1), "spend":round(sp,1), "age":round(age,1),
        })
    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD  —  forest green / cream / terracotta theme
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SegmentIQ — Customer Analytics</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,400;0,600;0,700;1,400&family=DM+Sans:wght@400;500;600&family=DM+Mono&display=swap" rel="stylesheet">
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:       #F5F0E8;
  --bg2:      #FDFAF4;
  --bg3:      #EDE7D9;
  --border:   #D9CEBE;
  --border2:  #C8BAA6;
  --text:     #1F1A14;
  --text2:    #6B5E4E;
  --text3:    #9C8C7A;
  --accent:   #3D7A5F;
  --accent2:  #2D6048;
  --terra:    #C0533A;
  --terra2:   #A03E27;
  --gold:     #C4963A;
  --success:  #3D7A5F;
  --warn:     #C4963A;
  --danger:   #C0533A;
  --radius:   10px;
  --radius-lg:16px;
}
html { scroll-behavior: smooth; }
body {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg);
  color: var(--text);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* ── HEADER ──────────────────────────────────────────── */
.site-header {
  background: var(--accent2);
  padding: 0 36px;
  height: 62px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  position: sticky;
  top: 0;
  z-index: 100;
}
.logo { display: flex; align-items: center; gap: 11px; }
.logo-icon {
  width: 34px; height: 34px;
  background: rgba(255,255,255,0.18);
  border: 1px solid rgba(255,255,255,0.25);
  border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  font-size: 17px;
}
.logo-text {
  font-family: 'Fraunces', serif;
  font-size: 18px;
  font-weight: 700;
  color: #fff;
  letter-spacing: -0.01em;
}
.logo-text span { color: #FBBF60; }
.header-center {
  display: flex; gap: 2px;
  background: rgba(0,0,0,0.18);
  border-radius: 10px;
  padding: 4px;
}
.htab {
  padding: 7px 20px;
  font-size: 13px;
  font-weight: 500;
  color: rgba(255,255,255,0.65);
  cursor: pointer;
  border-radius: 7px;
  transition: all .15s;
  white-space: nowrap;
}
.htab:hover { color: #fff; background: rgba(255,255,255,0.12); }
.htab.on { background: rgba(255,255,255,0.22); color: #fff; font-weight: 600; }
.header-right { display: flex; align-items: center; gap: 10px; }
.hbadge {
  background: rgba(255,255,255,0.15);
  border: 1px solid rgba(255,255,255,0.2);
  border-radius: 8px;
  padding: 5px 13px;
  font-size: 12px;
  color: rgba(255,255,255,0.85);
  display: flex; align-items: center; gap: 7px;
}
.dot-live {
  width: 7px; height: 7px;
  background: #FBBF60;
  border-radius: 50%;
  animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.35} }

/* ── MAIN ────────────────────────────────────────────── */
main {
  flex: 1;
  max-width: 1320px;
  width: 100%;
  margin: 0 auto;
  padding: 28px 24px;
}
.panel { display: none; }
.panel.on { display: block; animation: fadeUp .22s ease; }
@keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:none} }

/* ── GRID ────────────────────────────────────────────── */
.g  { display: grid; gap: 14px; margin-bottom: 14px; }
.g2 { grid-template-columns: 1fr 1fr; }
.g3 { grid-template-columns: 1fr 1fr 1fr; }
.g4 { grid-template-columns: repeat(4,1fr); }
@media(max-width:900px){ .g2,.g3,.g4 { grid-template-columns: 1fr; } }

/* ── CARDS ───────────────────────────────────────────── */
.card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 22px;
  transition: border-color .2s, box-shadow .2s;
}
.card:hover { border-color: var(--border2); box-shadow: 0 2px 12px rgba(61,122,95,0.08); }
.card-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text3);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-bottom: 14px;
}
.metric { font-size: 34px; font-weight: 700; letter-spacing: -0.03em; line-height: 1; color: var(--text); font-family: 'Fraunces', serif; }
.metric-sub { font-size: 12px; color: var(--text3); margin-top: 5px; }
.metric-card { position: relative; overflow: hidden; }
.metric-card::after { content: attr(data-icon); position: absolute; right: 18px; top: 14px; font-size: 28px; opacity: .1; }
.chart-wrap { position: relative; width: 100%; }

/* ── HERO ────────────────────────────────────────────── */
.hero {
  background: linear-gradient(135deg, var(--accent2) 0%, #3D7A5F 100%);
  border-radius: var(--radius-lg);
  padding: 30px 34px;
  margin-bottom: 18px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  color: #fff;
}
.hero-text h2 {
  font-family: 'Fraunces', serif;
  font-size: 24px; font-weight: 700;
  letter-spacing: -0.02em; margin-bottom: 7px;
}
.hero-text p { font-size: 14px; color: rgba(255,255,255,0.75); line-height: 1.65; max-width: 520px; }
.hero-stat { display: flex; gap: 32px; flex-shrink: 0; }
.hs { text-align: center; }
.hs-val { font-family: 'Fraunces', serif; font-size: 26px; font-weight: 700; color: #FBBF60; }
.hs-lbl { font-size: 11px; color: rgba(255,255,255,0.6); margin-top: 3px; text-transform: uppercase; letter-spacing: .06em; }

/* ── CONTROLS ────────────────────────────────────────── */
.ctrl-row { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
.ctrl-row label { font-size: 12px; color: var(--text2); font-weight: 600; min-width: 105px; }
.ctrl-row input[type=range] { flex: 1; accent-color: var(--accent); }
.ctrl-val { font-size: 13px; font-weight: 700; color: var(--text); min-width: 28px; text-align: right; font-family: 'DM Mono', monospace; }

input[type=number], select {
  width: 100%;
  padding: 9px 12px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 13px;
  color: var(--text);
  font-family: 'DM Sans', sans-serif;
  outline: none;
  transition: border-color .15s;
}
input[type=number]:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(61,122,95,0.15);
}

.btn {
  display: block; width: 100%;
  padding: 11px 20px;
  border: none; border-radius: 9px;
  font-size: 13px; font-weight: 600;
  cursor: pointer;
  transition: all .15s;
  font-family: 'DM Sans', sans-serif;
  margin-bottom: 8px;
}
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { background: var(--accent2); transform: translateY(-1px); box-shadow: 0 4px 14px rgba(61,122,95,0.35); }
.btn-secondary { background: var(--bg3); color: var(--text); border: 1px solid var(--border2); }
.btn-secondary:hover { background: var(--border); }
.btn:active { transform: scale(.98); }

/* ── LEGEND ──────────────────────────────────────────── */
.legend { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
.legend-item { display: flex; align-items: center; gap: 7px; font-size: 12px; color: var(--text2); }
.ldot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

/* ── TABLE ───────────────────────────────────────────── */
.table-wrap { overflow-x: auto; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 9px 12px; font-size: 11px; font-weight: 700; color: var(--text3); text-transform: uppercase; letter-spacing: .06em; border-bottom: 1px solid var(--border); }
td { padding: 10px 12px; border-bottom: 1px solid var(--border); color: var(--text); }
tr:last-child td { border: none; }
tr:hover td { background: var(--bg3); }

/* ── FORM ────────────────────────────────────────────── */
.form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.full { grid-column: 1/-1; }
.field label { display: block; font-size: 11px; font-weight: 600; color: var(--text3); margin-bottom: 5px; text-transform: uppercase; letter-spacing: .06em; }

/* ── PREDICT RESULT ──────────────────────────────────── */
.pred-cluster-badge { border-radius: 12px; padding: 18px 20px; margin-bottom: 16px; border: 1px solid; }
.stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: 13px; }
.stat-row:last-child { border: none; }
.stat-label { color: var(--text2); }
.stat-value { font-weight: 600; color: var(--text); }

/* ── REC CARDS ───────────────────────────────────────── */
.rec-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-left: 3px solid;
  border-radius: 0 10px 10px 0;
  padding: 14px 18px;
  margin-bottom: 10px;
}
.rec-header { display: flex; align-items: center; gap: 10px; margin-bottom: 7px; }
.rec-icon { font-size: 20px; }
.rec-title { font-size: 14px; font-weight: 700; color: var(--text); }
.rec-meta { font-size: 12px; color: var(--text3); margin-top: 3px; }
.seg-tag { padding: 2px 10px; border-radius: 99px; font-size: 11px; font-weight: 700; margin-left: auto; }

/* ── ALGO STEPS ──────────────────────────────────────── */
.algo-steps { display: flex; flex-direction: column; gap: 10px; }
.algo-step { display: flex; gap: 12px; align-items: flex-start; }
.step-num {
  width: 24px; height: 24px; border-radius: 50%;
  background: var(--accent); color: #fff;
  font-size: 11px; font-weight: 700;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0; margin-top: 1px;
}
.algo-step p { font-size: 13px; color: var(--text2); line-height: 1.55; }
.divider { border: none; border-top: 1px solid var(--border); margin: 16px 0; }

/* ── FOOTER ──────────────────────────────────────────── */
.site-footer {
  background: var(--accent2);
  color: rgba(255,255,255,0.85);
  padding: 0 36px;
  margin-top: auto;
}
.footer-top {
  display: grid;
  grid-template-columns: 2fr 1fr 1fr 1fr;
  gap: 40px;
  padding: 44px 0 36px;
}
@media(max-width:900px){ .footer-top { grid-template-columns: 1fr; gap: 24px; } }
.footer-brand p { font-size: 13px; color: rgba(255,255,255,0.6); line-height: 1.65; margin-top: 10px; max-width: 290px; }
.footer-col h4 { font-size: 11px; font-weight: 700; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: .09em; margin-bottom: 14px; }
.footer-col ul { list-style: none; display: flex; flex-direction: column; gap: 8px; }
.footer-col li { font-size: 13px; color: rgba(255,255,255,0.65); }
.footer-col li span { font-family: 'DM Mono', monospace; font-size: 12px; color: rgba(255,255,255,0.4); }
.footer-pills { display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px; }
.fpill { background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.15); border-radius: 6px; padding: 3px 10px; font-size: 11px; color: rgba(255,255,255,0.6); font-family: 'DM Mono', monospace; }
.footer-bottom {
  border-top: 1px solid rgba(255,255,255,0.12);
  padding: 16px 0;
  display: flex; align-items: center; justify-content: space-between;
  font-size: 12px; color: rgba(255,255,255,0.4);
}
.footer-bottom a { color: #FBBF60; text-decoration: none; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 3px; }
</style>
</head>
<body>

<!-- ═══ HEADER ═══ -->
<header class="site-header">
  <div class="logo">
    <div class="logo-icon">📊</div>
    <div class="logo-text">Segment<span>IQ</span></div>
  </div>

  <div class="header-center">
    <div class="htab on"  onclick="go('ov')">Overview</div>
    <div class="htab"     onclick="go('cl')">Clusters</div>
    <div class="htab"     onclick="go('pr')">Predict</div>
    <div class="htab"     onclick="go('ins')">Insights</div>
  </div>

  <div class="header-right">
    <div class="hbadge" id="dbadge">
      <span class="dot-live"></span>
      <span>Loading…</span>
    </div>
  </div>
</header>

<!-- ═══ MAIN ═══ -->
<main>

<!-- OVERVIEW -->
<div id="p-ov" class="panel on">
  <div class="hero">
    <div class="hero-text">
      <h2>Customer Segmentation Analysis</h2>
      <p>K-Means clustering groups retail customers by purchase behaviour — annual income, spending score and age — to reveal actionable segments and targeted marketing strategies.</p>
    </div>
    <div class="hero-stat">
      <div class="hs"><div class="hs-val" id="hs-k">—</div><div class="hs-lbl">Clusters</div></div>
      <div class="hs"><div class="hs-val" id="hs-sil">—</div><div class="hs-lbl">Silhouette</div></div>
      <div class="hs"><div class="hs-val" id="hs-n">—</div><div class="hs-lbl">Customers</div></div>
    </div>
  </div>

  <div class="g g4">
    <div class="card metric-card" data-icon="👥"><div class="card-title">Total Customers</div><div class="metric" id="m1">—</div><div class="metric-sub">in dataset</div></div>
    <div class="card metric-card" data-icon="💵"><div class="card-title">Avg Annual Income</div><div class="metric" id="m2">—</div><div class="metric-sub">k USD</div></div>
    <div class="card metric-card" data-icon="🛒"><div class="card-title">Avg Spending Score</div><div class="metric" id="m3">—</div><div class="metric-sub">out of 100</div></div>
    <div class="card metric-card" data-icon="📅"><div class="card-title">Avg Age</div><div class="metric" id="m4">—</div><div class="metric-sub">years old</div></div>
  </div>

  <div class="g g2">
    <div class="card"><div class="card-title">Income Distribution</div><div class="chart-wrap" style="height:220px"><canvas id="cInc"></canvas></div></div>
    <div class="card"><div class="card-title">Spending Score Distribution</div><div class="chart-wrap" style="height:220px"><canvas id="cSp"></canvas></div></div>
  </div>
  <div class="card"><div class="card-title">Raw Data — Income vs Spending Score</div><div class="chart-wrap" style="height:300px"><canvas id="cRaw"></canvas></div></div>
</div>

<!-- CLUSTERS -->
<div id="p-cl" class="panel">
  <div class="g g2">
    <div class="card">
      <div class="card-title">Algorithm Settings</div>
      <div class="ctrl-row"><label>Clusters (K)</label><input type="range" id="kS" min="2" max="8" value="__K__" oninput="kV.textContent=this.value"><span class="ctrl-val" id="kV">__K__</span></div>
      <div class="ctrl-row" style="margin-bottom:18px"><label>Max Iterations</label><input type="range" id="iS" min="50" max="500" value="300" oninput="iV.textContent=this.value"><span class="ctrl-val" id="iV">300</span></div>
      <button class="btn btn-primary" onclick="recluster()">⚙ Re-run K-Means</button>
      <button class="btn btn-secondary" onclick="go('pr')">Predict a Customer →</button>
      <hr class="divider">
      <div class="card-title">How K-Means Works</div>
      <div class="algo-steps">
        <div class="algo-step"><div class="step-num">1</div><p>Initialize K centroids using <strong>k-means++</strong> for smart placement</p></div>
        <div class="algo-step"><div class="step-num">2</div><p>Assign each customer to their nearest centroid (Euclidean distance)</p></div>
        <div class="algo-step"><div class="step-num">3</div><p>Recompute each centroid as the mean of its assigned points</p></div>
        <div class="algo-step"><div class="step-num">4</div><p>Repeat until centroids converge or max iterations reached</p></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">Elbow Curve — Optimal K</div>
      <div class="chart-wrap" style="height:220px"><canvas id="cElbow"></canvas></div>
      <hr class="divider">
      <div style="font-size:12px;color:var(--text3);line-height:1.65">
        The <strong style="color:var(--text)">elbow point</strong> is where WCSS drops sharply then flattens — the optimal K. Selected K highlighted in <span style="color:var(--terra)">terracotta</span>.
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-title">Cluster Map — Annual Income vs Spending Score</div>
    <div id="cleg" class="legend"></div>
    <div class="chart-wrap" style="height:360px"><canvas id="cMap"></canvas></div>
  </div>

  <div class="card" style="margin-top:14px">
    <div class="card-title">Cluster Profiles Summary</div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Cluster</th><th>Segment Name</th><th>Size</th><th>Share</th><th>Avg Income</th><th>Avg Spending</th><th>Avg Age</th></tr></thead>
        <tbody id="ptable"></tbody>
      </table>
    </div>
  </div>
</div>

<!-- PREDICT -->
<div id="p-pr" class="panel">
  <div class="g g2">
    <div class="card">
      <div class="card-title">Enter Customer Details</div>
      <div class="form-grid">
        <div class="field"><label>Age</label><input type="number" id="pAge" value="30" min="18" max="80"></div>
        <div class="field"><label>Gender</label><select id="pGen"><option>Female</option><option>Male</option></select></div>
        <div class="field"><label>Annual Income (k$)</label><input type="number" id="pInc" value="60" min="10" max="200"></div>
        <div class="field"><label>Spending Score (1–100)</label><input type="number" id="pSpend" value="50" min="1" max="100"></div>
        <div class="full" style="margin-top:8px"><button class="btn btn-primary" onclick="predict()">⚡ Classify Customer</button></div>
      </div>
    </div>
    <div class="card" id="pRes" style="display:none">
      <div class="card-title">Classification Result</div>
      <div id="pOut"></div>
    </div>
  </div>
  <div class="card" style="margin-top:0">
    <div class="card-title">Customer Position in Cluster Space</div>
    <div class="chart-wrap" style="height:330px"><canvas id="cPred"></canvas></div>
  </div>
</div>

<!-- INSIGHTS -->
<div id="p-ins" class="panel">
  <div class="g g3">
    <div class="card"><div class="card-title">Cluster Size Distribution</div><div class="chart-wrap" style="height:240px"><canvas id="cSz"></canvas></div></div>
    <div class="card"><div class="card-title">Spending Score by Cluster</div><div class="chart-wrap" style="height:240px"><canvas id="cSpend"></canvas></div></div>
    <div class="card"><div class="card-title">PCA 2D Projection</div><div class="chart-wrap" style="height:240px"><canvas id="cPca"></canvas></div></div>
  </div>
  <div class="card">
    <div class="card-title">Marketing Recommendations by Segment</div>
    <div id="recs"></div>
  </div>
</div>

</main>

<!-- ═══ FOOTER ═══ -->
<footer class="site-footer">
  <div class="footer-top">
    <div class="footer-brand">
      <div class="logo">
        <div class="logo-icon" style="width:30px;height:30px;font-size:15px">📊</div>
        <div class="logo-text" style="font-size:15px">Segment<span>IQ</span></div>
      </div>
      <p>Open-source retail analytics dashboard powered by K-Means clustering. Built for data scientists, retail strategists, and product teams.</p>
      <div class="footer-pills">
        <div class="fpill">Python 3.8+</div>
        <div class="fpill">scikit-learn</div>
        <div class="fpill">Chart.js</div>
        <div class="fpill">pandas</div>
      </div>
    </div>

    <div class="footer-col">
      <h4>Algorithm</h4>
      <ul>
        <li>K-Means Clustering</li>
        <li>k-means++ Init</li>
        <li>Elbow Method</li>
        <li>Silhouette Score</li>
        <li>PCA Projection</li>
      </ul>
    </div>

    <div class="footer-col">
      <h4>Features</h4>
      <ul>
        <li>Annual Income</li>
        <li>Spending Score</li>
        <li>Age</li>
        <li>Gender</li>
        <li>Real-time Predict</li>
      </ul>
    </div>

    <div class="footer-col">
      <h4>CLI Options</h4>
      <ul>
        <li><span>--data</span> CSV path</li>
        <li><span>--k</span> force clusters</li>
        <li><span>--port</span> server port</li>
        <li>Dataset: Kaggle Mall</li>
        <li>Built: __BUILD__</li>
      </ul>
    </div>
  </div>

  <div class="footer-bottom">
    <span>© 2025 SegmentIQ · K-Means Customer Segmentation</span>
    <span>Dataset: <a href="https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python" target="_blank">Kaggle Mall Customers</a> · Built with Python + scikit-learn</span>
  </div>
</footer>

<!-- ═══ SCRIPT ═══ -->
<script>
const CL  = __COLORS__;
const D   = __DATA__;
let   CLS = __CLUSTERS__;
const charts = {};

function go(id){
  document.querySelectorAll('.htab').forEach((t,i)=>t.classList.toggle('on',['ov','cl','pr','ins'][i]===id));
  document.querySelectorAll('.panel').forEach(p=>p.classList.toggle('on',p.id==='p-'+id));
}
function kill(id){if(charts[id]){charts[id].destroy();delete charts[id];}}
function mk(id,cfg){kill(id);charts[id]=new Chart(document.getElementById(id),cfg);}

function lightGrid(){return{color:'rgba(61,122,95,0.08)',borderColor:'rgba(61,122,95,0.08)'};}
function baseTicks(){return{color:'#9C8C7A',font:{size:11,family:"'DM Sans'"}}}
function baseTitle(t){return{display:true,text:t,color:'#6B5E4E',font:{size:11,family:"'DM Sans'"}}}
function bOpts(xl,yl){return{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle(xl)},y:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle(yl)}}}}
function avg(a,k){return a.reduce((s,r)=>s+r[k],0)/a.length;}
function hist(vals,bins=10){
  const mn=Math.min(...vals),mx=Math.max(...vals),bw=(mx-mn)/bins;
  const lb=Array.from({length:bins},(_,i)=>(mn+i*bw).toFixed(0));
  const ct=new Array(bins).fill(0);
  vals.forEach(v=>{let b=Math.floor((v-mn)/bw);if(b===bins)b--;ct[b]++;});
  return{labels:lb,data:ct};
}

function init(){
  const C=D.customers;
  document.getElementById('dbadge').innerHTML=`<span class="dot-live"></span><span>${C.length} customers loaded</span>`;
  document.getElementById('hs-k').textContent=D.bestK;
  document.getElementById('hs-sil').textContent=D.sil[D.bestK-2].toFixed(3);
  document.getElementById('hs-n').textContent=C.length;
  document.getElementById('m1').textContent=C.length;
  document.getElementById('m2').textContent='$'+avg(C,'Annual_Income').toFixed(1)+'k';
  document.getElementById('m3').textContent=avg(C,'Spending_Score').toFixed(1);
  document.getElementById('m4').textContent=avg(C,'Age').toFixed(1);

  const ih=hist(C.map(r=>r.Annual_Income));
  mk('cInc',{type:'bar',data:{labels:ih.labels,datasets:[{data:ih.data,backgroundColor:'rgba(61,122,95,0.25)',borderColor:'#3D7A5F',borderWidth:1.5,borderRadius:4}]},options:bOpts('Income (k$)','Count')});
  const sh=hist(C.map(r=>r.Spending_Score));
  mk('cSp',{type:'bar',data:{labels:sh.labels,datasets:[{data:sh.data,backgroundColor:'rgba(192,83,58,0.2)',borderColor:'#C0533A',borderWidth:1.5,borderRadius:4}]},options:bOpts('Spending Score','Count')});
  mk('cRaw',{type:'scatter',data:{datasets:[{data:C.map(r=>({x:r.Annual_Income,y:r.Spending_Score})),backgroundColor:'rgba(61,122,95,0.35)',pointRadius:4,pointHoverRadius:6}]},options:bOpts('Annual Income (k$)','Spending Score')});

  mk('cElbow',{type:'line',data:{labels:D.ks,datasets:[{data:D.wcss,borderColor:'#3D7A5F',backgroundColor:'rgba(61,122,95,0.08)',fill:true,tension:.4,pointRadius:5,
    pointBackgroundColor:D.ks.map(k=>k===D.bestK?'#C0533A':'#3D7A5F'),
    pointRadius:D.ks.map(k=>k===D.bestK?8:5)
  }]},options:{...bOpts('K','WCSS'),plugins:{legend:{display:false},tooltip:{callbacks:{title:([i])=>'K = '+i.label}}}}});

  drawClusters(); drawInsights(); updatePredChart();
}

function drawClusters(){
  const C=D.customers,L=CLS.labels,k=CLS.profile.length;
  const ds=CLS.profile.map((p,i)=>({label:'C'+i,data:C.filter((_,j)=>L[j]===i).map(r=>({x:r.Annual_Income,y:r.Spending_Score})),backgroundColor:CL[i]+'99',pointRadius:5,pointHoverRadius:7}));
  ds.push({label:'Centroids',data:CLS.centroids.map(c=>({x:c[0],y:c[1]})),backgroundColor:'#1F1A14',pointRadius:11,pointStyle:'star',pointHoverRadius:14});
  mk('cMap',{type:'scatter',data:{datasets:ds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle('Annual Income (k$)')},y:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle('Spending Score')}}}});

  document.getElementById('cleg').innerHTML=CLS.profile.map((p,i)=>
    `<div class="legend-item"><span class="ldot" style="background:${CL[i]}"></span>C${i} — ${p.icon} ${p.name}</div>`).join('');

  document.getElementById('ptable').innerHTML=CLS.profile.map((p,i)=>`
    <tr>
      <td><span class="ldot" style="background:${CL[i]};display:inline-block;margin-right:6px"></span><strong>C${i}</strong></td>
      <td>${p.icon} ${p.name}</td>
      <td>${p.size}</td>
      <td><span style="color:${CL[i]};font-weight:700">${p.pct}%</span></td>
      <td>$${p.income}k</td><td>${p.spend}</td><td>${p.age}</td>
    </tr>`).join('');
}

function drawInsights(){
  const k=CLS.profile.length;
  mk('cSz',{type:'bar',data:{labels:CLS.profile.map((_,i)=>'C'+i),datasets:[{data:CLS.profile.map(p=>p.size),backgroundColor:CL.slice(0,k),borderRadius:6,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false},ticks:baseTicks()},y:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle('Customers')}}}});
  mk('cSpend',{type:'bar',data:{labels:CLS.profile.map((_,i)=>'C'+i),datasets:[{data:CLS.profile.map(p=>p.spend),backgroundColor:CL.slice(0,k),borderRadius:6,borderSkipped:false}]},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{display:false},ticks:baseTicks()},y:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle('Spending Score'),min:0,max:100}}}});
  const pds=CLS.profile.map((_,i)=>({label:'C'+i,data:D.pca.filter((_,j)=>CLS.labels[j]===i).map(p=>({x:p[0],y:p[1]})),backgroundColor:CL[i]+'99',pointRadius:4}));
  mk('cPca',{type:'scatter',data:{datasets:pds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle(`PC1 (${D.pcaVar[0]}% var)`)},y:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle(`PC2 (${D.pcaVar[1]}% var)`)}}}});

  document.getElementById('recs').innerHTML=CLS.profile.map((p,i)=>`
    <div class="rec-card" style="border-left-color:${CL[i]}">
      <div class="rec-header">
        <span class="rec-icon">${p.icon}</span>
        <div><div class="rec-title">C${i} — ${p.name}</div><div class="rec-meta">$${p.income}k income · ${p.spend} spending · ${p.age} yrs avg</div></div>
        <span class="seg-tag" style="background:${CL[i]}22;color:${CL[i]}">${p.size} · ${p.pct}%</span>
      </div>
      <div style="font-size:13px;color:var(--text2);line-height:1.6">${p.rec}</div>
    </div>`).join('');
}

function updatePredChart(){
  const C=D.customers,L=CLS.labels;
  const ds=CLS.profile.map((_,i)=>({label:'C'+i,data:C.filter((_,j)=>L[j]===i).map(r=>({x:r.Annual_Income,y:r.Spending_Score})),backgroundColor:CL[i]+'77',pointRadius:4}));
  if(window._pp) ds.push({label:'New',data:[{x:window._pp.x,y:window._pp.y}],backgroundColor:'#1F1A14',pointRadius:13,pointStyle:'star',pointHoverRadius:16});
  mk('cPred',{type:'scatter',data:{datasets:ds},options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle('Annual Income (k$)')},y:{grid:lightGrid(),ticks:baseTicks(),title:baseTitle('Spending Score')}}}});
}

function d2(a,b){return a.reduce((s,v,i)=>s+(v-b[i])**2,0);}

function recluster(){
  const k=+document.getElementById('kS').value;
  const iters=+document.getElementById('iS').value;
  const C=D.customers;
  const pts=C.map(r=>[r.Annual_Income,r.Spending_Score,r.Age]);
  const mu=[0,1,2].map(d=>pts.reduce((s,p)=>s+p[d],0)/pts.length);
  const sd=[0,1,2].map(d=>Math.sqrt(pts.reduce((s,p)=>s+(p[d]-mu[d])**2,0)/pts.length)+1e-9);
  const sc=pts.map(p=>p.map((v,d)=>(v-mu[d])/sd[d]));
  let ce=[sc[Math.floor(Math.random()*sc.length)]];
  for(let i=1;i<k;i++){
    const ds2=sc.map(p=>Math.min(...ce.map(c=>d2(p,c))));
    const sum=ds2.reduce((a,b)=>a+b,0);let r=Math.random()*sum;
    for(let j=0;j<sc.length;j++){r-=ds2[j];if(r<=0){ce.push(sc[j]);break;}}
  }
  let lb=new Array(sc.length).fill(0);
  for(let it=0;it<iters;it++){
    lb=sc.map(p=>ce.map((c,i)=>({i,d:d2(p,c)})).sort((a,b)=>a.d-b.d)[0].i);
    const nce=Array.from({length:k},(_,c)=>{const g=sc.filter((_,i)=>lb[i]===c);return g.length?[0,1,2].map(d=>g.reduce((s,p)=>s+p[d],0)/g.length):ce[c]});
    if(nce.every((c,i)=>d2(c,ce[i])<1e-9))break;ce=nce;
  }
  const cents=ce.map(c=>[c[0]*sd[0]+mu[0],c[1]*sd[1]+mu[1]]);
  const inc_med=C.map(r=>r.Annual_Income).slice().sort((a,b)=>a-b)[Math.floor(C.length/2)];
  const sp_med=C.map(r=>r.Spending_Score).slice().sort((a,b)=>a-b)[Math.floor(C.length/2)];
  const NAMES=__NAMES__;const RECS=__RECS__;const ICONS=__ICONS__;
  CLS={labels:lb,centroids:cents,profile:Array.from({length:k},(_,c)=>{
    const g=C.filter((_,i)=>lb[i]===c);
    const inc=g.reduce((s,r)=>s+r.Annual_Income,0)/g.length;
    const sp=g.reduce((s,r)=>s+r.Spending_Score,0)/g.length;
    const age=g.reduce((s,r)=>s+r.Age,0)/g.length;
    const hi=inc>=inc_med,hs=sp>=sp_med,ni=hi&&hs?0:hi?1:hs?2:3;
    return{name:NAMES[ni],rec:RECS[ni],icon:ICONS[ni],size:g.length,pct:+(g.length/C.length*100).toFixed(1),income:+inc.toFixed(1),spend:+sp.toFixed(1),age:+age.toFixed(1)};
  })};
  drawClusters();drawInsights();updatePredChart();
}

function predict(){
  const inc=+document.getElementById('pInc').value;
  const sp=+document.getElementById('pSpend').value;
  const age=+document.getElementById('pAge').value;
  const C=D.customers;
  const mu=[C.reduce((s,r)=>s+r.Annual_Income,0)/C.length,C.reduce((s,r)=>s+r.Spending_Score,0)/C.length,C.reduce((s,r)=>s+r.Age,0)/C.length];
  const sd=[0,1,2].map(d=>Math.sqrt(C.reduce((s,r)=>s+([r.Annual_Income,r.Spending_Score,r.Age][d]-mu[d])**2,0)/C.length)+1e-9);
  const allPts=C.map(r=>[(r.Annual_Income-mu[0])/sd[0],(r.Spending_Score-mu[1])/sd[1],(r.Age-mu[2])/sd[2]]);
  const k=CLS.profile.length;
  const ce=Array.from({length:k},(_,c)=>{const g=allPts.filter((_,i)=>CLS.labels[i]===c);return g.length?[0,1,2].map(d=>g.reduce((s,p)=>s+p[d],0)/g.length):[0,0,0]});
  const inp=[(inc-mu[0])/sd[0],(sp-mu[1])/sd[1],(age-mu[2])/sd[2]];
  const c=ce.map((cv,i)=>({i,d:d2(inp,cv)})).sort((a,b)=>a.d-b.d)[0].i;
  const conf=Math.min(100,Math.max(0,100-d2(inp,ce[c])*12)).toFixed(1);
  const p=CLS.profile[c];
  document.getElementById('pRes').style.display='block';
  document.getElementById('pOut').innerHTML=`
    <div class="pred-cluster-badge" style="background:${CL[c]}14;border-color:${CL[c]}44">
      <div style="font-size:30px;margin-bottom:6px">${p.icon}</div>
      <div style="font-family:'Fraunces',serif;font-size:21px;font-weight:700;color:${CL[c]};margin-bottom:3px">C${c} — ${p.name}</div>
      <div style="font-size:12px;color:var(--text3)">Confidence: <strong style="color:var(--text)">${conf}%</strong></div>
    </div>
    <div class="stat-row"><span class="stat-label">Cluster avg income</span><span class="stat-value">$${p.income}k</span></div>
    <div class="stat-row"><span class="stat-label">Cluster avg spending</span><span class="stat-value">${p.spend}</span></div>
    <div class="stat-row"><span class="stat-label">Cluster avg age</span><span class="stat-value">${p.age} yrs</span></div>
    <div class="stat-row"><span class="stat-label">Cluster size</span><span class="stat-value">${p.size} customers</span></div>
    <div style="margin-top:14px;background:var(--bg3);border:1px solid var(--border);border-radius:9px;padding:12px 14px;font-size:13px;color:var(--text2);line-height:1.6">
      <strong style="color:var(--text)">Strategy:</strong> ${p.rec}
    </div>`;
  window._pp={x:inc,y:sp,c};
  updatePredChart();
}

init();
</script>
</body>
</html>"""


def build_html(df, ks, wcss, sil, bk, km, labels, scaler, features):
    C = df.copy()
    inc_med = float(C["Annual_Income"].median())
    sp_med  = float(C["Spending_Score"].median())
    prof    = make_profiles(C, labels, inc_med, sp_med)

    X_sc = scaler.transform(C[list(features)].values)
    pca  = PCA(n_components=2).fit(X_sc)
    proj = pca.transform(X_sc).tolist()
    pvar = [round(float(v)*100, 1) for v in pca.explained_variance_ratio_]
    cents = scaler.inverse_transform(km.cluster_centers_).tolist()

    html = HTML
    html = html.replace("__BUILD__",    BUILD_DT)
    html = html.replace("__K__",        str(bk), 2)
    html = html.replace("__COLORS__",   json.dumps(COLORS))
    html = html.replace("__DATA__",     json.dumps({
        "customers": C.to_dict(orient="records"),
        "ks": ks, "wcss": wcss, "sil": sil, "bestK": bk,
        "pca": proj, "pcaVar": pvar,
    }))
    html = html.replace("__CLUSTERS__", json.dumps({
        "labels":    labels.tolist(),
        "centroids": [c[:2] for c in cents],
        "profile":   prof,
    }))
    html = html.replace("__NAMES__", json.dumps(SEG_NAMES))
    html = html.replace("__RECS__",  json.dumps(SEG_RECS))
    html = html.replace("__ICONS__", json.dumps(SEG_ICONS))
    return html


# ─────────────────────────────────────────────────────────────────────────────
# SERVER
# ─────────────────────────────────────────────────────────────────────────────
def serve(html, port):
    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(html.encode())
        def log_message(self, *a): pass
    srv = http.server.HTTPServer(("localhost", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="K-Means Customer Segmentation Dashboard")
    parser.add_argument("--data", default=None,           help="Path to Mall_Customers.csv")
    parser.add_argument("--k",    type=int, default=None, help="Force number of clusters")
    parser.add_argument("--port", type=int, default=8765, help="Dashboard port (default: 8765)")
    args = parser.parse_args()

    print()
    print("┌─────────────────────────────────────────────┐")
    print("│   SegmentIQ — K-Means Customer Analytics    │")
    print(f"│   {BUILD_DT:<43}│")
    print("└─────────────────────────────────────────────┘")
    print()

    df = load_or_generate(args.data)

    features = ("Annual_Income", "Spending_Score")
    X  = df[list(features)].values.astype(float)
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)

    print("\n  ⟳  Computing elbow curve (K=2..10) …")
    ks, wcss, sil = elbow_data(X_sc)
    bk = args.k or auto_k(wcss)
    print(f"  ✓  Best K = {bk}  |  silhouette = {sil[bk-2]:.4f}")

    print(f"\n  ⟳  Running K-Means++ (K={bk}) …")
    km, labels = run_kmeans(X_sc, bk)
    final_sil  = silhouette_score(X_sc, labels)
    print(f"  ✓  Inertia = {km.inertia_:.1f}  |  silhouette = {final_sil:.4f}")

    print("\n  ⟳  Building dashboard …")
    html = build_html(df, ks, wcss, sil, bk, km, labels, sc, features)

    out = "customer_segmentation_dashboard.html"
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✓  Saved: {out}")

    srv = serve(html, args.port)
    url = f"http://localhost:{args.port}"
    print(f"\n  ✓  Dashboard → {url}")
    print("     Press Ctrl+C to stop.\n")
    webbrowser.open(url)

    try:
        while True: pass
    except KeyboardInterrupt:
        srv.shutdown()
        print("\n  ✓  Goodbye!\n")


if __name__ == "__main__":
    main()