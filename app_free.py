"""
Smart Money Radar — Ücretsiz Kripto Terminali
BTC & ETH için kurumsal para akışı analizi
Veriler: Binance API (ücretsiz, gerçek zamanlı)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests

# ────────────────────────────────────────────────────────────────────
# SAYFA AYARLARI
# ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Money Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={"About": "Smart Money Radar — Kurumsal Para Takip Terminali"}
)

# ────────────────────────────────────────────────────────────────────
# CSS
# ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #0d1117 !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stSidebar"]          { display: none !important; }
.block-container { padding: 1rem 2rem !important; max-width: 1200px !important; }
* { font-family: 'Inter', -apple-system, sans-serif; }

.stButton > button {
    background: #1e2530 !important; color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: all 0.2s !important;
}
.stButton > button:hover { border-color: #6366f1 !important; color: #6366f1 !important; }

.stTextInput > div > div > input {
    background: #1e2530 !important; color: #e2e8f0 !important;
    border: 1px solid #2d3748 !important; border-radius: 8px !important;
}
.stTextInput > label { color: #94a3b8 !important; font-size: 0.82rem !important; }

.smr-card {
    background: #161b22; border: 1px solid #21262d;
    border-radius: 12px; padding: 16px; margin-bottom: 16px;
}
.smr-header {
    font-size: 0.70rem; font-weight: 700; color: #6366f1;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
}
.premium-cta {
    background: linear-gradient(135deg, #1e1b4b, #1e2530);
    border: 1px solid #4338ca; border-radius: 8px;
    padding: 8px 12px; margin-top: 12px;
    text-align: center; font-size: 0.76rem; color: #a5b4fc;
}
hr { border-color: #21262d !important; margin: 0.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# VERİ KATMANI — BİNANCE API
# ────────────────────────────────────────────────────────────────────
def _binance_fetch(ticker: str, limit: int = 500):
    symbol = ticker.replace("-USD", "USDT").upper()
    url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
    try:
        resp = requests.get(url, timeout=8)
        data = resp.json()
        if isinstance(data, dict) and "code" in data:
            return None
        cols = ["ts","Open","High","Low","Close","Volume",
                "ct","qv","n","tbb","tbq","ig"]
        df = pd.DataFrame(data, columns=cols)
        for c in ["Open","High","Low","Close","Volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["Date"] = pd.to_datetime(df["ts"], unit="ms")
        df.set_index("Date", inplace=True)
        df.index = df.index.tz_localize(None)
        return df[["Open","High","Low","Close","Volume"]].dropna()
    except Exception:
        return None


@st.cache_data(ttl=300)
def get_data(ticker: str):
    return _binance_fetch(ticker)


# ────────────────────────────────────────────────────────────────────
# İNDİKATÖRLER
# ────────────────────────────────────────────────────────────────────
def calc_rsi(close, period=14):
    delta = close.diff()
    gain  = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss  = (-delta.where(delta < 0, 0.0)).rolling(period).mean().replace(0, 1e-9)
    return 100 - (100 / (1 + gain / loss))

def calc_macd(close):
    ema12  = close.ewm(span=12, adjust=False).mean()
    ema26  = close.ewm(span=26, adjust=False).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

def calc_obv(close, volume):
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()

def calc_force_index(close, volume, period=13):
    return (close.diff() * volume).ewm(span=period, adjust=False).mean()

def calc_bb_squeeze(df, bb_p=20, bb_s=2.0, kc_p=20, kc_m=1.5):
    c = df["Close"]; h = df["High"]; l = df["Low"]
    sma = c.rolling(bb_p).mean(); std = c.rolling(bb_p).std()
    atr = (h - l).rolling(kc_p).mean()
    sq  = ((sma + bb_s*std) < (sma + kc_m*atr)) & ((sma - bb_s*std) > (sma - kc_m*atr))
    return bool(sq.iloc[-1]) if len(sq.dropna()) > 0 else False


# ────────────────────────────────────────────────────────────────────
# ANALİZ FONKSİYONLARI
# ────────────────────────────────────────────────────────────────────
def analyze_long_radar(df):
    try:
        c = df["Close"]; v = df["Volume"]

        sma50  = c.rolling(50).mean()
        t_pass = bool(c.iloc[-1] > sma50.iloc[-1]) and bool(sma50.iloc[-1] > sma50.iloc[-6])

        obv    = calc_obv(c, v)
        a_pass = bool(obv.iloc[-1] > obv.rolling(20).mean().iloc[-1]) and bool(obv.iloc[-1] > obv.iloc[-10])

        sq_pass = calc_bb_squeeze(df)
        if not sq_pass:
            for back in range(2, 6):
                if len(df) > back + 22 and calc_bb_squeeze(df.iloc[:-back]):
                    sq_pass = True; break

        avg_v  = float(v.rolling(20).mean().iloc[-1])
        v_pass = bool(v.iloc[-1] > avg_v * 1.5) and bool(c.iloc[-1] > float(c.rolling(20).max().iloc[-2]))

        rsi    = calc_rsi(c)
        m_pass = bool(rsi.iloc[-1] > 50) and bool(rsi.iloc[-1] > rsi.iloc[-5])

        criteria = {
            "Trend Zemini":      (t_pass,  25),
            "Birikim (OBV)":     (a_pass,  20),
            "BB Sıkışması":      (sq_pass, 20),
            "Hacim Tetikleyici": (v_pass,  20),
            "Momentum (RSI)":    (m_pass,  15),
        }
        score     = sum(w for _, (p, w) in criteria.items() if p)
        passed    = sum(1 for _, (p, _) in criteria.items() if p)
        pre_launch = (passed >= 4 and not v_pass)

        if score >= 85:   status, sc = "🔥 Harekete Geç",      "#f59e0b"
        elif score >= 65: status, sc = "⚡ LONG İÇİN HAZIR",   "#10b981"
        elif pre_launch:  status, sc = "🎯 FİTİL ÇEKİLİYOR",  "#06b6d4"
        elif score >= 40: status, sc = "📊 İzleme Listesi",    "#94a3b8"
        else:             status, sc = "⛔ Henüz Değil",        "#ef4444"

        return {"score": score, "status": status, "status_col": sc, "criteria": criteria}
    except:
        return {"score": 0, "status": "Hesaplanamadı", "status_col": "#94a3b8", "criteria": {}}


def analyze_para_akisi(df):
    try:
        c = df["Close"]; v = df["Volume"]
        fi      = calc_force_index(c, v)
        fi_val  = float(fi.iloc[-1]); fi_prev = float(fi.iloc[-6])
        fi_max  = float(fi.abs().rolling(50).max().iloc[-1]) or 1
        fi_norm = fi_val / fi_max

        if fi_val > 0 and fi_val > fi_prev:   fi_s, fi_c = "Güçlenen Alım Baskısı",      "#10b981"
        elif fi_val > 0:                       fi_s, fi_c = "Alım Baskısı (zayıflıyor)",  "#34d399"
        elif fi_val < 0 and fi_val < fi_prev:  fi_s, fi_c = "Güçlenen Satış Baskısı",     "#ef4444"
        else:                                  fi_s, fi_c = "Satış Baskısı (azalıyor)",   "#f87171"

        h50 = float(df["High"].rolling(50).max().iloc[-1])
        l50 = float(df["Low"].rolling(50).min().iloc[-1])
        stp = (h50 + l50) / 2; cur = float(c.iloc[-1])
        sap = ((cur - stp) / stp) * 100

        if sap > 15:    stp_s, stp_c = "Dengenin çok üzerinde — Isınma var", "#ef4444"
        elif sap > 5:   stp_s, stp_c = "Dengenin üzerinde",                  "#f59e0b"
        elif sap > -5:  stp_s, stp_c = "Denge yakınında",                    "#94a3b8"
        elif sap > -15: stp_s, stp_c = "Dengenin altında",                   "#34d399"
        else:           stp_s, stp_c = "Derin iskonto bölgesi",              "#06b6d4"

        return {"fi_val": fi_val, "fi_norm": fi_norm, "fi_status": fi_s, "fi_col": fi_c,
                "stp": stp, "current": cur, "sapma": sap, "stp_status": stp_s, "stp_col": stp_c}
    except:
        return None


def analyze_teknik_gorunum(df):
    try:
        c = df["Close"]; v = df["Volume"]
        cur = float(c.iloc[-1])
        rsi_v = float(calc_rsi(c).iloc[-1])
        macd, sig = calc_macd(c)
        obv_up = bool(calc_obv(c, v).iloc[-1] > calc_obv(c, v).iloc[-10])
        ema20  = float(c.ewm(span=20, adjust=False).mean().iloc[-1])
        sma50  = float(c.rolling(50).mean().iloc[-1])
        sma200 = float(c.rolling(200).mean().iloc[-1])

        rows = []
        if rsi_v > 70:   rows.append(("RSI (14)", f"{rsi_v:.1f}", "Aşırı Alım — dikkat",  "#f59e0b"))
        elif rsi_v > 55: rows.append(("RSI (14)", f"{rsi_v:.1f}", "Güçlü Momentum",        "#10b981"))
        elif rsi_v > 45: rows.append(("RSI (14)", f"{rsi_v:.1f}", "Nötr",                  "#94a3b8"))
        else:            rows.append(("RSI (14)", f"{rsi_v:.1f}", "Zayıf / Aşırı Satım",  "#ef4444"))

        macd_bull = float(macd.iloc[-1]) > float(sig.iloc[-1])
        rows.append(("MACD", "", "Pozitif — Yükselen Baskı" if macd_bull else "Negatif — Düşen Baskı",
                     "#10b981" if macd_bull else "#ef4444"))

        rows.append(("OBV Trendi", "", "Yükseliyor ↑" if obv_up else "Düşüyor ↓",
                     "#10b981" if obv_up else "#ef4444"))

        for label, lvl in [("EMA20", ema20), ("SMA50", sma50), ("SMA200", sma200)]:
            above = cur > lvl
            rows.append((label, f"{lvl:,.2f}", "Fiyat Üstünde ✓" if above else "Fiyat Altında ✗",
                         "#10b981" if above else "#ef4444"))
        return rows
    except:
        return []


def analyze_teknik_seviyeler(df):
    try:
        c = df["Close"]; h = df["High"]; l = df["Low"]
        cur = float(c.iloc[-1])
        levels = {
            "52H Direnci": float(h.rolling(252).max().iloc[-1]),
            "20G Yüksek":  float(h.rolling(20).max().iloc[-1]),
            "SMA200":      float(c.rolling(200).mean().iloc[-1]),
            "SMA50":       float(c.rolling(50).mean().iloc[-1]),
            "EMA20":       float(c.ewm(span=20, adjust=False).mean().iloc[-1]),
            "20G Düşük":   float(l.rolling(20).min().iloc[-1]),
            "52L Desteği": float(l.rolling(252).min().iloc[-1]),
        }
        return {"current": cur, "levels": levels}
    except:
        return None


def analyze_canli_sinyaller(df):
    try:
        c = df["Close"]; v = df["Volume"]
        sinyaller = []

        avg_v = float(v.rolling(20).mean().iloc[-1]); son_v = float(v.iloc[-1])
        if son_v > avg_v * 1.5:
            sinyaller.append(("🔥", f"Hacim Artışı: Ortalamanın {son_v/avg_v:.1f}x üzerinde", "#f59e0b"))

        rsi_v = float(calc_rsi(c).iloc[-1])
        if rsi_v < 35:
            sinyaller.append(("🟢", f"RSI Aşırı Satım ({rsi_v:.1f}) — Dönüş ihtimali arttı", "#10b981"))
        elif rsi_v > 72:
            sinyaller.append(("🔴", f"RSI Aşırı Alım ({rsi_v:.1f}) — Dikkat gerekiyor", "#ef4444"))

        macd, sig = calc_macd(c)
        if float(macd.iloc[-1]) > float(sig.iloc[-1]) and float(macd.iloc[-2]) < float(sig.iloc[-2]):
            sinyaller.append(("⚡", "MACD Yükselen Kesişim — Momentum dönüyor",  "#10b981"))
        elif float(macd.iloc[-1]) < float(sig.iloc[-1]) and float(macd.iloc[-2]) > float(sig.iloc[-2]):
            sinyaller.append(("⚠️", "MACD Düşen Kesişim — Momentum zayıflıyor", "#ef4444"))

        obv   = calc_obv(c, v)
        p_up  = bool(c.iloc[-1] > c.iloc[-10]); o_up = bool(obv.iloc[-1] > obv.iloc[-10])
        if p_up and not o_up:
            sinyaller.append(("👀", "Fiyat yükseliyor ama OBV düşüyor — Gizli dağıtım olabilir", "#f59e0b"))
        elif not p_up and o_up:
            sinyaller.append(("🐳", "Fiyat düşerken OBV yükseliyor — Sessiz birikim sinyali", "#06b6d4"))

        sma200 = c.rolling(200).mean()
        if float(c.iloc[-1]) > float(sma200.iloc[-1]) and float(c.iloc[-5]) < float(sma200.iloc[-5]):
            sinyaller.append(("🚀", "SMA200 yukarı kırıldı — Güçlü yapısal sinyal", "#10b981"))

        if not sinyaller:
            sinyaller.append(("📊", "Belirgin aktif sinyal yok — Veri izleniyor", "#94a3b8"))
        return sinyaller
    except:
        return [("📊", "Hesaplanamadı", "#94a3b8")]


def analyze_yol_haritasi(df):
    try:
        c   = df["Close"]; cur = float(c.iloc[-1])
        s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
        a50 = cur > float(s50.iloc[-1]); s50r = float(s50.iloc[-1]) > float(s50.iloc[-6])
        a200 = cur > float(s200.iloc[-1])

        if a50 and s50r and a200:    bias, bc, bi = "Boğa",          "#10b981", "🐂"
        elif a50 and a200:           bias, bc, bi = "Temkinli Boğa", "#34d399", "📈"
        elif not a50 and not a200:   bias, bc, bi = "Ayı",           "#ef4444", "🐻"
        else:                        bias, bc, bi = "Nötr",          "#94a3b8", "⚖️"

        h50 = float(df["High"].rolling(50).max().iloc[-1])
        l50 = float(df["Low"].rolling(50).min().iloc[-1])
        mid = (h50 + l50) / 2
        if cur < mid: zone, zc, zi = "Discount (Ucuz Bölge)",  "#10b981", "💚"
        else:         zone, zc, zi = "Premium (Pahalı Bölge)", "#f59e0b", "🟡"

        if "Boğa" in bias and "Discount" in zone:
            sen = "Trend yukarı, fiyat ucuz bölgede. Kurumsal alım için elverişli koşullar. Tetikleyici sinyal bekleniyor."
        elif "Boğa" in bias:
            sen = "Trend yukarı ama fiyat pahalı bölgede. Yeni giriş için geri çekilme daha iyi fırsat sunabilir."
        elif "Ayı" in bias and "Discount" in zone:
            sen = "Trend aşağı, fiyat ucuz görünüyor. Ancak 'ucuz' daha ucuz olabilir. Trend dönüş sinyali olmadan risk yüksek."
        else:
            sen = "Trend aşağı ve fiyat pahalı bölgede. Yeni pozisyon için uygun koşullar henüz oluşmadı."

        return {"bias": bias, "bias_col": bc, "bias_icon": bi,
                "zone": zone, "zone_col": zc, "zone_icon": zi, "senaryo": sen}
    except:
        return None


# ────────────────────────────────────────────────────────────────────
# UI — GİRİŞ EKRANI
# ────────────────────────────────────────────────────────────────────
def render_login():
    st.markdown("""
    <div style="max-width:460px;margin:80px auto 0;text-align:center;">
        <div style="font-size:2.8rem;margin-bottom:8px;">📡</div>
        <div style="font-size:1.7rem;font-weight:800;color:#e2e8f0;margin-bottom:6px;">
            Smart Money Radar
        </div>
        <div style="font-size:0.88rem;color:#64748b;margin-bottom:8px;">
            BTC & ETH için kurumsal para akışı analizi
        </div>
        <div style="font-size:0.82rem;color:#6366f1;font-weight:600;margin-bottom:32px;">
            Ücretsiz · Kayıt gerektirmez · Anlık veri
        </div>
    </div>
    """, unsafe_allow_html=True)

    _, col, _ = st.columns([1, 2, 1])
    with col:
        name  = st.text_input("Adın", placeholder="Adın Soyadın")
        email = st.text_input("E-posta", placeholder="ornek@mail.com")
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        if st.button("📡  Ücretsiz Erişim Al", use_container_width=True):
            if name and email and "@" in email:
                st.session_state["logged_in"]  = True
                st.session_state["user_name"]  = name
                st.session_state["user_email"] = email
                st.rerun()
            else:
                st.error("Lütfen adınızı ve geçerli bir e-posta adresi girin.")

        st.markdown("""
        <div style="margin-top:20px;text-align:center;font-size:0.73rem;color:#475569;line-height:1.7;">
            Spam yok. İstediğiniz zaman çıkabilirsiniz.<br>
            <span style="color:#6366f1;font-weight:600;">Premium →</span>
            <span style="color:#64748b;">Tüm BIST hisseleri + AI analizi + Gelişmiş tarama</span>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# UI — PANEL RENDER FONKSİYONLARI
# ────────────────────────────────────────────────────────────────────
def _card_open(title: str, color: str = "#6366f1") -> str:
    return (f'<div class="smr-card">'
            f'<div class="smr-header" style="color:{color};">{title}</div>')

def _card_close() -> str:
    return "</div>"

def _cta(txt: str) -> str:
    return f'<div class="premium-cta">💎 {txt} → <b>Premium\'da</b></div>'


def render_long_radar(data: dict):
    score = data["score"]; status = data["status"]; sc = data["status_col"]
    bar_col = "#10b981" if score >= 75 else ("#f59e0b" if score >= 50 else "#ef4444")

    rows = ""
    for name, (passed, weight) in data["criteria"].items():
        c = "#e2e8f0" if passed else "#475569"
        rows += (f'<div style="display:flex;justify-content:space-between;padding:5px 0;'
                 f'border-bottom:1px solid #21262d;">'
                 f'<span style="font-size:0.77rem;color:{c};">{"✅" if passed else "⬜"} {name}</span>'
                 f'<span style="font-size:0.72rem;color:#64748b;">{weight}p</span></div>')

    st.markdown(
        _card_open("🎯 LONG RADAR — Akıllı Para Hazırlık Skoru", "#6366f1")
        + f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:10px;">'
          f'<span style="font-size:3rem;font-weight:800;color:{sc};">{score}</span>'
          f'<span style="font-size:0.9rem;color:#64748b;">/100</span>'
          f'<span style="font-size:0.88rem;font-weight:700;color:{sc};margin-left:8px;">{status}</span></div>'
        + f'<div style="background:#21262d;border-radius:4px;height:6px;margin-bottom:14px;">'
          f'<div style="background:{bar_col};width:{score}%;height:6px;border-radius:4px;"></div></div>'
        + rows
        + _cta("Tüm BIST hisseleri için LONG RADAR")
        + _card_close(),
        unsafe_allow_html=True
    )


def render_para_akisi(data):
    if not data:
        st.markdown(_card_open("💰 PARA AKIŞ İVMESİ & FİYAT DENGESİ")
                    + "<p style='color:#94a3b8;font-size:0.8rem;'>Veri alınamadı</p>"
                    + _card_close(), unsafe_allow_html=True); return

    fi_pct = int(min(100, max(0, (data["fi_norm"] + 1) / 2 * 100)))
    st.markdown(
        _card_open("💰 PARA AKIŞ İVMESİ & FİYAT DENGESİ", "#06b6d4")
        + f'<div style="margin-bottom:14px;">'
          f'<div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Para Akış İvmesi (Force Index)</div>'
          f'<div style="font-size:1.05rem;font-weight:700;color:{data["fi_col"]};">{data["fi_status"]}</div>'
          f'<div style="background:#21262d;border-radius:4px;height:5px;margin-top:6px;">'
          f'<div style="background:{data["fi_col"]};width:{fi_pct}%;height:5px;border-radius:4px;"></div></div></div>'
        + f'<div style="margin-bottom:8px;">'
          f'<div style="font-size:0.68rem;color:#64748b;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Fiyat Dengesi (STP Seviyesi)</div>'
          f'<div style="display:flex;justify-content:space-between;align-items:center;">'
          f'<div style="font-size:1.05rem;font-weight:700;color:{data["stp_col"]};">{data["stp_status"]}</div>'
          f'<div style="text-align:right;">'
          f'<div style="font-size:0.73rem;color:#64748b;">Denge: <b style="color:#e2e8f0;">{data["stp"]:,.2f}</b></div>'
          f'<div style="font-size:0.73rem;color:#64748b;">Sapma: <b style="color:{data["stp_col"]};">%{data["sapma"]:+.1f}</b></div>'
          f'</div></div></div>'
        + _cta("ICT Sniper + Harmonik Formasyonlar")
        + _card_close(),
        unsafe_allow_html=True
    )


def render_teknik_gorunum(rows: list):
    if not rows:
        st.markdown(_card_open("📊 TEKNİK GÖRÜNÜM")
                    + "<p style='color:#94a3b8;font-size:0.8rem;'>Veri alınamadı</p>"
                    + _card_close(), unsafe_allow_html=True); return

    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid #21262d;">'
        f'<span style="font-size:0.77rem;color:#94a3b8;">{lbl}{(" — " + val) if val else ""}</span>'
        f'<span style="font-size:0.77rem;font-weight:600;color:{col};">{txt}</span></div>'
        for lbl, val, txt, col in rows
    )
    st.markdown(
        _card_open("📊 TEKNİK GÖRÜNÜM", "#8b5cf6")
        + rows_html
        + _cta("HARSI + SuperTrend + Minervini")
        + _card_close(),
        unsafe_allow_html=True
    )


def render_teknik_seviyeler(data):
    if not data:
        st.markdown(_card_open("📐 TEKNİK SEVİYELER")
                    + "<p style='color:#94a3b8;font-size:0.8rem;'>Veri alınamadı</p>"
                    + _card_close(), unsafe_allow_html=True); return

    cur = data["current"]
    fiyat_html = (
        f'<div style="background:#1e2530;border:1px solid #6366f1;border-radius:8px;'
        f'padding:7px 12px;margin-bottom:10px;display:flex;justify-content:space-between;">'
        f'<span style="font-size:0.76rem;color:#94a3b8;">📍 GÜNCEL FİYAT</span>'
        f'<span style="font-size:0.88rem;font-weight:700;color:#6366f1;">${cur:,.2f}</span></div>'
    )
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:5px 6px;border-bottom:1px solid #21262d;">'
        f'<span style="font-size:0.75rem;color:#94a3b8;">{lbl}</span>'
        f'<div><span style="font-size:0.8rem;font-weight:600;color:#e2e8f0;">${lvl:,.2f}</span>'
        f'<span style="font-size:0.68rem;color:{"#10b981" if cur > lvl else "#ef4444"};margin-left:6px;">'
        f'{"▲" if cur > lvl else "▼"}</span></div></div>'
        for lbl, lvl in sorted(data["levels"].items(), key=lambda x: x[1], reverse=True)
    )
    st.markdown(
        _card_open("📐 TEKNİK SEVİYELER", "#f59e0b")
        + fiyat_html + rows_html
        + _cta("Fibonacci + Arz-Talep Bölgeleri")
        + _card_close(),
        unsafe_allow_html=True
    )


def render_canli_sinyaller(sinyaller: list):
    rows_html = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:7px 0;border-bottom:1px solid #21262d;">'
        f'<span style="font-size:0.95rem;flex-shrink:0;">{icon}</span>'
        f'<span style="font-size:0.77rem;color:{col};line-height:1.4;">{txt}</span></div>'
        for icon, txt, col in sinyaller
    )
    st.markdown(
        _card_open("⚡ CANLI SİNYALLER", "#06b6d4")
        + rows_html
        + _cta("RSI Uyumsuzluk + SFP Tuzak + Harmonik")
        + _card_close(),
        unsafe_allow_html=True
    )


def render_yol_haritasi(data):
    if not data:
        st.markdown(_card_open("🗺️ TEKNİK YOL HARİTASI")
                    + "<p style='color:#94a3b8;font-size:0.8rem;'>Veri alınamadı</p>"
                    + _card_close(), unsafe_allow_html=True); return

    st.markdown(
        _card_open("🗺️ TEKNİK YOL HARİTASI", "#10b981")
        + f'<div style="display:flex;gap:12px;margin-bottom:14px;">'
          f'<div style="flex:1;background:#1e2530;border:1px solid {data["bias_col"]}40;border-radius:8px;padding:10px;text-align:center;">'
          f'<div style="font-size:1.3rem;">{data["bias_icon"]}</div>'
          f'<div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;margin:2px 0;">Makro Yön</div>'
          f'<div style="font-size:0.85rem;font-weight:700;color:{data["bias_col"]};">{data["bias"]}</div></div>'
          f'<div style="flex:1;background:#1e2530;border:1px solid {data["zone_col"]}40;border-radius:8px;padding:10px;text-align:center;">'
          f'<div style="font-size:1.3rem;">{data["zone_icon"]}</div>'
          f'<div style="font-size:0.62rem;color:#64748b;text-transform:uppercase;margin:2px 0;">Konum</div>'
          f'<div style="font-size:0.8rem;font-weight:700;color:{data["zone_col"]};">{data["zone"]}</div></div></div>'
        + f'<div style="background:#161b22;border-left:3px solid {data["bias_col"]};padding:8px 12px;border-radius:0 6px 6px 0;">'
          f'<div style="font-size:0.76rem;color:#94a3b8;line-height:1.5;">{data["senaryo"]}</div></div>'
        + _cta("AI Destekli 25 Yıllık Stratejist Analizi")
        + _card_close(),
        unsafe_allow_html=True
    )


# ────────────────────────────────────────────────────────────────────
# UI — ANA UYGULAMA
# ────────────────────────────────────────────────────────────────────
def render_main():
    ticker = st.session_state.get("ticker", "BTC-USD")
    name   = st.session_state.get("user_name", "")

    # Header
    h1, _, h3 = st.columns([4, 2, 1])
    with h1:
        st.markdown(
            f'<div style="font-size:1.25rem;font-weight:800;color:#e2e8f0;">📡 Smart Money Radar</div>'
            f'<div style="font-size:0.72rem;color:#64748b;">Hoş geldin, {name} · Kurumsal para takip terminali</div>',
            unsafe_allow_html=True
        )
    with h3:
        if st.button("Çıkış", use_container_width=True):
            st.session_state.clear(); st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # BTC / ETH seçici
    c1, c2, c3 = st.columns([1, 1, 5])
    with c1:
        if st.button("₿  Bitcoin", use_container_width=True):
            st.session_state["ticker"] = "BTC-USD"; st.rerun()
    with c2:
        if st.button("⟠  Ethereum", use_container_width=True):
            st.session_state["ticker"] = "ETH-USD"; st.rerun()

    # Veri çek
    with st.spinner(f"{ticker} verisi Binance'den alınıyor..."):
        df = get_data(ticker)

    if df is None or len(df) < 50:
        st.error("Veri alınamadı. Lütfen birkaç dakika sonra tekrar deneyin.")
        return

    # Fiyat başlığı
    cur = float(df["Close"].iloc[-1]); prev = float(df["Close"].iloc[-2])
    change = ((cur - prev) / prev) * 100
    chg_col  = "#10b981" if change >= 0 else "#ef4444"
    chg_icon = "▲" if change >= 0 else "▼"
    coin_name = "Bitcoin" if "BTC" in ticker else "Ethereum"
    coin_icon = "₿" if "BTC" in ticker else "⟠"

    st.markdown(
        f'<div style="display:flex;align-items:baseline;flex-wrap:wrap;gap:10px;margin:12px 0 20px;">'
        f'<span style="font-size:1.6rem;font-weight:800;color:#e2e8f0;">{coin_icon} {coin_name}</span>'
        f'<span style="font-size:1.6rem;font-weight:700;color:#e2e8f0;">${cur:,.2f}</span>'
        f'<span style="font-size:0.95rem;font-weight:600;color:{chg_col};">{chg_icon} %{abs(change):.2f}</span>'
        f'<span style="font-size:0.7rem;color:#475569;margin-left:2px;">· Binance · Günlük kapanış</span>'
        f'</div>',
        unsafe_allow_html=True
    )

    # Analizleri hesapla
    radar_data   = analyze_long_radar(df)
    para_data    = analyze_para_akisi(df)
    gorunum_data = analyze_teknik_gorunum(df)
    seviye_data  = analyze_teknik_seviyeler(df)
    sinyal_data  = analyze_canli_sinyaller(df)
    yol_data     = analyze_yol_haritasi(df)

    # 2 kolonlu layout
    left, right = st.columns(2, gap="medium")
    with left:
        render_long_radar(radar_data)
        render_para_akisi(para_data)
        render_teknik_gorunum(gorunum_data)
    with right:
        render_canli_sinyaller(sinyal_data)
        render_yol_haritasi(yol_data)
        render_teknik_seviyeler(seviye_data)

    # Footer
    st.markdown("""
    <hr>
    <div style="text-align:center;padding:14px 0;">
        <div style="font-size:0.75rem;color:#475569;margin-bottom:10px;">
            Yatırım tavsiyesi değildir. Veriler Binance API üzerinden alınmaktadır.<br>
            <a href="https://twitter.com/SMRadar_2026" style="color:#6366f1;text-decoration:none;font-weight:600;">@SMRadar_2026</a>
        </div>
        <div style="display:inline-block;padding:10px 24px;
             background:linear-gradient(135deg,#1e1b4b,#1e2530);
             border:1px solid #4338ca;border-radius:8px;">
            <span style="font-size:0.8rem;color:#a5b4fc;">
                💎 Tüm BIST hisseleri · AI analizi · ICT Sniper · Formasyon tarayıcı ·
                <b><a href="https://smartmoneyradar.app" style="color:#818cf8;text-decoration:none;">
                Premium\'a geç →</a></b>
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# ANA AKIŞ
# ────────────────────────────────────────────────────────────────────
def main():
    if not st.session_state.get("logged_in"):
        render_login()
    else:
        render_main()

main()
