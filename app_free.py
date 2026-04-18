"""
Smart Money Radar — Ücretsiz Kripto Terminali
BTC & ETH için kurumsal para akışı analizi
Veriler: Binance API (ücretsiz, gerçek zamanlı)
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
# CSS — LIGHT MODE
# ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background: #f8fafc !important; }
[data-testid="stHeader"]           { background: transparent !important; }
[data-testid="stSidebar"]          { display: none !important; }
.block-container { padding: 1rem 2rem !important; max-width: 1200px !important; }
* { font-family: 'Inter', -apple-system, sans-serif; }

.stButton > button {
    background: #ffffff !important; color: #334155 !important;
    border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    font-weight: 600 !important; transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #6366f1 !important; color: #6366f1 !important;
    background: #f5f3ff !important;
}
.stTextInput > div > div > input {
    background: #ffffff !important; color: #1e293b !important;
    border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
}
.stTextInput > label { color: #64748b !important; font-size: 0.82rem !important; }

.smr-card {
    background: #ffffff; border: 1px solid #e2e8f0;
    border-radius: 12px; padding: 16px; margin-bottom: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.smr-header {
    font-size: 0.70rem; font-weight: 700; color: #6366f1;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
}
.premium-box {
    background: linear-gradient(135deg, #f5f3ff, #ede9fe);
    border: 1px solid #a5b4fc; border-radius: 10px;
    padding: 10px 12px; margin-top: 14px;
}
.premium-box-title {
    font-size: 0.67rem; font-weight: 800; color: #4338ca;
    text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 7px;
}
.lock-pill {
    display: inline-block; padding: 3px 8px; margin: 2px 2px 2px 0;
    background: #ede9fe; border: 1px solid #c4b5fd;
    border-radius: 5px; font-size: 0.67rem; color: #5b21b6; font-weight: 600;
}
.premium-link {
    display: block; text-align: center; margin-top: 8px;
    font-size: 0.76rem; font-weight: 700; color: #4338ca; text-decoration: none;
}
hr { border-color: #e2e8f0 !important; margin: 0.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# VERİ KATMANI — BİNANCE (çoklu endpoint) + yfinance + CoinGecko
# ────────────────────────────────────────────────────────────────────
_BINANCE_HOSTS = [
    "api.binance.com",
    "api1.binance.com",
    "api2.binance.com",
    "api3.binance.com",
]

def _parse_binance_klines(data):
    cols = ["ts","Open","High","Low","Close","Volume",
            "ct","qv","n","tbb","tbq","ig"]
    df = pd.DataFrame(data, columns=cols)
    for c in ["Open","High","Low","Close","Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["Date"] = pd.to_datetime(df["ts"], unit="ms")
    df.set_index("Date", inplace=True)
    df.index = df.index.tz_localize(None)
    return df[["Open","High","Low","Close","Volume"]].dropna()

def _try_binance(ticker: str, limit: int = 500):
    symbol = ticker.replace("-USD", "USDT").upper()
    for host in _BINANCE_HOSTS:
        url = f"https://{host}/api/v3/klines?symbol={symbol}&interval=1d&limit={limit}"
        try:
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if isinstance(data, dict) and "code" in data:
                continue
            if not data:
                continue
            df = _parse_binance_klines(data)
            if len(df) > 50:
                return df
        except Exception:
            continue
    return None

def _try_coingecko(ticker: str):
    coin_map = {
        "BTC-USD":  "bitcoin",
        "ETH-USD":  "ethereum",
        "SOL-USD":  "solana",
        "XRP-USD":  "ripple",
        "DOGE-USD": "dogecoin",
        "AVAX-USD": "avalanche-2",
    }
    coin_id  = coin_map.get(ticker.upper())
    if not coin_id:
        return None
    try:
        ohlc_url = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                    f"/ohlc?vs_currency=usd&days=365")
        ohlc = requests.get(ohlc_url, timeout=12).json()
        vol_url  = (f"https://api.coingecko.com/api/v3/coins/{coin_id}"
                    f"/market_chart?vs_currency=usd&days=365&interval=daily")
        vol_data = requests.get(vol_url, timeout=12).json()
        odf = pd.DataFrame(ohlc, columns=["ts","Open","High","Low","Close"])
        odf["Date"] = pd.to_datetime(odf["ts"], unit="ms").dt.normalize()
        odf.set_index("Date", inplace=True)
        vdf = pd.DataFrame(vol_data["total_volumes"], columns=["ts","Volume"])
        vdf["Date"] = pd.to_datetime(vdf["ts"], unit="ms").dt.normalize()
        vdf.set_index("Date", inplace=True)
        df = odf.join(vdf[["Volume"]], how="left")
        df["Volume"] = df["Volume"].fillna(0)
        df = df[["Open","High","Low","Close","Volume"]].dropna(subset=["Close"])
        return df if len(df) > 50 else None
    except Exception:
        return None

def _try_yfinance(ticker: str):
    try:
        import yfinance as yf
        raw = yf.download(ticker, period="2y", interval="1d",
                          progress=False, auto_adjust=True)
        if raw is None or len(raw) < 50:
            return None
        result = pd.DataFrame(index=raw.index)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            s = raw[col]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            result[col] = pd.to_numeric(s, errors="coerce")
        result = result.dropna()
        if result.index.tz is not None:
            result.index = result.index.tz_localize(None)
        return result if len(result) > 50 else None
    except Exception:
        return None

@st.cache_data(ttl=300)
def get_data(ticker: str):
    df = _try_binance(ticker)
    if df is not None:
        return df
    df = _try_yfinance(ticker)
    if df is not None:
        return df
    return _try_coingecko(ticker)


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
        if score >= 85:   status, sc = "🔥 Harekete Geç",     "#f59e0b"
        elif score >= 65: status, sc = "⚡ LONG İÇİN HAZIR",  "#10b981"
        elif pre_launch:  status, sc = "🎯 FİTİL ÇEKİLİYOR", "#06b6d4"
        elif score >= 40: status, sc = "📊 İzleme Listesi",   "#64748b"
        else:             status, sc = "⛔ Henüz Değil",       "#ef4444"
        return {"score": score, "status": status, "status_col": sc, "criteria": criteria}
    except:
        return {"score": 0, "status": "Hesaplanamadı", "status_col": "#64748b", "criteria": {}}


def _calc_stp(df):
    """Gerçek STP: Typical Price (HLC/3) üzerinden EWM span=6 (app.py ile aynı)"""
    tp   = (df["High"] + df["Low"] + df["Close"]) / 3
    ema1 = tp.ewm(span=6, adjust=False).mean()
    return ema1

def analyze_para_akisi(df):
    try:
        c = df["Close"]; v = df["Volume"]
        fi      = calc_force_index(c, v)
        fi_val  = float(fi.iloc[-1]); fi_prev = float(fi.iloc[-6])
        fi_max  = float(fi.abs().rolling(50).max().iloc[-1]) or 1
        fi_norm = fi_val / fi_max
        if fi_val > 0 and fi_val > fi_prev:   fi_s, fi_c = "Güçlenen Alım Baskısı",     "#10b981"
        elif fi_val > 0:                       fi_s, fi_c = "Alım Baskısı (zayıflıyor)", "#16a34a"
        elif fi_val < 0 and fi_val < fi_prev:  fi_s, fi_c = "Güçlenen Satış Baskısı",    "#ef4444"
        else:                                  fi_s, fi_c = "Satış Baskısı (azalıyor)",  "#dc2626"
        stp_series = _calc_stp(df)
        stp = float(stp_series.iloc[-1]); cur = float(c.iloc[-1])
        sap = ((cur - stp) / stp) * 100
        if sap > 15:    stp_s, stp_c = "Dengenin çok üzerinde — Isınma var", "#ef4444"
        elif sap > 5:   stp_s, stp_c = "Dengenin üzerinde",                  "#f59e0b"
        elif sap > -5:  stp_s, stp_c = "Denge yakınında",                    "#64748b"
        elif sap > -15: stp_s, stp_c = "Dengenin altında",                   "#10b981"
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
        if rsi_v > 70:   rows.append(("RSI (14)", f"{rsi_v:.1f}", "Aşırı Alım — dikkat", "#f59e0b"))
        elif rsi_v > 55: rows.append(("RSI (14)", f"{rsi_v:.1f}", "Güçlü Momentum",       "#10b981"))
        elif rsi_v > 45: rows.append(("RSI (14)", f"{rsi_v:.1f}", "Nötr",                 "#64748b"))
        else:            rows.append(("RSI (14)", f"{rsi_v:.1f}", "Zayıf / Aşırı Satım", "#ef4444"))
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
            sinyaller.append(("📊", "Belirgin aktif sinyal yok — Veri izleniyor", "#64748b"))
        return sinyaller
    except:
        return [("📊", "Hesaplanamadı", "#64748b")]


def analyze_yol_haritasi(df):
    try:
        c   = df["Close"]; cur = float(c.iloc[-1])
        s50 = c.rolling(50).mean(); s200 = c.rolling(200).mean()
        a50 = cur > float(s50.iloc[-1]); s50r = float(s50.iloc[-1]) > float(s50.iloc[-6])
        a200 = cur > float(s200.iloc[-1])
        if a50 and s50r and a200:  bias, bc, bi = "Boğa",          "#10b981", "🐂"
        elif a50 and a200:         bias, bc, bi = "Temkinli Boğa", "#16a34a", "📈"
        elif not a50 and not a200: bias, bc, bi = "Ayı",           "#ef4444", "🐻"
        else:                      bias, bc, bi = "Nötr",          "#64748b", "⚖️"
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
# UI YARDIMCILARI
# ────────────────────────────────────────────────────────────────────
def _card_open(title: str, color: str = "#6366f1") -> str:
    return (f'<div class="smr-card">'
            f'<div class="smr-header" style="color:{color};">{title}</div>')

def _card_close() -> str:
    return "</div>"

def _locked(*features) -> str:
    """Premium'da kilitli özellik kutusu — çoklu pill'ler"""
    pills = "".join(
        f'<span class="lock-pill">🔒 {f}</span>' for f in features
    )
    return (
        '<div class="premium-box">'
        '<div class="premium-box-title">💎 Premium\'da Açık</div>'
        f'{pills}'
        '<div style="text-align:center;margin-top:8px;">'
        '<span style="font-size:0.75rem;font-weight:700;color:#4338ca;">🔜 smartmoneyradar.app</span>'
        '<span style="font-size:0.70rem;color:#94a3b8;margin-left:6px;">— Çok Yakında</span>'
        '</div>'
        '</div>'
    )


# ────────────────────────────────────────────────────────────────────
# GRAFİKLER — FİYAT + STP & RSI
# ────────────────────────────────────────────────────────────────────
def render_grafikler(df: pd.DataFrame):
    last30 = df.tail(30).copy()
    c30    = last30["Close"]

    # STP: gerçek formül — Typical Price EWM span=6 (app.py ile aynı)
    stp  = _calc_stp(df).reindex(last30.index)

    # RSI
    rsi30 = calc_rsi(df["Close"]).reindex(last30.index)

    cur_price = float(c30.iloc[-1])
    stp_val   = float(stp.iloc[-1])
    above_stp = cur_price > stp_val
    fill_col  = "rgba(16,185,129,0.10)" if above_stp else "rgba(239,68,68,0.10)"
    rsi_val   = float(rsi30.iloc[-1])
    rsi_col   = "#ef4444" if rsi_val > 70 else ("#10b981" if rsi_val < 30 else "#6366f1")

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
        subplot_titles=["📈 Fiyat & STP Denge Çizgisi (Son 30 Gün)", "📊 RSI (14)"]
    )

    # ── Fiyat ──
    fig.add_trace(go.Scatter(
        x=last30.index, y=stp,
        name="STP Dengesi",
        line=dict(color="#f59e0b", width=1.5, dash="dash"),
        hovertemplate="%{y:,.2f}<extra>STP</extra>"
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=last30.index, y=c30,
        name="Fiyat",
        line=dict(color="#6366f1", width=2),
        fill="tonexty", fillcolor=fill_col,
        hovertemplate="%{y:,.2f}<extra>Fiyat</extra>"
    ), row=1, col=1)

    # ── RSI bantları ──
    fig.add_hrect(y0=70, y1=100, row=2, col=1,
                  fillcolor="rgba(239,68,68,0.06)", line_width=0)
    fig.add_hrect(y0=0, y1=30, row=2, col=1,
                  fillcolor="rgba(16,185,129,0.06)", line_width=0)
    for lvl, col, dash in [(70,"#ef4444","dash"),(50,"#94a3b8","dot"),(30,"#10b981","dash")]:
        fig.add_hline(y=lvl, row=2, col=1,
                      line_dash=dash, line_color=col, line_width=1)

    fig.add_trace(go.Scatter(
        x=last30.index, y=rsi30,
        name="RSI",
        line=dict(color=rsi_col, width=2),
        fill="tozeroy", fillcolor="rgba(99,102,241,0.06)",
        hovertemplate="%{y:.1f}<extra>RSI</extra>"
    ), row=2, col=1)

    fig.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="white",
        plot_bgcolor="#f8fafc",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.04,
                    font=dict(size=10, color="#475569")),
        hovermode="x unified",
        font=dict(family="Inter, sans-serif", color="#475569"),
    )
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", tickfont=dict(size=9))
    fig.update_yaxes(range=[0, 100], row=2, col=1)

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Premium CTA
    st.markdown(
        '<div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);'
        'border:1px solid #a5b4fc;border-radius:10px;padding:10px 16px;'
        'text-align:center;margin-bottom:20px;">'
        '<span style="font-size:0.78rem;color:#4338ca;font-weight:600;">'
        '💎 Premium\'da: İnteraktif ICT Mum Grafiği · VWAP Bantları · '
        'Fibonacci Seviyeleri · Arz-Talep Bölgeleri · 500+ BIST Hissesi · '
        '<span style="color:#6366f1;font-weight:800;">🔜 smartmoneyradar.app — Çok Yakında</span>'
        '</span></div>',
        unsafe_allow_html=True
    )


def render_premium_tablo():
    # (özellik, ücretsiz, premium, premium'da vurgu rengi)
    rows = [
        # ── Veri ──
        ("BTC & ETH Analizi",                        "%10", "%100", None),
        ("500+ BIST Hissesi (tam piyasa)",            "❌", "✅", "#10b981"),
        # ── Tarama Motorları ──
        ('"7 Onaylı" Kusursuz Algoritma',            "❌", "✅", "#f59e0b"),
        ("Altın Fırsatlar — Skor > 85 Taraması",     "❌", "✅", "#f59e0b"),
        ("Platin Formasyonlar Tarayıcısı",            "❌", "✅", "#f59e0b"),
        ("Her Sabah Hazır Setup Listesi",             "❌", "✅", "#f59e0b"),
        # ── Formasyon & Price Action ──
        ("AI Formasyon Tespiti (Fincan, Bayrak…)",   "❌", "✅", "#6366f1"),
        ("Harmonik Formasyonlar + Pop-up Uyarı",     "❌", "✅", "#6366f1"),
        ("Detaylı Price Action Paneli",               "❌", "✅", "#6366f1"),
        ("RSI Uyumsuzluk & SFP Tuzak Sinyali",       "❌", "✅", "#6366f1"),
        # ── Smart Money / ICT ──
        ("Smart Money Likidite Alımı Uyarısı",       "❌", "✅", "#06b6d4"),
        ("Balina İzi & Kurumsal Blok Takibi",        "❌", "✅", "#06b6d4"),
        ("Tam ICT Paneli (VWAP, OB, FVG…)",          "❌", "✅", "#06b6d4"),
        ("Arz-Talep Bölgeleri + Fibonacci",           "❌", "✅", "#06b6d4"),
        # ── AI & Trend ──
        ("Güçlü Boğa / Trend Şablonu Etiketi",       "❌", "✅", "#8b5cf6"),
        ("Minervini Stage Analysis",                  "❌", "✅", "#8b5cf6"),
        ("AI Destekli 25 Yıllık Stratejist Yorumu",  "❌", "✅", "#8b5cf6"),
    ]

    rows_html = ""
    for i, (feat, free, prem, pc) in enumerate(rows):
        bg = "#f9fafb" if i % 2 == 0 else "white"
        prem_col = pc if pc else "#10b981"
        rows_html += (
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'padding:5px 8px;border-bottom:1px solid #f1f5f9;background:{bg};">'
            f'<span style="font-size:0.72rem;color:#334155;">{feat}</span>'
            f'<div style="display:flex;gap:20px;min-width:90px;justify-content:center;">'
            f'<span style="font-size:0.82rem;">{free}</span>'
            f'<span style="font-size:0.82rem;color:{prem_col};font-weight:700;">{prem}</span>'
            f'</div></div>'
        )

    st.markdown(
        '<div class="smr-card">'
        # Başlık + slogan
        '<div class="smr-header" style="color:#6366f1;">🚀 Ücretsiz mi? Premium mu?</div>'
        '<div style="font-size:0.73rem;color:#475569;margin-bottom:10px;line-height:1.5;">'
        'Grafik çizmeye, indikatör kontrol etmeye son. Patron Terminal algoritması '
        '<b>tüm piyasayı saniyeler içinde tarar</b> — sen sadece izlersin ve kazancına odaklanırsın.'
        '</div>'
        # Tablo başlığı
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:5px 8px;border-bottom:2px solid #e2e8f0;margin-bottom:2px;">'
        '<span style="font-size:0.65rem;font-weight:700;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;">Özellik</span>'
        '<div style="display:flex;gap:20px;min-width:90px;justify-content:center;">'
        '<span style="font-size:0.65rem;font-weight:700;color:#94a3b8;text-transform:uppercase;">Ücretsiz</span>'
        '<span style="font-size:0.65rem;font-weight:700;color:#6366f1;text-transform:uppercase;">Premium</span>'
        '</div></div>'
        + rows_html +
        # "Ve çok daha fazlası" satırı
        '<div style="text-align:center;padding:7px 8px;font-size:0.75rem;'
        'font-weight:800;color:#6366f1;letter-spacing:0.3px;border-bottom:1px solid #f1f5f9;">'
        '✨ Ve çok daha fazlası...</div>'
        # Alt CTA
        '<div style="background:linear-gradient(135deg,#f5f3ff,#ede9fe);'
        'border-radius:8px;padding:10px 12px;margin-top:12px;text-align:center;">'
        '<div style="font-size:0.82rem;font-weight:900;color:#4338ca;margin-bottom:3px;">'
        '💎 Patron Terminal — Profesyonellerin Radarı</div>'
        '<div style="font-size:0.71rem;color:#6366f1;font-weight:600;">'
        '🔜 smartmoneyradar.app — Çok Yakında</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True
    )


# ────────────────────────────────────────────────────────────────────
# UI — GİRİŞ EKRANI
# ────────────────────────────────────────────────────────────────────
def render_login():
    st.markdown("""
    <div style="max-width:460px;margin:80px auto 0;text-align:center;">
        <div style="font-size:2.8rem;margin-bottom:8px;">📡</div>
        <div style="font-size:1.7rem;font-weight:800;color:#1e293b;margin-bottom:6px;">
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
                # Supabase'e kaydet
                try:
                    import requests as _req
                    _req.post(
                        "https://bzzdviatkawrguwoxqnp.supabase.co/rest/v1/users",
                        headers={
                            "apikey": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ6emR2aWF0a2F3cmd1d294cW5wIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0NjY5MDcsImV4cCI6MjA5MjA0MjkwN30.TdnoZDRruInxCaRWBfE0xz-w2L1bFaiXBDDjQgz2g9c",
                            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJ6emR2aWF0a2F3cmd1d294cW5wIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzY0NjY5MDcsImV4cCI6MjA5MjA0MjkwN30.TdnoZDRruInxCaRWBfE0xz-w2L1bFaiXBDDjQgz2g9c",
                            "Content-Type": "application/json",
                            "Prefer": "return=minimal"
                        },
                        json={"email": email, "name": name},
                        timeout=5
                    )
                except Exception:
                    pass  # Kayıt başarısız olsa bile kullanıcı içeri girsin
                st.session_state["logged_in"]  = True
                st.session_state["user_name"]  = name
                st.session_state["user_email"] = email
                st.rerun()
            else:
                st.error("Lütfen adınızı ve geçerli bir e-posta adresi girin.")
        st.markdown("""
        <div style="margin-top:20px;text-align:center;font-size:0.73rem;color:#94a3b8;line-height:1.7;">
            Spam yok. İstediğiniz zaman çıkabilirsiniz.<br>
            <span style="color:#6366f1;font-weight:600;">Premium →</span>
            <span style="color:#94a3b8;"> Tüm BIST hisseleri + AI analizi + Gelişmiş tarama</span>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────
# UI — PANEL RENDER FONKSİYONLARI
# ────────────────────────────────────────────────────────────────────
def render_long_radar(data: dict):
    score = data["score"]; status = data["status"]; sc = data["status_col"]
    bar_col = "#10b981" if score >= 75 else ("#f59e0b" if score >= 50 else "#ef4444")

    rows = ""
    for name, (passed, weight) in data["criteria"].items():
        c = "#1e293b" if passed else "#94a3b8"
        rows += (
            f'<div style="display:flex;justify-content:space-between;padding:6px 0;'
            f'border-bottom:1px solid #f1f5f9;">'
            f'<span style="font-size:0.77rem;color:{c};">{"✅" if passed else "⬜"} {name}</span>'
            f'<span style="font-size:0.72rem;color:#94a3b8;">{weight}p</span></div>'
        )

    st.markdown(
        _card_open("🎯 LONG RADAR — Akıllı Para Hazırlık Skoru", "#6366f1")
        + f'<div style="display:flex;align-items:baseline;gap:8px;margin-bottom:10px;">'
          f'<span style="font-size:3rem;font-weight:800;color:{sc};">{score}</span>'
          f'<span style="font-size:0.9rem;color:#94a3b8;">/100</span>'
          f'<span style="font-size:0.88rem;font-weight:700;color:{sc};margin-left:8px;">{status}</span></div>'
        + f'<div style="background:#f1f5f9;border-radius:4px;height:6px;margin-bottom:14px;">'
          f'<div style="background:{bar_col};width:{score}%;height:6px;border-radius:4px;"></div></div>'
        + rows
        + _locked(
            "500+ BIST Hissesi Taraması",
            "ICT Bottomline Skoru",
            "Haftalık + Aylık LONG RADAR",
            "Tam Otomatik Alarm",
            "Minervini Stage Analysis",
        )
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
          f'<div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Para Akış İvmesi (Force Index)</div>'
          f'<div style="font-size:1.05rem;font-weight:700;color:{data["fi_col"]};">{data["fi_status"]}</div>'
          f'<div style="background:#f1f5f9;border-radius:4px;height:5px;margin-top:6px;">'
          f'<div style="background:{data["fi_col"]};width:{fi_pct}%;height:5px;border-radius:4px;"></div></div></div>'
        + f'<div style="margin-bottom:8px;">'
          f'<div style="font-size:0.68rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.5px;margin-bottom:4px;">Fiyat Dengesi (STP Seviyesi)</div>'
          f'<div style="display:flex;justify-content:space-between;align-items:center;">'
          f'<div style="font-size:1.05rem;font-weight:700;color:{data["stp_col"]};">{data["stp_status"]}</div>'
          f'<div style="text-align:right;">'
          f'<div style="font-size:0.73rem;color:#94a3b8;">Denge: <b style="color:#1e293b;">{data["stp"]:,.2f}</b></div>'
          f'<div style="font-size:0.73rem;color:#94a3b8;">Sapma: <b style="color:{data["stp_col"]};">%{data["sapma"]:+.1f}</b></div>'
          f'</div></div></div>'
        + _locked(
            "Para Akışı Momentum Grafiği",
            "Smart Money Flow Göstergesi",
            "Kurumsal Blok İzleme",
            "Elder Ray + Force Index Detay",
        )
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
        f'padding:6px 0;border-bottom:1px solid #f1f5f9;">'
        f'<span style="font-size:0.77rem;color:#64748b;">{lbl}{(" — " + val) if val else ""}</span>'
        f'<span style="font-size:0.77rem;font-weight:600;color:{col};">{txt}</span></div>'
        for lbl, val, txt, col in rows
    )
    st.markdown(
        _card_open("📊 TEKNİK GÖRÜNÜM", "#8b5cf6")
        + rows_html
        + _locked(
            "Teknik Skor Kartı (A–F)",
            "ICT Odaklı Mum Grafiği",
            "SuperTrend + HARSI",
            "Minervini Kriterleri",
            "Bollinger + ATR Detay",
        )
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
        f'<div style="background:#f5f3ff;border:1px solid #a5b4fc;border-radius:8px;'
        f'padding:7px 12px;margin-bottom:10px;display:flex;justify-content:space-between;">'
        f'<span style="font-size:0.76rem;color:#64748b;">📍 GÜNCEL FİYAT</span>'
        f'<span style="font-size:0.88rem;font-weight:700;color:#6366f1;">${cur:,.2f}</span></div>'
    )
    rows_html = "".join(
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:5px 6px;border-bottom:1px solid #f1f5f9;">'
        f'<span style="font-size:0.75rem;color:#64748b;">{lbl}</span>'
        f'<div><span style="font-size:0.8rem;font-weight:600;color:#1e293b;">${lvl:,.2f}</span>'
        f'<span style="font-size:0.68rem;color:{"#10b981" if cur > lvl else "#ef4444"};margin-left:6px;">'
        f'{"▲" if cur > lvl else "▼"}</span></div></div>'
        for lbl, lvl in sorted(data["levels"].items(), key=lambda x: x[1], reverse=True)
    )
    st.markdown(
        _card_open("📐 TEKNİK SEVİYELER", "#f59e0b")
        + fiyat_html + rows_html
        + _locked(
            "Fibonacci Seviyeleri",
            "Arz-Talep Bölgeleri",
            "ICT Likidite Havuzları",
            "Günlük Pivot Noktaları",
            "Yearly Open / Quarterly Level",
        )
        + _card_close(),
        unsafe_allow_html=True
    )


def render_canli_sinyaller(sinyaller: list):
    rows_html = "".join(
        f'<div style="display:flex;align-items:flex-start;gap:8px;padding:7px 0;border-bottom:1px solid #f1f5f9;">'
        f'<span style="font-size:0.95rem;flex-shrink:0;">{icon}</span>'
        f'<span style="font-size:0.77rem;color:{col};line-height:1.4;">{txt}</span></div>'
        for icon, txt, col in sinyaller
    )
    st.markdown(
        _card_open("⚡ CANLI SİNYALLER", "#06b6d4")
        + rows_html
        + _locked(
            "RSI Uyumsuzluk (Divergence)",
            "SFP Tuzak Sinyali",
            "Harmonik Formasyon Alarmı",
            "AI Sinyal Açıklaması",
            "Push Bildirimi",
        )
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
          f'<div style="flex:1;background:#f8fafc;border:1px solid {data["bias_col"]}60;border-radius:8px;padding:10px;text-align:center;">'
          f'<div style="font-size:1.3rem;">{data["bias_icon"]}</div>'
          f'<div style="font-size:0.62rem;color:#94a3b8;text-transform:uppercase;margin:2px 0;">Makro Yön</div>'
          f'<div style="font-size:0.85rem;font-weight:700;color:{data["bias_col"]};">{data["bias"]}</div></div>'
          f'<div style="flex:1;background:#f8fafc;border:1px solid {data["zone_col"]}60;border-radius:8px;padding:10px;text-align:center;">'
          f'<div style="font-size:1.3rem;">{data["zone_icon"]}</div>'
          f'<div style="font-size:0.62rem;color:#94a3b8;text-transform:uppercase;margin:2px 0;">Konum</div>'
          f'<div style="font-size:0.8rem;font-weight:700;color:{data["zone_col"]};">{data["zone"]}</div></div></div>'
        + f'<div style="background:#f8fafc;border-left:3px solid {data["bias_col"]};padding:8px 12px;border-radius:0 6px 6px 0;margin-bottom:4px;">'
          f'<div style="font-size:0.76rem;color:#475569;line-height:1.5;">{data["senaryo"]}</div></div>'
        + _locked(
            "ICT Odaklı Mum Grafiği (interaktif)",
            "VWAP Sapma Analizi",
            "AI Destekli Stratejist Yorumu",
            "Risk / Ödül Hesaplayıcı",
            "Senaryo Bazlı Fiyat Hedefleri",
        )
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
            f'<div style="font-size:1.25rem;font-weight:800;color:#1e293b;">📡 Smart Money Radar</div>'
            f'<div style="font-size:0.72rem;color:#94a3b8;">Hoş geldin, {name} · Kurumsal para takip terminali</div>',
            unsafe_allow_html=True
        )
    with h3:
        if st.button("Çıkış", use_container_width=True):
            st.session_state.clear(); st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # Kripto seçici
    coins = [
        ("₿ BTC",  "BTC-USD"),
        ("⟠ ETH",  "ETH-USD"),
        ("◎ SOL",  "SOL-USD"),
        ("✕ XRP",  "XRP-USD"),
        ("Ð DOGE", "DOGE-USD"),
        ("▲ AVAX", "AVAX-USD"),
    ]
    btn_cols = st.columns(len(coins))
    for col, (label, t) in zip(btn_cols, coins):
        with col:
            active = ticker == t
            if st.button(label, use_container_width=True, type="primary" if active else "secondary"):
                st.session_state["ticker"] = t; st.rerun()

    # Veri çek
    with st.spinner(f"{ticker} verisi alınıyor..."):
        df = get_data(ticker)

    if df is None or len(df) < 50:
        st.error("Veri alınamadı. Lütfen birkaç dakika sonra tekrar deneyin.")
        return

    # Fiyat başlığı
    cur = float(df["Close"].iloc[-1]); prev = float(df["Close"].iloc[-2])
    change = ((cur - prev) / prev) * 100
    chg_col  = "#10b981" if change >= 0 else "#ef4444"
    chg_icon = "▲" if change >= 0 else "▼"
    coin_info = {
        "BTC-USD":  ("Bitcoin",  "₿"),
        "ETH-USD":  ("Ethereum", "⟠"),
        "SOL-USD":  ("Solana",   "◎"),
        "XRP-USD":  ("XRP",      "✕"),
        "DOGE-USD": ("Dogecoin", "Ð"),
        "AVAX-USD": ("Avalanche","▲"),
    }
    coin_name, coin_icon = coin_info.get(ticker, ("Kripto", "🪙"))

    st.markdown(
        f'<div style="display:flex;align-items:baseline;flex-wrap:wrap;gap:10px;margin:12px 0 20px;">'
        f'<span style="font-size:1.6rem;font-weight:800;color:#1e293b;">{coin_icon} {coin_name}</span>'
        f'<span style="font-size:1.6rem;font-weight:700;color:#1e293b;">${cur:,.2f}</span>'
        f'<span style="font-size:0.95rem;font-weight:600;color:{chg_col};">{chg_icon} %{abs(change):.2f}</span>'
        f'<span style="font-size:0.7rem;color:#94a3b8;margin-left:2px;">· Günlük kapanış</span>'
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

    # ── ÜST BÖLÜM: Grafik (70%) + Long Radar (30%) ──
    col_chart, col_radar = st.columns([7, 3], gap="medium")
    with col_chart:
        render_grafikler(df)
    with col_radar:
        render_long_radar(radar_data)

    # ── ALT BÖLÜM: 2 eşit kolon ──
    left, right = st.columns(2, gap="medium")
    with left:
        render_para_akisi(para_data)
        render_teknik_gorunum(gorunum_data)
        render_premium_tablo()
    with right:
        render_canli_sinyaller(sinyal_data)
        render_yol_haritasi(yol_data)
        render_teknik_seviyeler(seviye_data)

    # Footer
    st.markdown("""
    <hr>
    <div style="text-align:center;padding:14px 0;">
        <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:10px;">
            Yatırım tavsiyesi değildir. Veriler Binance API üzerinden alınmaktadır.<br>
            <a href="https://twitter.com/SMRadar_2026" style="color:#6366f1;text-decoration:none;font-weight:600;">@SMRadar_2026</a>
        </div>
        <div style="display:inline-block;padding:10px 28px;
             background:linear-gradient(135deg,#f5f3ff,#ede9fe);
             border:1px solid #a5b4fc;border-radius:10px;">
            <span style="font-size:0.8rem;color:#4338ca;font-weight:600;">
                💎 500+ BIST Hissesi · AI Analizi · ICT Sniper · Formasyon Tarayıcı · Alarm Sistemi ·
                <span style="color:#6366f1;font-weight:800;">🔜 smartmoneyradar.app — Çok Yakında</span>
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
