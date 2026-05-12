"""
Altcoin Potential Screener — @quantyscreener_bot
Referensi: @liabilitree FlowState

Data sources (semua cloud-friendly, tidak diblokir GitHub Actions):
- OHLCV      : CoinGecko API (gratis, no API key)
- Funding Rate: Bybit API (gratis, no API key)
- L/S Ratio  : Bybit API (gratis, no API key)
- OI         : Bybit API (gratis, no API key)
"""

import os
import io
import time
import requests
import warnings
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import spearmanr

warnings.filterwarnings('ignore')

# ── Credentials ──────────────────────────────────────────────
TELEGRAM_TOKEN   = os.environ.get('TELEGRAM_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '8724989560')

# ── API Base URLs (semua cloud-friendly) ──────────────────────
COINGECKO = 'https://api.coingecko.com/api/v3'
BYBIT     = 'https://api.bybit.com'

HEADERS = {'User-Agent': 'Mozilla/5.0 (compatible; screener-bot/1.0)'}

# ── Map symbol Binance → CoinGecko ID ────────────────────────
# CoinGecko pakai ID bukan symbol, jadi perlu mapping
COINGECKO_IDS = {
    'SOLUSDT'    : 'solana',
    'BTCUSDT'    : 'bitcoin',
    'ETHUSDT'    : 'ethereum',
    'BNBUSDT'    : 'binancecoin',
    'XRPUSDT'    : 'ripple',
    'AVAXUSDT'   : 'avalanche-2',
    'ADAUSDT'    : 'cardano',
    'DOGEUSDT'   : 'dogecoin',
    'LINKUSDT'   : 'chainlink',
    'DOTUSDT'    : 'polkadot',
    'NEARUSDT'   : 'near',
    'ATOMUSDT'   : 'cosmos',
    'LTCUSDT'    : 'litecoin',
    'UNIUSDT'    : 'uniswap',
    'AAVEUSDT'   : 'aave',
    'APTUSDT'    : 'aptos',
    'ARBUSDT'    : 'arbitrum',
    'OPUSDT'     : 'optimism',
    'INJUSDT'    : 'injective-protocol',
    'SUIUSDT'    : 'sui',
    'APEUSDT'    : 'apecoin',
    'KAITOUSDT'  : 'kaito',
    'ZECUSDT'    : 'zcash',
    'TAOUSDT'    : 'bittensor',
    'ONDOUSDT'   : 'ondo-finance',
    'LDOUSDT'    : 'lido-dao',
    'TONUSDT'    : 'the-open-network',
    'WLDUSDT'    : 'worldcoin-wld',
    'AXSUSDT'    : 'axie-infinity',
    'ORDIUSDT'   : 'ordinals',
}

# Bybit symbol (biasanya sama dengan Binance tapi tanpa USDT, lalu tambah USDT)
# Bybit pakai format: SOLUSDT, BTCUSDT, dll — sama persis

SYMBOLS = list(COINGECKO_IDS.keys())

# ── Tier thresholds ───────────────────────────────────────────
TIER_A_THRESHOLD = 0.60
TIER_B_THRESHOLD = 0.30
SHORT_THRESHOLD  = -0.40

# ── Triple confirmation ───────────────────────────────────────
FR_SQUEEZE_LEVEL = -0.10   # funding rate < -0.10% = squeeze candidate
FR_CROWDED_LEVEL = +0.10   # funding rate > +0.10% = crowded long
LS_CROWDED_LONG  = 1.50
LS_CROWDED_SHORT = 0.70

# ── Risk management ───────────────────────────────────────────
TOTAL_CAPITAL = 500
MAX_RISK_PCT  = 0.025
TARGET_RR     = 2.0
WIN_RATE      = 0.55
IC_THRESHOLD  = 0.03

# ── Colors ────────────────────────────────────────────────────
PINK_DARK  = '#c0155a'
PINK_MID   = '#e8327a'
PINK_LIGHT = '#f5a0c0'

SIGNAL_COLS = [
    'reversal_1d', 'liquidity_30d', 'liquidation_imbalance',
    'funding_rate_contrarian', 'vol_compression_30d', 'ls_ratio_contrarian',
    'momentum_30d', 'volume_compression_30d', 'oi_price_signal'
]


# ════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════

def safe_get(url, params=None, retries=3, delay=2):
    """Request dengan retry otomatis."""
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 429:
                # Rate limited — tunggu lebih lama
                wait = delay * (attempt + 2)
                print(f'    Rate limited, waiting {wait}s...')
                time.sleep(wait)
            else:
                print(f'    HTTP {r.status_code}: {url}')
                time.sleep(delay)
        except Exception as e:
            print(f'    Attempt {attempt+1} failed: {e}')
            time.sleep(delay)
    return None


# ════════════════════════════════════════════════════════════
# DATA FETCHING — CoinGecko (OHLCV)
# ════════════════════════════════════════════════════════════

def get_ohlcv_coingecko(symbol, days=31):
    """
    Ambil OHLCV dari CoinGecko.
    Gratis, tidak perlu API key, tidak diblokir cloud server.
    Return DataFrame dengan index datetime, kolom: open/high/low/close/volume
    """
    cg_id = COINGECKO_IDS.get(symbol)
    if not cg_id:
        return None

    # CoinGecko market_chart: harga per jam untuk 31 hari
    data = safe_get(
        f'{COINGECKO}/coins/{cg_id}/market_chart',
        params={'vs_currency': 'usd', 'days': days, 'interval': 'hourly'}
    )
    if not data:
        return None

    try:
        prices  = pd.DataFrame(data['prices'],        columns=['time', 'close'])
        volumes = pd.DataFrame(data['total_volumes'], columns=['time', 'volume'])

        prices['time']  = pd.to_datetime(prices['time'],  unit='ms')
        volumes['time'] = pd.to_datetime(volumes['time'], unit='ms')

        df = prices.merge(volumes, on='time').set_index('time')
        df['close']  = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Buat OHLCV dari close (CoinGecko hourly hanya punya close & volume)
        df['open']  = df['close'].shift(1)
        df['high']  = df[['open','close']].max(axis=1)
        df['low']   = df[['open','close']].min(axis=1)

        # Quote volume proxy (close × volume tidak tersedia, pakai volume langsung)
        df['quote_vol']        = df['volume']
        df['taker_buy_base']   = df['volume'] * 0.5   # estimasi 50/50
        df['taker_sell_base']  = df['volume'] * 0.5

        return df[['open','high','low','close','volume',
                   'taker_buy_base','taker_sell_base','quote_vol']].dropna()

    except Exception as e:
        print(f'    CoinGecko parse error {symbol}: {e}')
        return None


# ════════════════════════════════════════════════════════════
# DATA FETCHING — Bybit (Funding Rate, L/S, OI)
# ════════════════════════════════════════════════════════════

def get_funding_bybit(symbol, limit=200):
    """
    Funding rate historis dari Bybit.
    Bybit tidak blokir GitHub Actions IP.
    """
    data = safe_get(
        f'{BYBIT}/v5/market/funding/history',
        params={'category': 'linear', 'symbol': symbol, 'limit': limit}
    )
    if not data or data.get('retCode') != 0:
        return None

    try:
        rows = data['result']['list']
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df['fundingRateTimestamp'] = pd.to_datetime(df['fundingRateTimestamp'].astype(float), unit='ms')
        df['fundingRate']          = df['fundingRate'].astype(float)
        df = df.set_index('fundingRateTimestamp').sort_index()
        return df[['fundingRate']]
    except Exception as e:
        print(f'    Bybit funding parse error {symbol}: {e}')
        return None


def get_current_funding_bybit(symbol):
    """Funding rate terkini dari Bybit."""
    data = safe_get(
        f'{BYBIT}/v5/market/tickers',
        params={'category': 'linear', 'symbol': symbol}
    )
    if not data or data.get('retCode') != 0:
        return 0.0
    try:
        tickers = data['result']['list']
        if not tickers:
            return 0.0
        return float(tickers[0].get('fundingRate', 0))
    except Exception:
        return 0.0


def get_ls_ratio_bybit(symbol, period='1h', limit=200):
    """Long/Short ratio dari Bybit."""
    data = safe_get(
        f'{BYBIT}/v5/market/account-ratio',
        params={'category': 'linear', 'symbol': symbol,
                'period': period, 'limit': limit}
    )
    if not data or data.get('retCode') != 0:
        return None
    try:
        rows = data['result']['list']
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df['timestamp']    = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        df['buyRatio']     = df['buyRatio'].astype(float)
        df['sellRatio']    = df['sellRatio'].astype(float)
        df['longShortRatio'] = df['buyRatio'] / (df['sellRatio'] + 1e-8)
        df = df.set_index('timestamp').sort_index()
        return df[['longShortRatio']]
    except Exception as e:
        print(f'    Bybit L/S parse error {symbol}: {e}')
        return None


def get_oi_bybit(symbol, period='1h', limit=200):
    """Open Interest historis dari Bybit."""
    data = safe_get(
        f'{BYBIT}/v5/market/open-interest',
        params={'category': 'linear', 'symbol': symbol,
                'intervalTime': period, 'limit': limit}
    )
    if not data or data.get('retCode') != 0:
        return None
    try:
        rows = data['result']['list']
        if not rows:
            return None
        df = pd.DataFrame(rows)
        df['timestamp']       = pd.to_datetime(df['timestamp'].astype(float), unit='ms')
        df['openInterest']    = df['openInterest'].astype(float)
        df = df.set_index('timestamp').sort_index()
        return df[['openInterest']]
    except Exception as e:
        print(f'    Bybit OI parse error {symbol}: {e}')
        return None


# ════════════════════════════════════════════════════════════
# SIGNAL COMPUTATION
# ════════════════════════════════════════════════════════════

def compute_signals(symbol):
    """
    Hitung 9 sinyal untuk satu koin.
    OHLCV dari CoinGecko, Derivatif dari Bybit.
    """
    # Rate limit CoinGecko: 30 req/min untuk free tier
    # Tambah delay kecil antar koin
    time.sleep(2.5)

    kl = get_ohlcv_coingecko(symbol, days=31)
    if kl is None or len(kl) < 100:
        return None, None

    fund   = get_funding_bybit(symbol, 200)
    oi     = get_oi_bybit(symbol, '1h', 200)
    ls     = get_ls_ratio_bybit(symbol, '1h', 200)
    fr_now = get_current_funding_bybit(symbol)

    df = kl.copy()
    df['fwd_return_24h'] = df['close'].pct_change(24).shift(-24)

    # 1. reversal_1d — mean reversion
    df['reversal_1d'] = -df['close'].pct_change(24)

    # 2. liquidity_30d — log volume
    df['liquidity_30d'] = np.log1p(df['quote_vol'].rolling(30*24).mean())

    # 3. liquidation_imbalance — proxy dari taker ratio
    df['liquidation_imbalance'] = (
        df['taker_sell_base'] - df['taker_buy_base']
    ) / (df['volume'] + 1e-8)

    # 4. funding_rate_contrarian
    if fund is not None and len(fund) > 10:
        fh = fund.resample('1h').ffill().reindex(df.index, method='ffill')
        fm = fh['fundingRate'].rolling(90).mean()
        fs = fh['fundingRate'].rolling(90).std()
        df['funding_rate_contrarian'] = -((fh['fundingRate'] - fm) / (fs + 1e-8))
    else:
        df['funding_rate_contrarian'] = np.nan

    # 5. vol_compression_30d
    df['vol_compression_30d'] = -(
        df['volume'] / (df['volume'].rolling(30*24).mean() + 1e-8)
    )

    # 6. ls_ratio_contrarian
    if ls is not None and len(ls) > 10:
        lsh = ls.reindex(df.index, method='ffill')
        lsm = lsh['longShortRatio'].rolling(168).mean()
        lss = lsh['longShortRatio'].rolling(168).std()
        df['ls_ratio_contrarian'] = -((lsh['longShortRatio'] - lsm) / (lss + 1e-8))
    else:
        df['ls_ratio_contrarian'] = np.nan

    # 7. momentum_30d
    df['momentum_30d'] = df['close'].pct_change(30*24)

    # 8. volume_compression_30d
    vm = df['volume'].rolling(30*24).mean()
    vs = df['volume'].rolling(30*24).std()
    df['volume_compression_30d'] = vs / (vm + 1e-8)

    # 9. oi_price_signal
    if oi is not None and len(oi) > 10:
        oh = oi.reindex(df.index, method='ffill')
        df['oi_price_signal'] = (
            oh['openInterest'].pct_change(24) - df['close'].pct_change(24)
        )
    else:
        df['oi_price_signal'] = np.nan

    df['symbol'] = symbol

    # Metadata snapshot
    ls_now  = float(ls['longShortRatio'].iloc[-1]) if ls is not None and len(ls) > 0 else 1.0
    oi_usd  = float(oi['openInterest'].iloc[-1] * df['close'].iloc[-1]) if oi is not None and len(oi) > 0 else 0
    vol_24h = float(df['quote_vol'].tail(24).sum())

    meta = {
        'symbol'  : symbol,
        'close'   : float(df['close'].iloc[-1]),
        'fr'      : round(fr_now * 100, 4),
        'ls'      : round(ls_now, 2),
        'oi_usd'  : oi_usd,
        'vol_24h' : vol_24h,
    }

    keep = ['close', 'volume', 'fwd_return_24h', 'symbol'] + SIGNAL_COLS
    return df[keep].dropna(subset=['fwd_return_24h', 'reversal_1d']), meta


# ════════════════════════════════════════════════════════════
# IC, REGIME, SCORING
# ════════════════════════════════════════════════════════════

def compute_ic(master):
    ic_results = {}
    for sig in SIGNAL_COLS:
        daily_ic = []
        for date, grp in master.groupby(master.index.date):
            g = grp[['fwd_return_24h', sig]].dropna()
            if len(g) >= 5:
                ic_d, _ = spearmanr(g[sig], g['fwd_return_24h'])
                if not np.isnan(ic_d):
                    daily_ic.append(ic_d)
        if not daily_ic:
            continue
        ic_mean = np.mean(daily_ic)
        ic_std  = np.std(daily_ic)
        ic_results[sig] = {
            'IC Mean': round(ic_mean, 4),
            'ICIR'   : round(ic_mean / ic_std if ic_std > 0 else 0, 2),
        }
    return ic_results


def detect_regime(master, window=14):
    rows = []
    for date, grp in master.groupby(master.index.date):
        g = grp[['fwd_return_24h', 'momentum_30d', 'reversal_1d']].dropna()
        if len(g) < 5:
            continue
        ic_mom, _ = spearmanr(g['momentum_30d'], g['fwd_return_24h'])
        ic_rev, _ = spearmanr(g['reversal_1d'],  g['fwd_return_24h'])
        if not np.isnan(ic_mom) and not np.isnan(ic_rev):
            rows.append({'date': date, 'ic_mom': ic_mom, 'ic_rev': ic_rev})
    if not rows:
        return 'UNKNOWN', 0, 0.5

    rdf       = pd.DataFrame(rows).set_index('date')
    roll_diff = (rdf['ic_mom'].rolling(window).mean()
               - rdf['ic_rev'].rolling(window).mean())
    cur       = roll_diff.iloc[-1]

    if cur > 0.01:
        regime     = 'MOMENTUM'
        confidence = float(np.mean(roll_diff.dropna().tail(window) > 0))
    elif cur < -0.01:
        regime     = 'REVERSAL'
        confidence = float(np.mean(roll_diff.dropna().tail(window) < 0))
    else:
        regime     = 'MIXED'
        confidence = 0.5

    return regime, cur, confidence


def get_regime_weights(regime, confidence, ic_results):
    weights = {
        sig: abs(ic_results.get(sig, {}).get('IC Mean', 0.001))
        for sig in SIGNAL_COLS
    }
    boost = 1 + confidence * 1.5
    if regime == 'MOMENTUM':
        weights['momentum_30d']          *= boost
        weights['liquidity_30d']         *= boost * 0.8
        weights['reversal_1d']           *= 0.5
        weights['liquidation_imbalance'] *= 0.7
    elif regime == 'REVERSAL':
        weights['reversal_1d']             *= boost
        weights['liquidation_imbalance']   *= boost * 0.8
        weights['funding_rate_contrarian'] *= boost * 0.7
        weights['ls_ratio_contrarian']     *= boost * 0.7
        weights['momentum_30d']            *= 0.3
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def zscore_cross(s):
    return (s - s.mean()) / (s.std() + 1e-8)


def assign_tier(score, ls_ratio):
    if score >= TIER_A_THRESHOLD:
        return 'TIER_B' if ls_ratio > LS_CROWDED_LONG else 'TIER_A'
    elif score >= TIER_B_THRESHOLD:
        return 'TIER_B'
    elif score <= SHORT_THRESHOLD:
        return 'SHORT'
    return 'NEUTRAL'


def compute_composite(master, meta_df, ic_results, regime_weights):
    cutoff = master.index.max() - pd.Timedelta(hours=24)
    snap   = master[master.index >= cutoff].groupby('symbol')[SIGNAL_COLS + ['close']].mean()

    for sig in SIGNAL_COLS:
        snap[f'{sig}_z'] = zscore_cross(snap[sig])

    parts = []
    for sig in SIGNAL_COLS:
        z_col = f'{sig}_z'
        if z_col not in snap.columns:
            continue
        ic_val    = ic_results.get(sig, {}).get('IC Mean', 0)
        direction = 1 if ic_val >= 0 else -1
        weight    = regime_weights.get(sig, 0)
        parts.append(direction * snap[z_col] * weight)

    snap['composite_score'] = sum(parts)
    snap = snap.join(meta_df[['fr', 'ls', 'oi_usd', 'vol_24h']], how='left')
    snap['tier'] = snap.apply(
        lambda r: assign_tier(r['composite_score'], r.get('ls', 1.0)), axis=1
    )

    ranking = snap[['composite_score', 'close', 'fr', 'ls', 'oi_usd', 'vol_24h', 'tier']]\
                  .dropna(subset=['composite_score'])\
                  .sort_values('composite_score', ascending=False)
    ranking['rank'] = range(1, len(ranking) + 1)
    return ranking


# ════════════════════════════════════════════════════════════
# TRIPLE CONFIRMATION
# ════════════════════════════════════════════════════════════

def triple_confirmation(score, fr_pct, ls_ratio, tier):
    if tier == 'NEUTRAL':
        return False, None

    direction  = 'LONG' if tier in ['TIER_A', 'TIER_B'] else 'SHORT'
    pass_count = 1  # score sudah pass

    if direction == 'LONG':
        if fr_pct < 0:
            pass_count += 1
        if ls_ratio < LS_CROWDED_LONG:
            pass_count += 1
    else:
        if fr_pct > 0:
            pass_count += 1
        if ls_ratio > LS_CROWDED_LONG:
            pass_count += 1

    return pass_count >= 2, direction


# ════════════════════════════════════════════════════════════
# CHART — 3 panel FlowState
# ════════════════════════════════════════════════════════════

def build_chart(ranking):
    shortlist = ranking[ranking['tier'].isin(['TIER_A', 'TIER_B', 'SHORT'])]\
                    .sort_values('composite_score', ascending=True).tail(12)

    syms    = [s.replace('USDT', '') for s in shortlist.index]
    scores  = shortlist['composite_score'].values
    fr_vals = shortlist['fr'].fillna(0).values
    ls_vals = shortlist['ls'].fillna(1.0).values
    tiers   = shortlist['tier'].values

    score_colors = [
        PINK_DARK if t == 'TIER_A' else (PINK_LIGHT if t == 'TIER_B' else '#888899')
        for t in tiers
    ]
    fr_colors = [
        PINK_DARK if v < FR_SQUEEZE_LEVEL else (PINK_MID if v < 0 else PINK_LIGHT)
        for v in fr_vals
    ]
    ls_colors = [
        '#e74c3c' if v > LS_CROWDED_LONG else (PINK_DARK if v < LS_CROWDED_SHORT else PINK_LIGHT)
        for v in ls_vals
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, max(6, len(syms) * 0.6)))
    fig.patch.set_facecolor('#ffffff')
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M')
    fig.suptitle(
        f'Altcoin Potential Screener — {now_str}\nShortlist Kandidat Derivatif Trading',
        fontsize=11, fontweight='bold', color='#111118', y=1.03
    )
    y_pos = np.arange(len(syms))

    def style_ax(ax):
        ax.set_facecolor('#fafafa')
        ax.tick_params(labelsize=7, colors='#555566')
        for sp in ax.spines.values():
            sp.set_color('#ddddee')

    # Panel 1: Multi-Factor Score
    ax1 = axes[0]
    style_ax(ax1)
    ax1.barh(y_pos, scores, color=score_colors, height=0.6, edgecolor='none')
    for i, val in enumerate(scores):
        ax1.text(val + 0.01, i, f'{val:+.2f}', va='center', ha='left',
                 fontsize=8, color='#111118', fontfamily='monospace')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(syms, fontsize=8.5, color='#111118')
    ax1.set_xlabel('Multi-Factor Score', fontsize=8, color='#555566')
    ax1.set_title('Multi-Factor Score\n(IC-weighted)', fontsize=9,
                  fontweight='bold', color='#111118')
    ax1.axvline(0, color='#cccccc', lw=1)
    ax1.legend(handles=[
        mpatches.Patch(color=PINK_DARK,  label='Tier A'),
        mpatches.Patch(color=PINK_LIGHT, label='Tier B'),
    ], fontsize=7, loc='lower right', facecolor='white', edgecolor='#ddddee')

    # Panel 2: Funding Rate
    ax2 = axes[1]
    style_ax(ax2)
    ax2.barh(y_pos, fr_vals, color=fr_colors, height=0.6, edgecolor='none')
    for i, val in enumerate(fr_vals):
        ax2.text(val - 0.02 if val < 0 else val + 0.02, i, f'{val:+.2f}%',
                 va='center', ha='right' if val < 0 else 'left',
                 fontsize=8, color='#111118', fontfamily='monospace')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(syms, fontsize=8.5, color='#111118')
    ax2.set_xlabel('Funding Rate (%)', fontsize=8, color='#555566')
    ax2.set_title('Funding Rate\n(negatif = short bias = potensi squeeze)',
                  fontsize=9, fontweight='bold', color='#111118')
    ax2.axvline(0, color='#cccccc', lw=1)
    ax2.legend(handles=[
        mpatches.Patch(color=PINK_DARK,  label='Negatif (bullish)'),
        mpatches.Patch(color=PINK_LIGHT, label='Positif (crowded)'),
    ], fontsize=7, loc='lower right', facecolor='white', edgecolor='#ddddee')

    # Panel 3: L/S Ratio
    ax3 = axes[2]
    style_ax(ax3)
    ax3.barh(y_pos, ls_vals, color=ls_colors, height=0.6, edgecolor='none')
    for i, val in enumerate(ls_vals):
        ax3.text(val + 0.01, i, f'{val:.2f}', va='center', ha='left',
                 fontsize=8, color='#111118', fontfamily='monospace')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(syms, fontsize=8.5, color='#111118')
    ax3.set_xlabel('Long/Short Ratio', fontsize=8, color='#555566')
    ax3.set_title('Long/Short Ratio\n(>1.5 = crowded long = hati-hati)',
                  fontsize=9, fontweight='bold', color='#111118')
    ax3.axvline(1.0, color='#cccccc', lw=1)
    ax3.axvline(LS_CROWDED_LONG, color='#ffaaaa', lw=0.8, ls='--',
                alpha=0.7, label=f'Warning ({LS_CROWDED_LONG})')
    ax3.legend(fontsize=7, loc='lower right', facecolor='white', edgecolor='#ddddee')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    plt.close()
    return buf


# ════════════════════════════════════════════════════════════
# TRADING PLAN
# ════════════════════════════════════════════════════════════

def build_plan(symbol, entry, capital, direction='LONG'):
    sl     = entry * (1 - MAX_RISK_PCT) if direction == 'LONG' else entry * (1 + MAX_RISK_PCT)
    tp     = entry * (1 + MAX_RISK_PCT * TARGET_RR) if direction == 'LONG' else entry * (1 - MAX_RISK_PCT * TARGET_RR)
    risk   = capital * MAX_RISK_PCT
    reward = capital * MAX_RISK_PCT * TARGET_RR
    ev     = (WIN_RATE * reward) - ((1 - WIN_RATE) * risk)
    return dict(
        symbol=symbol, direction=direction,
        entry=round(entry, 6), sl=round(sl, 6), tp=round(tp, 6),
        sl_pct=round(MAX_RISK_PCT * 100, 2),
        tp_pct=round(MAX_RISK_PCT * TARGET_RR * 100, 2),
        capital=round(capital, 2), risk=round(risk, 2),
        reward=round(reward, 2), ev=round(ev, 2), ev_pos=ev > 0
    )


# ════════════════════════════════════════════════════════════
# TELEGRAM
# ════════════════════════════════════════════════════════════

def send_photo(token, chat_id, buf, caption=''):
    try:
        buf.seek(0)
        r = requests.post(
            f'https://api.telegram.org/bot{token}/sendPhoto',
            data={'chat_id': chat_id, 'caption': caption},
            files={'photo': ('screener.png', buf, 'image/png')},
            timeout=30
        )
        return r.status_code == 200
    except Exception as e:
        print(f'send_photo error: {e}')
        return False


def send_message(token, chat_id, text):
    try:
        chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
        for chunk in chunks:
            requests.post(
                f'https://api.telegram.org/bot{token}/sendMessage',
                json={'chat_id': chat_id, 'text': chunk},
                timeout=30
            )
        return True
    except Exception as e:
        print(f'send_message error: {e}')
        return False


def format_usd(val):
    if val >= 1e9: return f'${val/1e9:.1f}B'
    if val >= 1e6: return f'${val/1e6:.1f}M'
    if val >= 1e3: return f'${val/1e3:.1f}K'
    return f'${val:.0f}'


def format_report(ranking, plans, regime, confidence, regime_score):
    now  = datetime.now().strftime('%d %b %Y %H:%M')
    msg  = f'ALTCOIN SCREENER — {now} WIB\n'
    msg += f'Regime: {regime} ({confidence:.0%}) | Score: {regime_score:+.4f}\n'
    msg += f'Data: CoinGecko + Bybit\n'
    msg += '─' * 38 + '\n\n'

    longs  = [p for p in plans if p['direction'] == 'LONG']
    shorts = [p for p in plans if p['direction'] == 'SHORT']

    if longs:
        msg += 'LONG CANDIDATES\n'
        for p in longs:
            row    = ranking.loc[p['symbol']]
            ev_tag = 'EV+' if p['ev_pos'] else 'EV-'
            msg += f"  {p['symbol'].replace('USDT','')} [{row['tier']}]\n"
            msg += f"  Entry ${p['entry']} | SL ${p['sl']} | TP ${p['tp']}\n"
            msg += f"  FR={row['fr']:+.2f}% | L/S={row['ls']:.2f} | [{ev_tag}] EV=${p['ev']:+.2f}\n\n"

    if shorts:
        msg += 'SHORT CANDIDATES\n'
        for p in shorts:
            row    = ranking.loc[p['symbol']]
            ev_tag = 'EV+' if p['ev_pos'] else 'EV-'
            msg += f"  {p['symbol'].replace('USDT','')} [SHORT]\n"
            msg += f"  Entry ${p['entry']} | SL ${p['sl']} | TP ${p['tp']}\n"
            msg += f"  FR={row['fr']:+.2f}% | L/S={row['ls']:.2f} | [{ev_tag}] EV=${p['ev']:+.2f}\n\n"

    if not longs and not shorts:
        msg += 'Tidak ada kandidat yang lolos triple confirmation.\n\n'

    if plans:
        total_risk = sum(p['risk'] for p in plans)
        total_ev   = sum(p['ev'] for p in plans)
        msg += f'RISK SUMMARY\n'
        msg += f'  Max loss : -${total_risk:.2f}\n'
        msg += f'  Total EV : +${total_ev:.2f}\n\n'

    msg += 'TOP 10 SCORE\n'
    for sym, row in ranking.head(10).iterrows():
        tier_s = {'TIER_A':'A','TIER_B':'B','SHORT':'S','NEUTRAL':'N'}.get(row['tier'], '?')
        msg += f"  {sym.replace('USDT',''):<10} {row['composite_score']:+.3f} [{tier_s}] FR={row.get('fr',0):+.2f}%\n"

    msg += '\nData valid intraday. Update tiap 4 jam.'
    return msg


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    print(f'[{datetime.now().strftime("%H:%M:%S")}] Screener started')
    print(f'Data sources: CoinGecko (OHLCV) + Bybit (Funding/L/S/OI)')

    if not TELEGRAM_TOKEN:
        print('ERROR: TELEGRAM_TOKEN tidak ada di environment variable')
        return

    # 1. Fetch semua koin
    print(f'Fetching {len(SYMBOLS)} coins...')
    all_data, all_meta, failed = [], [], []

    for sym in SYMBOLS:
        print(f'  {sym}...', end=' ', flush=True)
        df, meta = compute_signals(sym)
        if df is not None and len(df) > 20 and meta is not None:
            all_data.append(df)
            all_meta.append(meta)
            print(f'OK | FR={meta["fr"]:+.2f}% | L/S={meta["ls"]:.2f}')
        else:
            failed.append(sym)
            print('SKIP')

    if not all_data:
        send_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
                     'ERROR: Semua koin gagal. Kemungkinan CoinGecko rate limit. Coba lagi dalam 1 jam.')
        return

    master  = pd.concat(all_data)
    meta_df = pd.DataFrame(all_meta).set_index('symbol')
    print(f'Master: {len(master):,} rows | {master["symbol"].nunique()} coins')
    if failed:
        print(f'Failed ({len(failed)}): {failed}')

    # 2. IC
    print('Computing IC...')
    ic_results = compute_ic(master)

    # 3. Regime
    print('Detecting regime...')
    regime, regime_score, confidence = detect_regime(master)
    regime_weights = get_regime_weights(regime, confidence, ic_results)
    print(f'Regime: {regime} ({confidence:.0%})')

    # 4. Score & ranking
    print('Computing scores...')
    ranking = compute_composite(master, meta_df, ic_results, regime_weights)

    # 5. Triple confirmation
    print('Triple confirmation...')
    plans       = []
    score_total = ranking[ranking['tier'] != 'NEUTRAL']['composite_score'].abs().sum()

    for sym, row in ranking[ranking['tier'] != 'NEUTRAL'].iterrows():
        confirmed, direction = triple_confirmation(
            row['composite_score'], row.get('fr', 0),
            row.get('ls', 1.0), row['tier']
        )
        if confirmed and direction:
            alloc = (abs(row['composite_score']) / score_total) * TOTAL_CAPITAL \
                    if score_total > 0 else 100
            plans.append(build_plan(sym, row['close'], alloc, direction))
            print(f'  CONFIRMED: {sym} {direction}')

    # 6. Chart
    print('Building chart...')
    chart_buf = build_chart(ranking)

    # 7. Send
    print('Sending to Telegram...')
    now_str = datetime.now().strftime('%d %b %Y %H:%M WIB')
    report  = format_report(ranking, plans, regime, confidence, regime_score)

    send_photo(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, chart_buf, f'Screener — {now_str}')
    send_message(TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, report)

    print(f'[{datetime.now().strftime("%H:%M:%S")}] Done — {len(plans)} candidates')


if __name__ == '__main__':
    main()

