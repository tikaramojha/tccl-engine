"""
MTF-Q-LDA+ (Multi-Timeframe Quantum-LDA Plus)
=============================================
Full implementation — drop in daily CSV and run.

Usage:
    python mtf_qlda_plus.py data.csv
    python mtf_qlda_plus.py data.csv data_30m.csv   # optional 30m file

CSV format required:
    date,open,high,low,close,volume
    2024-01-01,42000,43000,41500,42800,1234.5
    ...

Output:
    mtf_qlda_output.csv  — signals, equity curve, weights
    mtf_qlda_report.txt  — performance summary
"""

import numpy as np
import pandas as pd
import math
import sys
import os
from scipy.signal import hilbert
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from datetime import datetime

EPS = 1e-12

# ═══════════════════════════════════════════════════════════
# MATH UTILITIES
# ═══════════════════════════════════════════════════════════

def softmax(p):
    e = np.exp(p - np.nanmax(p))
    return e / (np.nansum(e) + EPS)

def wrap_phase(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def hilbert_phase(series, window):
    """
    Hilbert instantaneous phase for a price series.
    Returns: dphase (phase derivative), phases (raw phase)
    """
    ln = np.log(np.maximum(series, EPS))
    phases = np.full(len(series), np.nan)

    for i in range(window, len(series)):
        seg = ln[i - window:i]
        t = np.arange(window)
        coefs = np.polyfit(t, seg, 1)
        trend = np.polyval(coefs, t)
        segd = seg - trend
        analytic = hilbert(segd)
        phases[i] = np.angle(analytic[-1])

    dphase = np.diff(phases, prepend=np.nan)
    dphase = wrap_phase(dphase)
    return dphase, phases

def compute_C_gamma_S(series, window):
    """
    Coherence proxy (C), tailness/kurtosis proxy (gamma), trend strength (S)
    """
    ln = np.log(np.maximum(series, EPS))
    n = len(series)
    C     = np.full(n, np.nan)
    gamma = np.full(n, np.nan)
    S     = np.full(n, np.nan)

    for i in range(window, n):
        seg = ln[i - window:i]
        t = np.arange(window)
        segd = seg - np.polyval(np.polyfit(t, seg, 1), t)
        analytic = hilbert(segd)
        amp = np.abs(analytic)
        C[i] = np.mean(amp) / (np.std(amp) + EPS)
        S[i] = math.tanh(np.mean(segd) / (np.std(segd) + EPS))
        r = np.diff(seg)
        if len(r) > 1:
            mu_r = np.mean(r)
            sigma_r = np.std(r)
            gamma[i] = np.mean((r - mu_r)**4) / ((sigma_r**2 + EPS)**2)
        else:
            gamma[i] = 0.0

    return C, gamma, S

# ═══════════════════════════════════════════════════════════
# WEIGHT OPTIMIZER
# ═══════════════════════════════════════════════════════════

def optimize_weights(feat_mat, y, prior_w, reg=5.0):
    """
    L-BFGS-B optimization: minimize -corr(composite, next_ret) + L2 regularization
    """
    if np.isnan(feat_mat).any() or len(y) < 20:
        return prior_w

    p0 = np.log(prior_w + 1e-9)

    def obj(p):
        w = softmax(p)
        comp = feat_mat.dot(w)
        mask = ~np.isnan(comp) & ~np.isnan(y)
        if mask.sum() < 5:
            return 1.0
        corr = np.corrcoef(comp[mask], y[mask])[0, 1]
        if np.isnan(corr):
            corr = 0.0
        regterm = reg * np.sum((w - prior_w)**2)
        return -corr + regterm

    res = minimize(obj, p0, method='L-BFGS-B', options={'maxiter': 200})
    if res.success:
        return softmax(res.x)
    else:
        return prior_w

# ═══════════════════════════════════════════════════════════
# FEATURE BUILDER
# ═══════════════════════════════════════════════════════════

def build_features(df_day, df_30m=None):
    """
    Build all features for three timeframes: 30m, daily, weekly
    """
    prices = df_day['close'].values
    n = len(prices)

    print(f"  Building features for {n} daily bars...")

    # --- 30m series ---
    if df_30m is not None:
        high_res = df_30m['close'].values
        print(f"  Using real 30m data: {len(high_res)} bars")
    else:
        minutes_per_day = 48
        high_res = np.repeat(prices, minutes_per_day)
        # Add small noise to simulate intraday variation
        np.random.seed(42)
        high_res = high_res * (1 + 0.002 * np.random.randn(len(high_res)))
        print(f"  Synthesizing 30m data: {len(high_res)} bars")

    # --- Weekly series ---
    weekly = prices[::7]
    print(f"  Weekly series: {len(weekly)} bars")

    # --- Hilbert phases ---
    w_30m, w_daily, w_weekly = 12, 5, 4

    print("  Computing Hilbert phases (30m)...")
    dphase_30m_full, _ = hilbert_phase(high_res, w_30m)
    # Downsample to daily
    dphase_30m = dphase_30m_full.reshape(n, -1)[:, -1] if df_30m is None else \
                 np.array([dphase_30m_full[min(i*48+47, len(dphase_30m_full)-1)] for i in range(n)])

    print("  Computing Hilbert phases (daily)...")
    dphase_daily, _ = hilbert_phase(prices, w_daily)

    print("  Computing Hilbert phases (weekly)...")
    dphase_weekly_full, _ = hilbert_phase(weekly, w_weekly)
    dphase_weekly = np.full(n, np.nan)
    for i, idx in enumerate(range(0, n, 7)):
        val = dphase_weekly_full[i] if i < len(dphase_weekly_full) else np.nan
        for j in range(7):
            if idx + j < n:
                dphase_weekly[idx + j] = val

    # --- Velocities ---
    vel_30m    = np.diff(dphase_30m,    prepend=np.nan)
    vel_daily  = np.diff(dphase_daily,  prepend=np.nan)
    vel_weekly = np.diff(dphase_weekly, prepend=np.nan)

    # --- C, gamma, S ---
    print("  Computing coherence features (30m)...")
    C_30m, gamma_30m, S_30m = compute_C_gamma_S(
        high_res.reshape(n, -1)[:, -1] if df_30m is None else high_res[:n], w_30m)

    print("  Computing coherence features (daily)...")
    C_daily, gamma_daily, S_daily = compute_C_gamma_S(prices, w_daily)

    print("  Computing coherence features (weekly)...")
    C_weekly_full, gamma_weekly_full, S_weekly_full = compute_C_gamma_S(weekly, w_weekly)

    C_weekly     = np.full(n, np.nan)
    gamma_weekly = np.full(n, np.nan)
    S_weekly     = np.full(n, np.nan)
    for i, idx in enumerate(range(0, n, 7)):
        valC = C_weekly_full[i]     if i < len(C_weekly_full)     else np.nan
        valg = gamma_weekly_full[i] if i < len(gamma_weekly_full) else np.nan
        valS = S_weekly_full[i]     if i < len(S_weekly_full)     else np.nan
        for j in range(7):
            if idx + j < n:
                C_weekly[idx+j]     = valC
                gamma_weekly[idx+j] = valg
                S_weekly[idx+j]     = valS

    feat = pd.DataFrame({
        'dphase_30m':   dphase_30m,   'vel_30m':    vel_30m,
        'C_30m':        C_30m,        'g_30m':      gamma_30m,  'S_30m':  S_30m,
        'dphase_daily': dphase_daily, 'vel_daily':  vel_daily,
        'C_daily':      C_daily,      'g_daily':    gamma_daily,'S_daily':S_daily,
        'dphase_weekly':dphase_weekly,'vel_weekly': vel_weekly,
        'C_weekly':     C_weekly,     'g_weekly':   gamma_weekly,'S_weekly':S_weekly
    }, index=df_day.index)

    print("  Features built ✓")
    return feat

# ═══════════════════════════════════════════════════════════
# MAIN ENGINE
# ═══════════════════════════════════════════════════════════

def run_mtf_qlda(df_day, df_30m=None, initial_weights=None):
    if initial_weights is None:
        initial_weights = [0.3, 0.4, 0.3]

    feat   = build_features(df_day, df_30m)
    prices = df_day['close'].values
    n      = len(prices)

    next_ret = np.concatenate([np.diff(prices) / (prices[:-1] + EPS), [0.0]])
    prior_w  = np.array(initial_weights, dtype=float)

    weights_over_time = np.zeros((n, 3))
    comp = np.full(n, np.nan)
    D    = np.ones(n)
    E    = np.full(n, np.nan)
    size = np.zeros(n)

    # ATR proxy
    logrets    = np.concatenate([[0.0], np.diff(np.log(np.maximum(prices, EPS)))])
    atr        = pd.Series(np.abs(logrets)).rolling(window=14, min_periods=1).mean().values * prices
    median_atr = np.nanmedian(atr)

    # Initialize weights
    for t in range(min(len(prior_w), n)):
        weights_over_time[t] = prior_w

    # --- Weight optimization loop ---
    warmup = 120
    print(f"\nRunning weight optimization (warmup={warmup} days)...")

    for t in range(warmup, n):
        if t % 50 == 0:
            print(f"  Optimizing weights at day {t}/{n}...")
        start_opt = max(0, t - 90)
        alpha = 0.6

        feat_30    = np.sin(np.nan_to_num(feat['dphase_30m'].values[start_opt:t])) \
                   + alpha * np.nan_to_num(feat['vel_30m'].values[start_opt:t])
        feat_daily = np.sin(np.nan_to_num(feat['dphase_daily'].values[start_opt:t])) \
                   + alpha * np.nan_to_num(feat['vel_daily'].values[start_opt:t])
        feat_wkly  = np.sin(np.nan_to_num(feat['dphase_weekly'].values[start_opt:t])) \
                   + alpha * np.nan_to_num(feat['vel_weekly'].values[start_opt:t])

        feat_mat = np.vstack([feat_30, feat_daily, feat_wkly]).T
        y        = next_ret[start_opt:t]
        w_new    = optimize_weights(feat_mat, y, prior_w, reg=5.0)
        prior_w  = 0.8 * prior_w + 0.2 * w_new
        weights_over_time[t] = prior_w

    # --- Composite signal + Energy + Sizing ---
    print("\nComputing composite signals and sizing...")
    alpha = 0.6

    for t in range(n):
        w = weights_over_time[t]

        s30 = np.sin(np.nan_to_num(feat['dphase_30m'].values[t]))   \
            + alpha * np.nan_to_num(feat['vel_30m'].values[t])
        sd  = np.sin(np.nan_to_num(feat['dphase_daily'].values[t])) \
            + alpha * np.nan_to_num(feat['vel_daily'].values[t])
        sw  = np.sin(np.nan_to_num(feat['dphase_weekly'].values[t]))\
            + alpha * np.nan_to_num(feat['vel_weekly'].values[t])

        comp[t] = w[0]*s30 + w[1]*sd + w[2]*sw

        Cbar = w[0]*np.nan_to_num(feat['C_30m'].values[t])   \
             + w[1]*np.nan_to_num(feat['C_daily'].values[t]) \
             + w[2]*np.nan_to_num(feat['C_weekly'].values[t])
        gbar = w[0]*np.nan_to_num(feat['g_30m'].values[t])   \
             + w[1]*np.nan_to_num(feat['g_daily'].values[t]) \
             + w[2]*np.nan_to_num(feat['g_weekly'].values[t])
        Sbar = w[0]*np.nan_to_num(feat['S_30m'].values[t])   \
             + w[1]*np.nan_to_num(feat['S_daily'].values[t]) \
             + w[2]*np.nan_to_num(feat['S_weekly'].values[t])

        mu   = Cbar * gbar * Sbar * comp[t]
        E[t] = 100 * (mu ** 2)
        D[t] = 1.0 if comp[t] >= 0 else -1.0

        # Consensus tiers
        d30 = 1.0 if math.sin(np.nan_to_num(feat['dphase_30m'].values[t]))   >= 0 else -1.0
        dd  = 1.0 if math.sin(np.nan_to_num(feat['dphase_daily'].values[t])) >= 0 else -1.0
        dw  = 1.0 if math.sin(np.nan_to_num(feat['dphase_weekly'].values[t]))>= 0 else -1.0

        agree = int(d30 == np.sign(comp[t])) \
              + int(dd  == np.sign(comp[t])) \
              + int(dw  == np.sign(comp[t]))

        s = 1.0 if agree == 3 else (0.6 if agree == 2 else 0.2)

        E95   = np.nanpercentile(E[:t+1], 95) if t > 10 else 1.0
        E95   = max(E95, EPS)
        E_norm = E[t] / E95
        s      = s * min(1.0, E_norm * 2.0)
        s      = s / (1.0 + atr[t] / (median_atr + EPS))
        size[t] = s

    # --- Meta-model (Logistic Regression) ---
    print("\nTraining meta-model (Logistic Regression L2)...")
    X_rows, y_rows = [], []

    for t in range(warmup + 10, n - 1):
        row  = [math.sin(np.nan_to_num(feat['dphase_30m'].values[t])),
                np.nan_to_num(feat['vel_30m'].values[t]),
                np.nan_to_num(feat['C_30m'].values[t]),
                np.nan_to_num(feat['g_30m'].values[t]),
                np.nan_to_num(feat['S_30m'].values[t])]
        row += [math.sin(np.nan_to_num(feat['dphase_daily'].values[t])),
                np.nan_to_num(feat['vel_daily'].values[t]),
                np.nan_to_num(feat['C_daily'].values[t]),
                np.nan_to_num(feat['g_daily'].values[t]),
                np.nan_to_num(feat['S_daily'].values[t])]
        row += [math.sin(np.nan_to_num(feat['dphase_weekly'].values[t])),
                np.nan_to_num(feat['vel_weekly'].values[t]),
                np.nan_to_num(feat['C_weekly'].values[t]),
                np.nan_to_num(feat['g_weekly'].values[t]),
                np.nan_to_num(feat['S_weekly'].values[t])]
        row += [comp[t], E[t], size[t], atr[t] / (median_atr + EPS)]
        X_rows.append(row)
        y_rows.append(1 if next_ret[t] > 0 else 0)

    X_arr  = np.array(X_rows)
    y_arr  = np.array(y_rows)
    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(np.nan_to_num(X_arr))
    model  = LogisticRegression(C=1.0, penalty='l2', solver='liblinear')

    model_trained = False
    if len(X_sc) > 50:
        model.fit(X_sc, y_arr)
        model_trained = True
        print(f"  Meta-model trained on {len(X_sc)} samples ✓")
    else:
        print(f"  Not enough samples for meta-model ({len(X_sc)}), using composite only")

    # --- Trade simulation ---
    print("\nSimulating trades...")
    equity     = 10000.0
    equity_curve = []
    meta_probs = []
    prev_pos   = 0.0
    tc_rate    = 0.0004
    positions  = []
    signals    = []

    for t in range(n - 1):
        # Meta-model probability
        if model_trained and t >= warmup + 10:
            row  = [math.sin(np.nan_to_num(feat['dphase_30m'].values[t])),
                    np.nan_to_num(feat['vel_30m'].values[t]),
                    np.nan_to_num(feat['C_30m'].values[t]),
                    np.nan_to_num(feat['g_30m'].values[t]),
                    np.nan_to_num(feat['S_30m'].values[t])]
            row += [math.sin(np.nan_to_num(feat['dphase_daily'].values[t])),
                    np.nan_to_num(feat['vel_daily'].values[t]),
                    np.nan_to_num(feat['C_daily'].values[t]),
                    np.nan_to_num(feat['g_daily'].values[t]),
                    np.nan_to_num(feat['S_daily'].values[t])]
            row += [math.sin(np.nan_to_num(feat['dphase_weekly'].values[t])),
                    np.nan_to_num(feat['vel_weekly'].values[t]),
                    np.nan_to_num(feat['C_weekly'].values[t]),
                    np.nan_to_num(feat['g_weekly'].values[t]),
                    np.nan_to_num(feat['S_weekly'].values[t])]
            row += [comp[t], E[t], size[t], atr[t] / (median_atr + EPS)]
            x_t = np.array(row).reshape(1, -1)
            px  = model.predict_proba(scaler.transform(np.nan_to_num(x_t)))[0, 1]
            meta_probs.append(px)

            if   px > 0.55: use = size[t]
            elif px < 0.45: use = -size[t]
            else:            use = 0.0
        else:
            px  = 0.5
            meta_probs.append(px)
            use = size[t] * D[t]

        # Cap exposure
        use = np.clip(use, -1.0, 1.0)

        pnl      = equity * use * next_ret[t]
        turnover = abs(use - prev_pos)
        cost     = equity * turnover * tc_rate
        equity   = equity + pnl - cost

        equity_curve.append(equity)
        positions.append(use)
        signals.append('LONG' if use > 0.1 else ('SHORT' if use < -0.1 else 'FLAT'))
        prev_pos = use

    equity_curve.append(equity)
    meta_probs.append(0.5)
    positions.append(0.0)
    signals.append('FLAT')

    # --- Output DataFrame ---
    df_out = pd.DataFrame({
        'date':       df_day.index,
        'close':      prices,
        'comp':       np.round(comp, 6),
        'energy':     np.round(E, 4),
        'size':       np.round(size, 4),
        'direction':  D,
        'meta_prob':  np.round(meta_probs, 4),
        'position':   np.round(positions, 4),
        'signal':     signals,
        'w_30m':      np.round(weights_over_time[:, 0], 4),
        'w_daily':    np.round(weights_over_time[:, 1], 4),
        'w_weekly':   np.round(weights_over_time[:, 2], 4),
        'equity':     np.round(equity_curve, 2),
    })

    return df_out, weights_over_time, model if model_trained else None

# ═══════════════════════════════════════════════════════════
# PERFORMANCE REPORT
# ═══════════════════════════════════════════════════════════

def generate_report(df_out, output_path='mtf_qlda_report.txt'):
    equity = df_out['equity'].dropna().values
    returns = np.diff(equity) / (equity[:-1] + EPS)

    total_return   = (equity[-1] / equity[0] - 1) * 100
    ann_return     = ((equity[-1] / equity[0]) ** (252 / max(len(equity), 1)) - 1) * 100
    vol            = np.std(returns) * np.sqrt(252) * 100
    sharpe         = (np.mean(returns) / (np.std(returns) + EPS)) * np.sqrt(252)
    neg_ret        = returns[returns < 0]
    sortino        = (np.mean(returns) / (np.std(neg_ret) + EPS)) * np.sqrt(252)
    peak           = np.maximum.accumulate(equity)
    drawdown       = (equity - peak) / (peak + EPS)
    max_dd         = drawdown.min() * 100

    long_trades  = (df_out['signal'] == 'LONG').sum()
    short_trades = (df_out['signal'] == 'SHORT').sum()
    flat_trades  = (df_out['signal'] == 'FLAT').sum()

    # Hit rate
    df_tmp = df_out.copy()
    df_tmp['next_ret'] = df_tmp['close'].pct_change().shift(-1)
    correct = ((df_tmp['position'] > 0.1) & (df_tmp['next_ret'] > 0)) | \
              ((df_tmp['position'] < -0.1) & (df_tmp['next_ret'] < 0))
    active  = df_tmp['position'].abs() > 0.1
    hit_rate = correct[active].mean() * 100 if active.sum() > 0 else 0

    report = f"""
╔══════════════════════════════════════════════════════════════╗
║         MTF-Q-LDA+ PERFORMANCE REPORT                       ║
╠══════════════════════════════════════════════════════════════╣
║  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                       ║
║  Bars analyzed: {len(df_out):>6}                                    ║
╠══════════════════════════════════════════════════════════════╣
║  RETURNS                                                     ║
║  Total Return:        {total_return:>+8.2f}%                          ║
║  Annualized Return:   {ann_return:>+8.2f}%                          ║
║  Annualized Vol:      {vol:>8.2f}%                          ║
╠══════════════════════════════════════════════════════════════╣
║  RISK-ADJUSTED                                               ║
║  Sharpe Ratio:        {sharpe:>8.3f}                          ║
║  Sortino Ratio:       {sortino:>8.3f}                          ║
║  Max Drawdown:        {max_dd:>+8.2f}%                          ║
╠══════════════════════════════════════════════════════════════╣
║  SIGNALS                                                     ║
║  Hit Rate (active):   {hit_rate:>8.2f}%                          ║
║  LONG days:           {long_trades:>8}                          ║
║  SHORT days:          {short_trades:>8}                          ║
║  FLAT days:           {flat_trades:>8}                          ║
╠══════════════════════════════════════════════════════════════╣
║  FINAL EQUITY                                                ║
║  Start: $10,000   End: ${equity[-1]:>10,.2f}                  ║
╚══════════════════════════════════════════════════════════════╝

ALGORITHM: MTF-Q-LDA+
  - 3 Timeframes: 30m synthetic, Daily, Weekly
  - Hilbert instantaneous phase + velocity
  - Dynamic softmax weights (L-BFGS-B optimizer)
  - Logistic Regression meta-model (L2 regularized)
  - Consensus tiers (3/2/1 TF agreement)
  - Energy gating (coherence × kurtosis × trend × composite)
  - ATR position sizing
  - Transaction cost: 4bps per turnover

USAGE NOTES:
  - Warmup period: 120 days (signals before this are less reliable)
  - Refit meta-model every 30 days in production
  - Use walk-forward validation before live trading
  - This is NOT financial advice
"""

    with open(output_path, 'w') as f:
        f.write(report)

    print(report)
    return report

# ═══════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 60)
    print("  MTF-Q-LDA+  Multi-Timeframe Quantum-LDA Plus")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("\nUsage:  python mtf_qlda_plus.py data.csv")
        print("        python mtf_qlda_plus.py data.csv data_30m.csv")
        print("\nCSV format: date,open,high,low,close,volume")
        print("\nGenerating DEMO with synthetic BTC-like data...")

        # Generate synthetic demo data
        np.random.seed(42)
        n_days = 400
        dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
        price = 30000.0
        prices_demo = [price]
        for _ in range(n_days - 1):
            price *= (1 + np.random.normal(0.0003, 0.02))
            prices_demo.append(price)

        df_demo = pd.DataFrame({
            'date':   dates,
            'open':   [p * (1 + np.random.normal(0, 0.005)) for p in prices_demo],
            'high':   [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices_demo],
            'low':    [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices_demo],
            'close':  prices_demo,
            'volume': np.random.randint(1000, 50000, n_days),
        }).set_index('date')

        print(f"  Demo data: {n_days} days of synthetic prices")
        df_day  = df_demo
        df_30m  = None
    else:
        daily_path = sys.argv[1]
        print(f"\nLoading daily data: {daily_path}")
        df_day = pd.read_csv(daily_path, parse_dates=['date']).set_index('date')
        print(f"  Loaded {len(df_day)} rows | {df_day.index[0]} → {df_day.index[-1]}")

        df_30m = None
        if len(sys.argv) >= 3:
            m30_path = sys.argv[2]
            print(f"Loading 30m data: {m30_path}")
            df_30m = pd.read_csv(m30_path, parse_dates=['date']).set_index('date')
            print(f"  Loaded {len(df_30m)} rows")

    # Run
    df_out, weights, model = run_mtf_qlda(df_day, df_30m)

    # Save output
    out_csv = 'mtf_qlda_output.csv'
    df_out.to_csv(out_csv, index=False)
    print(f"\n✅ Output saved: {out_csv}")

    # Report
    generate_report(df_out, 'mtf_qlda_report.txt')
    print("✅ Report saved: mtf_qlda_report.txt")

    # Show last 10 signals
    print("\n--- LAST 10 SIGNALS ---")
    cols = ['date','close','comp','meta_prob','signal','position','energy']
    print(df_out[cols].tail(10).to_string(index=False))

    print("\n✅ MTF-Q-LDA+ complete!")
