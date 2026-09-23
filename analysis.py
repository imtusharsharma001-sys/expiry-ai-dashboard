import math

import numpy as np
import pandas as pd


def calculate_weekly_outlook(daily_close, latest_price, horizon_sessions=5):
    """Estimate up/down/sideways probabilities over a selected session horizon."""
    prices = pd.Series(daily_close, dtype=float).dropna()
    horizon = max(1, int(horizon_sessions))
    if len(prices) < 22 or latest_price <= 0:
        return None
    log_returns = np.log(prices / prices.shift(1)).dropna()
    daily_sigma = float(log_returns.tail(60).std())
    if not np.isfinite(daily_sigma) or daily_sigma <= 0:
        daily_sigma = 0.01 / np.sqrt(5)
    sigma = max(daily_sigma * np.sqrt(horizon), 0.002 * np.sqrt(horizon / 5))
    momentum_5 = float(prices.iloc[-1] / prices.iloc[-6] - 1)
    momentum_20 = float(prices.iloc[-1] / prices.iloc[-21] - 1)
    scale = np.sqrt(horizon / 5)
    drift = float(np.clip((0.6 * momentum_5 + 0.25 * momentum_20) * scale, -0.75 * sigma, 0.75 * sigma))

    def normal_cdf(z):
        return 0.5 * (1 + math.erf(z / np.sqrt(2)))

    up_threshold, down_threshold = np.log(1.005), np.log(0.995)
    up = 1 - normal_cdf((up_threshold - drift) / sigma)
    down = normal_cdf((down_threshold - drift) / sigma)
    sideways = max(0.0, 1 - up - down)
    total = up + down + sideways
    up, down, sideways = (100 * p / total for p in (up, down, sideways))
    return {
        "horizon_sessions": horizon,
        "up_pct": up,
        "down_pct": down,
        "sideways_pct": sideways,
        "expected_price": latest_price * math.exp(drift),
        "low": latest_price * math.exp(drift - sigma),
        "high": latest_price * math.exp(drift + sigma),
        "momentum_5": momentum_5,
        "momentum_20": momentum_20,
    }


def calculate_technical_indicators(ohlc):
    """Return common trend, momentum, volatility, volume and level indicators."""
    if ohlc is None or ohlc.empty or "Close" not in ohlc:
        return {"factors": [], "score": 0, "bias": "🟡 RANGE / WAIT"}
    close = ohlc["Close"].astype(float).dropna()
    if close.empty:
        return {"factors": [], "score": 0, "bias": "🟡 RANGE / WAIT"}
    high = ohlc.get("High", close).reindex(close.index).astype(float)
    low = ohlc.get("Low", close).reindex(close.index).astype(float)
    volume = ohlc.get("Volume", pd.Series(index=close.index, dtype=float)).reindex(close.index).astype(float)
    last = float(close.iloc[-1])
    factors, votes = [], []

    def add(name, value, reading, vote=None):
        factors.append((name, reading if reading else "Unavailable"))
        if vote is not None:
            votes.append(int(vote))

    for span in (9, 21, 50):
        ema = close.ewm(span=span, adjust=False).mean().iloc[-1]
        add(f"Price vs EMA {span}", ema, f"{'Above' if last >= ema else 'Below'} · {ema:,.2f}", 1 if last >= ema else -1)
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    if np.isfinite(sma20):
        add("SMA 20", sma20, f"{sma20:,.2f}", 1 if last >= sma20 else -1)
    if np.isfinite(sma50):
        add("SMA 50", sma50, f"{sma50:,.2f}", 1 if last >= sma50 else -1)

    delta = close.diff()
    gain, loss = delta.clip(lower=0).rolling(14).mean(), (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = float((100 - 100 / (1 + rs)).iloc[-1]) if len(close) >= 15 and np.isfinite(rs.iloc[-1]) else 50.0
    add("RSI 14", rsi, f"{rsi:.1f}", 1 if 50 <= rsi <= 70 else (-1 if rsi < 45 or rsi > 75 else 0))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    macd_up = float(macd.iloc[-1]) >= float(signal.iloc[-1])
    add("MACD (12, 26, 9)", macd.iloc[-1], f"{'Bullish' if macd_up else 'Bearish'} · {macd.iloc[-1]:.2f}", 1 if macd_up else -1)

    mid = close.rolling(20).mean()
    std = close.rolling(20).std()
    upper, lower = mid + 2 * std, mid - 2 * std
    if len(close) >= 20 and np.isfinite(mid.iloc[-1]) and float(upper.iloc[-1] - lower.iloc[-1]) > 0:
        pct_b = (last - float(lower.iloc[-1])) / float(upper.iloc[-1] - lower.iloc[-1])
        add("Bollinger %B (20, 2)", pct_b, f"{pct_b:.2f}", 1 if pct_b >= 0.5 else -1)

    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().iloc[-1]) if len(close) >= 14 else float(tr.mean())
    if np.isfinite(atr):
        add("ATR 14 / price", atr, f"{atr:,.2f} · {100 * atr / last:.2f}%")
    if len(close) >= 14:
        plus_dm = high.diff().where((high.diff() > -low.diff()) & (high.diff() > 0), 0.0)
        minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (-low.diff() > 0), 0.0)
        atr_smooth = tr.rolling(14).mean().replace(0, np.nan)
        plus_di = 100 * plus_dm.rolling(14).mean() / atr_smooth
        minus_di = 100 * minus_dm.rolling(14).mean() / atr_smooth
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
        adx = float(dx.rolling(14).mean().iloc[-1])
        if np.isfinite(adx):
            add("ADX 14", adx, f"{adx:.1f} · {'trending' if adx >= 20 else 'weak trend'}")

    if len(close) >= 14 and float(high.rolling(14).max().iloc[-1] - low.rolling(14).min().iloc[-1]) > 0:
        lo14, hi14 = float(low.rolling(14).min().iloc[-1]), float(high.rolling(14).max().iloc[-1])
        stoch = 100 * (last - lo14) / (hi14 - lo14)
        add("Stochastic %K (14)", stoch, f"{stoch:.1f}", 1 if 20 <= stoch <= 80 and stoch >= 50 else (-1 if stoch < 50 else 0))
    support, resistance = float(low.tail(20).min()), float(high.tail(20).max())
    add("20-candle support / resistance", support, f"{support:,.2f} / {resistance:,.2f}")
    if len(volume) >= 20 and float(volume.tail(20).mean()) > 0 and float(volume.iloc[-1]) > 0:
        ratio = float(volume.iloc[-1] / volume.tail(20).mean())
        add("Volume vs 20-candle average", ratio, f"{ratio:.2f}×", 1 if ratio >= 1.2 and last >= float(close.iloc[-2]) else (-1 if ratio >= 1.2 else 0))
    if float(volume.fillna(0).sum()) > 0:
        typical = (high + low + close) / 3
        vwap = float((typical * volume).sum() / volume.sum())
        add("Session VWAP", vwap, f"{'Above' if last >= vwap else 'Below'} · {vwap:,.2f}", 1 if last >= vwap else -1)

    score = sum(votes)
    bias = "🟢 BULLISH" if score >= 3 else ("🔴 BEARISH" if score <= -3 else "🟡 RANGE / WAIT")
    return {"factors": factors, "score": score, "bias": bias}


def calculate_factor_backtest(ohlc, max_checks=200, min_history=60):
    """Walk-forward hit-rate for the technical-factor score, with abstention stats."""
    if ohlc is None or ohlc.empty or "Close" not in ohlc:
        return None
    data = ohlc.dropna(subset=["Close"])
    if len(data) <= min_history:
        return None
    first = max(min_history, len(data) - max(1, int(max_checks)))
    up_correct = down_correct = up_signals = down_signals = signals = checks = 0
    actual_up_count = actual_down_count = 0
    actual_outcomes, predictions = [], []
    for next_i in range(first, len(data)):
        before = data.iloc[:next_i]
        movement = float(data["Close"].iloc[next_i] / data["Close"].iloc[next_i - 1] - 1)
        if abs(movement) < 1e-12:
            continue
        actual_up = movement > 0
        actual_up_count += actual_up
        actual_down_count += not actual_up
        checks += 1
        result = calculate_technical_indicators(before)
        score = result["score"]
        if score >= 3:
            predictions.append(1)
            up_signals += 1
            signals += 1
            up_correct += actual_up
        elif score <= -3:
            predictions.append(-1)
            down_signals += 1
            signals += 1
            down_correct += not actual_up
        else:
            predictions.append(0)
        actual_outcomes.append(1 if actual_up else -1)
    if not checks:
        return None
    correct = up_correct + down_correct
    hit_rate = 100 * correct / signals if signals else None
    baseline = 100 * max(actual_up_count, actual_down_count) / checks
    # Moving-block bootstrap keeps nearby candle outcomes together instead of
    # pretending every consecutive candle is an independent observation.
    if signals:
        rng = np.random.default_rng(2718)
        actual_array, prediction_array = np.asarray(actual_outcomes), np.asarray(predictions)
        block = min(5, checks)
        blocks_per_draw = int(np.ceil(checks / block))
        bootstrap_hit, bootstrap_edge = [], []
        offsets = np.arange(block)
        for _ in range(500):
            starts = rng.integers(0, checks, size=blocks_per_draw)
            indices = np.concatenate([(start + offsets) % checks for start in starts])[:checks]
            actual_sample, prediction_sample = actual_array[indices], prediction_array[indices]
            selected = prediction_sample != 0
            if not selected.any():
                continue
            hit = float(np.mean(prediction_sample[selected] == actual_sample[selected]))
            sample_baseline = max(float(np.mean(actual_sample == 1)), float(np.mean(actual_sample == -1)))
            bootstrap_hit.append(100 * hit)
            bootstrap_edge.append(100 * (hit - sample_baseline))
        ci_low, ci_high = np.percentile(bootstrap_hit, [2.5, 97.5]) if bootstrap_hit else (None, None)
        edge_ci_low = float(np.percentile(bootstrap_edge, 5)) if bootstrap_edge else None
    else:
        ci_low = ci_high = edge_ci_low = None
    up_hit = 100 * up_correct / up_signals if up_signals else None
    down_hit = 100 * down_correct / down_signals if down_signals else None
    return {
        "checks": checks,
        "signals": signals,
        "coverage_pct": 100 * signals / checks,
        "hit_rate_pct": hit_rate,
        "ci_low_pct": ci_low,
        "ci_high_pct": ci_high,
        "edge_ci_low_pct": edge_ci_low,
        "up_signals": up_signals,
        "up_hit_pct": up_hit,
        "down_signals": down_signals,
        "down_hit_pct": down_hit,
        "baseline_pct": baseline,
    }


def calculate_prediction_outlook(close_values, max_backtests=200, factor_score=0):
    """Estimate next-candle up/down probabilities from momentum and factor score."""
    close = pd.Series(close_values, dtype=float).dropna()
    if len(close) < 10:
        return None

    def forecast(prices):
        returns = prices.pct_change().dropna()
        fast = float(prices.iloc[-1] / prices.iloc[-2] - 1) if len(prices) > 1 else 0.0
        slow = float(prices.iloc[-1] / prices.iloc[-7] - 1) if len(prices) > 6 else fast
        sigma = float(returns.tail(60).std()) if len(returns) > 1 else 0.0
        if not np.isfinite(sigma) or sigma < 1e-6:
            sigma = 0.001
        drift = float(np.clip(0.7 * fast + 0.3 * slow, -sigma, sigma))
        return drift, sigma

    drift, sigma = forecast(close)
    drift = float(np.clip(drift + np.clip(float(factor_score), -8, 8) * sigma * 0.08, -sigma, sigma))
    up_probability = 100 * (0.5 * (1 + math.erf(drift / (sigma * np.sqrt(2)))))
    down_probability = 100 - up_probability

    first = max(7, len(close) - max(1, int(max_backtests)))
    correct, samples = 0, 0
    for next_i in range(first, len(close)):
        history = close.iloc[:next_i]
        historical_drift, _ = forecast(history)
        actual_move = float(close.iloc[next_i] / close.iloc[next_i - 1] - 1)
        if abs(actual_move) < 1e-12:
            continue
        predicted_up = historical_drift >= 0
        actual_up = actual_move > 0
        correct += predicted_up == actual_up
        samples += 1
    return {
        "up_pct": up_probability,
        "down_pct": down_probability,
        "accuracy_pct": 100 * correct / samples if samples else None,
        "sample_count": samples,
    }


def analyze_option_chain(chain, spot):
    """Summarize an uploaded option-chain table without fetching/scraping exchange data."""
    if chain is None or chain.empty:
        return None
    def find_column(contains, excluded=()):
        for column in chain.columns:
            name = " ".join(str(column).lower().replace("_", " ").split())
            if all(token in name for token in contains) and not any(token in name for token in excluded):
                return column
        return None
    strike_col = find_column(("strike",))
    call_oi_col = find_column(("oi",), ("put", "pe"))
    put_oi_col = find_column(("oi",), ("call", "ce"))
    # Also accept the compact template names: strike, call_oi, put_oi.
    if strike_col is None:
        strike_col = next((c for c in chain.columns if str(c).strip().lower() in {"strike", "strike price"}), None)
    if call_oi_col is None:
        call_oi_col = next((c for c in chain.columns if str(c).strip().lower() in {"call_oi", "ce_oi", "call oi", "ce oi"}), None)
    if put_oi_col is None:
        put_oi_col = next((c for c in chain.columns if str(c).strip().lower() in {"put_oi", "pe_oi", "put oi", "pe oi"}), None)
    if not all((strike_col, call_oi_col, put_oi_col)):
        return {"error": "Required columns: strike, call_oi, put_oi (or exchange columns containing Strike, CALLS OI, PUTS OI)."}
    data = pd.DataFrame({
        "strike": pd.to_numeric(chain[strike_col].astype(str).str.replace(",", "", regex=False), errors="coerce"),
        "call_oi": pd.to_numeric(chain[call_oi_col].astype(str).str.replace(",", "", regex=False), errors="coerce"),
        "put_oi": pd.to_numeric(chain[put_oi_col].astype(str).str.replace(",", "", regex=False), errors="coerce"),
    }).dropna()
    data = data[(data["strike"] > 0) & (data["call_oi"] >= 0) & (data["put_oi"] >= 0)]
    if data.empty or data["call_oi"].sum() <= 0:
        return {"error": "Option-chain rows have no valid strikes or open interest."}
    call_wall = float(data.loc[data["call_oi"].idxmax(), "strike"])
    put_wall = float(data.loc[data["put_oi"].idxmax(), "strike"])
    pcr = float(data["put_oi"].sum() / data["call_oi"].sum())
    settlements = data["strike"].to_numpy()
    strikes = data["strike"].to_numpy()
    call_oi = data["call_oi"].to_numpy()
    put_oi = data["put_oi"].to_numpy()
    payouts = (np.maximum(settlements[:, None] - strikes[None, :], 0) * call_oi[None, :]
               + np.maximum(strikes[None, :] - settlements[:, None], 0) * put_oi[None, :]).sum(axis=1)
    max_pain = float(settlements[int(np.argmin(payouts))])
    return {"call_wall": call_wall, "put_wall": put_wall, "pcr": pcr, "max_pain": max_pain, "rows": len(data)}


def calculate_option_greeks(spot, strike, days_to_expiry, volatility, option_type, risk_free_rate=0.06):
    """Black-Scholes model Greeks per index unit; theta is per calendar day."""
    spot, strike = float(spot), float(strike)
    days, volatility = float(days_to_expiry), float(volatility)
    if spot <= 0 or strike <= 0 or days <= 0 or volatility <= 0:
        raise ValueError("Spot, strike, days and volatility must be positive.")
    t = days / 365
    root_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (risk_free_rate + volatility ** 2 / 2) * t) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    cdf = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    gamma = pdf / (spot * volatility * root_t)
    theta_common = -(spot * pdf * volatility) / (2 * root_t)
    if option_type.upper() in {"CE", "CALL"}:
        delta = cdf(d1)
        theta = theta_common - risk_free_rate * strike * math.exp(-risk_free_rate * t) * cdf(d2)
    elif option_type.upper() in {"PE", "PUT"}:
        delta = cdf(d1) - 1
        theta = theta_common + risk_free_rate * strike * math.exp(-risk_free_rate * t) * cdf(-d2)
    else:
        raise ValueError("option_type must be CE or PE")
    return {"delta": delta, "gamma": gamma, "theta_per_day": theta / 365}


def calculate_market_levels(prediction_ohlc, daily_ohlc, spot):
    """Build transparent ATR, recent swing and classic pivot levels for a paper plan."""
    if prediction_ohlc is None or prediction_ohlc.empty or "Close" not in prediction_ohlc:
        return None
    close = prediction_ohlc["Close"].astype(float).dropna()
    high = prediction_ohlc.get("High", close).reindex(close.index).astype(float)
    low = prediction_ohlc.get("Low", close).reindex(close.index).astype(float)
    previous = close.shift(1)
    true_range = pd.concat([(high - low).abs(), (high - previous).abs(), (low - previous).abs()], axis=1).max(axis=1)
    atr = float(true_range.rolling(14, min_periods=1).mean().iloc[-1])
    result = {"atr": atr, "recent_support": float(low.tail(20).min()), "recent_resistance": float(high.tail(20).max())}
    if daily_ohlc is not None and not daily_ohlc.empty and {"High", "Low", "Close"}.issubset(daily_ohlc.columns):
        row = daily_ohlc.iloc[-2] if len(daily_ohlc) > 1 else daily_ohlc.iloc[-1]
        h, l, c = float(row["High"]), float(row["Low"]), float(row["Close"])
        pivot = (h + l + c) / 3
        result.update({"pivot": pivot, "s1": 2 * pivot - h, "s2": pivot - (h - l), "r1": 2 * pivot - l, "r2": pivot + (h - l), "previous_high": h, "previous_low": l})
    else:
        pivot = float(close.tail(20).mean())
        result.update({"pivot": pivot, "s1": pivot - atr, "s2": pivot - 2 * atr, "r1": pivot + atr, "r2": pivot + 2 * atr, "previous_high": result["recent_resistance"], "previous_low": result["recent_support"]})
    result["bullish_trigger"] = max(result["pivot"], result["previous_high"])
    result["bearish_trigger"] = min(result["pivot"], result["previous_low"])
    result["bullish_stop"] = min(result["recent_support"], float(spot) - 1.5 * atr)
    result["bearish_stop"] = max(result["recent_resistance"], float(spot) + 1.5 * atr)
    result["bullish_targets"] = [max(result["r1"], float(spot) + atr), max(result["r2"], float(spot) + 2 * atr)]
    result["bearish_targets"] = [min(result["s1"], float(spot) - atr), min(result["s2"], float(spot) - 2 * atr)]
    return result


def black_scholes_price(spot, strike, days_to_expiry, volatility, option_type, risk_free_rate=0.06):
    """Theoretical European option value; does not represent a live market quote."""
    spot, strike = float(spot), float(strike)
    days, volatility = float(days_to_expiry), float(volatility)
    if spot <= 0 or strike <= 0 or days <= 0 or volatility <= 0:
        raise ValueError("Spot, strike, days and volatility must be positive.")
    t = days / 365.0
    root_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * t) / (volatility * root_t)
    d2 = d1 - volatility * root_t
    cdf = lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    call = spot * cdf(d1) - strike * math.exp(-risk_free_rate * t) * cdf(d2)
    if option_type.upper() in {"CE", "CALL"}:
        return max(0.0, call)
    if option_type.upper() in {"PE", "PUT"}:
        return max(0.0, call - spot + strike * math.exp(-risk_free_rate * t))
    raise ValueError("option_type must be CE or PE")
