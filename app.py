import hmac
import os
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from analysis import analyze_option_chain, black_scholes_price, calculate_factor_backtest, calculate_market_levels, calculate_option_greeks, calculate_prediction_outlook, calculate_technical_indicators, calculate_weekly_outlook

st.set_page_config(page_title="Expiry AI", page_icon="📈", layout="wide")

def read_setting(name):
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except st.errors.StreamlitSecretNotFoundError:
        return ""

# Optional shared-site login. Set DASHBOARD_PASSWORD on the server to enable it.
dashboard_password = read_setting("DASHBOARD_PASSWORD").strip()
if dashboard_password and not st.session_state.get("dashboard_authenticated", False):
    st.title("📈 EXPIRY AI")
    with st.form("dashboard_login"):
        entered_password = st.text_input("Dashboard password", type="password")
        submitted = st.form_submit_button("Open dashboard")
    if submitted:
        if hmac.compare_digest(entered_password, dashboard_password):
            st.session_state["dashboard_authenticated"] = True
            st.rerun()
        st.error("Password sahi nahi hai.")
    st.stop()

cache_dir = Path(__file__).resolve().parent / ".cache"
cache_dir.mkdir(parents=True, exist_ok=True)
yf.set_tz_cache_location(str(cache_dir))
st.title("📈 EXPIRY AI")
st.caption("NIFTY • BANKNIFTY • SENSEX | Paper-analysis dashboard")

with st.sidebar:
    st.subheader("🤖 AI summary")
    server_api_key = read_setting("OPENAI_API_KEY")
    if server_api_key:
        api_key = server_api_key
        st.caption("AI key server par securely set hai.")
    else:
        api_key = st.text_input("OpenAI key", type="password", help="Local testing ke liye. Public website par key server environment mein set karein.")

st.warning("PAPER MODE ONLY — This version does not place real orders. It is designed to help analyse the market and test a process before risking money.")

@st.cache_data(ttl=15)
def get_data(symbol, period="5d", interval="5m"):
    try:
        df = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [column[0] for column in df.columns]
        return df.sort_index().dropna()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=900)
def get_market_news(query, freshness="recent-7-days-v1"):
    url = "https://news.google.com/rss/search"
    try:
        response = requests.get(
            url,
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            headers={"User-Agent": "ExpiryAI-PaperDashboard/1.0"},
            timeout=12,
        )
        response.raise_for_status()
        root = ET.fromstring(response.content)
        items = []
        for item in root.findall("./channel/item"):
            source_node = item.find("source")
            published = item.findtext("pubDate", "")
            try:
                published_at = parsedate_to_datetime(published)
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - published_at > timedelta(days=7):
                    continue
            except (TypeError, ValueError, OverflowError):
                continue
            items.append({
                "title": item.findtext("title", "Market headline"),
                "link": item.findtext("link", ""),
                "source": source_node.text if source_node is not None and source_node.text else "News source",
                "published": published,
            })
            if len(items) == 8:
                break
        return items
    except (requests.RequestException, ET.ParseError, ValueError):
        return []

@st.cache_data(ttl=60)
def get_factor_backtest(data):
    return calculate_factor_backtest(data)


symbols = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK", "SENSEX": "^BSESN"}
timeframes = {
    "1 minute": {"interval": "1m", "period": "1d"},
    "5 minutes": {"interval": "5m", "period": "5d"},
    "15 minutes": {"interval": "15m", "period": "1mo"},
    "30 minutes": {"interval": "30m", "period": "1mo"},
    "1 hour": {"interval": "60m", "period": "3mo"},
    "1 day": {"interval": "1d", "period": "1y"},
}

@st.fragment(run_every="15s")
def render_dashboard():
    choice = st.selectbox("Select index", list(symbols))
    chart_timeframe = st.selectbox("Candle chart timeframe", list(timeframes), index=1)
    prediction_timeframe = st.selectbox("Prediction timeframe", list(timeframes), index=1)
    outlook_choices = {"1 day (1 session)": 1, "3 days (3 sessions)": 3, "1 week (5 sessions)": 5, "2 weeks (10 sessions)": 10, "1 month (20 sessions)": 20}
    outlook_label = st.selectbox("Direction probability timeframe", list(outlook_choices), index=2)
    refresh = st.button("🔄 START / REFRESH ANALYSIS")
    with st.expander("Optional broker data · VIX and futures OI"):
        ctx1, ctx2, ctx3 = st.columns(3)
        india_vix_text = ctx1.text_input("India VIX (optional)", placeholder="e.g. 14.2", key="context_vix")
        futures_price_text = ctx2.text_input("Futures price change %", placeholder="e.g. +0.4", key="context_fut_price")
        futures_oi_text = ctx3.text_input("Futures OI change %", placeholder="e.g. +2.1", key="context_fut_oi")
        lead_names = ("HDFC Bank", "ICICI Bank") if choice == "BANKNIFTY" else (("HDFC Bank", "Reliance") if choice == "NIFTY" else ("HDFC Bank", "Reliance"))
        ctx4, ctx5, ctx6 = st.columns(3)
        lead_a_text = ctx4.text_input(f"{lead_names[0]} change %", placeholder="optional", key="context_lead_a")
        lead_b_text = ctx5.text_input(f"{lead_names[1]} change %", placeholder="optional", key="context_lead_b")
        gift_change_text = ctx6.text_input("GIFT Nifty change %", placeholder="optional", key="context_gift")
    def parse_optional_number(value):
        try:
            parsed = float(value.strip().replace("%", "").replace(",", ""))
            return parsed if np.isfinite(parsed) else None
        except (AttributeError, ValueError):
            return None
    india_vix = parse_optional_number(india_vix_text)
    futures_price_change = parse_optional_number(futures_price_text)
    futures_oi_change = parse_optional_number(futures_oi_text)
    lead_a_change, lead_b_change = parse_optional_number(lead_a_text), parse_optional_number(lead_b_text)
    gift_change = parse_optional_number(gift_change_text)
    constituent_vote = 1 if lead_a_change is not None and lead_b_change is not None and lead_a_change > 0 and lead_b_change > 0 else (-1 if lead_a_change is not None and lead_b_change is not None and lead_a_change < 0 and lead_b_change < 0 else 0)
    gift_vote = 1 if gift_change is not None and gift_change > 0 else (-1 if gift_change is not None and gift_change < 0 else 0)
    futures_build = "Not supplied"
    futures_vote = 0
    if futures_price_change is not None and futures_oi_change is not None:
        if futures_price_change > 0 and futures_oi_change > 0:
            futures_build, futures_vote = "Long buildup", 1
        elif futures_price_change < 0 and futures_oi_change > 0:
            futures_build, futures_vote = "Short buildup", -1
        elif futures_price_change > 0 and futures_oi_change < 0:
            futures_build = "Short covering"
        elif futures_price_change < 0 and futures_oi_change < 0:
            futures_build = "Long unwinding"
        else:
            futures_build = "Mixed / flat"
    if refresh:
        get_data.clear()

    symbol = symbols[choice]
    chart_tf, pred_tf = timeframes[chart_timeframe], timeframes[prediction_timeframe]
    chart_df = get_data(symbol, period=chart_tf["period"], interval=chart_tf["interval"])
    prediction_df = get_data(symbol, period=pred_tf["period"], interval=pred_tf["interval"])
    quote_df = get_data(symbol, period="1d", interval="1m")
    weekly_df = get_data(symbol, period="1y", interval="1d")
    required = {"Open", "High", "Low", "Close"}
    if chart_df.empty or not required.issubset(chart_df.columns):
        st.error(f"Yahoo Finance candle data for {choice} at {chart_timeframe} is unavailable. Try another timeframe or Refresh.")
        return
    if prediction_df.empty or "Close" not in prediction_df:
        st.error(f"Yahoo Finance prediction data for {choice} at {prediction_timeframe} is unavailable. Try another timeframe or Refresh.")
        return

    close = prediction_df["Close"].astype(float).dropna()
    if close.empty:
        st.error("No usable prediction prices were returned. Try Refresh.")
        return
    if not quote_df.empty and "Close" in quote_df and not quote_df["Close"].dropna().empty:
        quote_close = quote_df["Close"].dropna()
        last, price_time = float(quote_close.iloc[-1]), quote_close.index[-1]
    else:
        last, price_time = float(close.iloc[-1]), close.index[-1]

    pred_last = float(close.iloc[-1])
    ret_fast = pred_last / float(close.iloc[-2]) - 1 if len(close) > 1 else 0.0
    ret_slow = pred_last / float(close.iloc[-7]) - 1 if len(close) > 6 else ret_fast
    technical = calculate_technical_indicators(prediction_df)
    factor_backtest = get_factor_backtest(prediction_df)
    combined_score = technical["score"] + futures_vote + constituent_vote + gift_vote
    factor_bias = "🟢 BULLISH" if combined_score >= 3 else ("🔴 BEARISH" if combined_score <= -3 else "🟡 RANGE / WAIT")
    validation_ready = bool(factor_backtest and factor_backtest["signals"] >= 30 and factor_backtest["edge_ci_low_pct"] is not None and factor_backtest["edge_ci_low_pct"] > 0)
    factor_readings = dict(technical["factors"])
    rsi = float(factor_readings.get("RSI 14", "50").split()[0])
    ema9 = float(pred_last)
    ema21 = float(pred_last)

    returns = close.pct_change().dropna()
    vol = float(returns.tail(60).std()) if len(returns) > 1 else 0.002
    if not np.isfinite(vol):
        vol = 0.002
    bars_per_session = {"1m": 375, "5m": 75, "15m": 25, "30m": 13, "60m": 6.25, "1d": 1}
    vix_bar_sigma = (india_vix / 100) / np.sqrt(252 * bars_per_session[pred_tf["interval"]]) if india_vix is not None and india_vix > 0 else 0.0
    expected_pct = max(0.001, vol, vix_bar_sigma)
    drift = float(np.clip(0.7 * ret_fast + 0.3 * ret_slow + np.clip(combined_score, -8, 8) * expected_pct * 0.08, -expected_pct, expected_pct))
    predicted_price = last * (1 + drift)
    low, high = predicted_price * (1 - expected_pct), predicted_price * (1 + expected_pct)
    weekly = calculate_weekly_outlook(weekly_df["Close"] if not weekly_df.empty and "Close" in weekly_df else pd.Series(dtype=float), last, outlook_choices[outlook_label])
    prediction_outlook = calculate_prediction_outlook(close, factor_score=combined_score)
    if prediction_outlook and prediction_outlook["up_pct"] >= 55:
        bias = "🟢 BULLISH"
    elif prediction_outlook and prediction_outlook["down_pct"] >= 55:
        bias = "🔴 BEARISH"
    else:
        bias = "🟡 RANGE / WAIT"
    market_levels = calculate_market_levels(prediction_df, weekly_df, last)
    vix_multiplier = 1.5 if india_vix is not None and india_vix >= 20 else (1.25 if india_vix is not None and india_vix >= 18 else (0.9 if india_vix is not None and india_vix <= 14 else 1.0))
    if market_levels:
        market_levels["bullish_stop"] = min(market_levels["recent_support"], last - 1.5 * market_levels["atr"] * vix_multiplier)
        market_levels["bearish_stop"] = max(market_levels["recent_resistance"], last + 1.5 * market_levels["atr"] * vix_multiplier)
        market_levels["bullish_targets"] = [max(market_levels["r1"], last + market_levels["atr"] * vix_multiplier), max(market_levels["r2"], last + 2 * market_levels["atr"] * vix_multiplier)]
        market_levels["bearish_targets"] = [min(market_levels["s1"], last - market_levels["atr"] * vix_multiplier), min(market_levels["s2"], last - 2 * market_levels["atr"] * vix_multiplier)]
    with st.expander("Add current option-chain data for OI/PCR levels"):
        st.caption("Upload a fresh CSV from your broker or exchange. No chain values are guessed or scraped.")
        template_csv = pd.DataFrame(columns=["strike", "call_oi", "put_oi"]).to_csv(index=False).encode("utf-8")
        st.download_button("Download option-chain CSV template", template_csv, file_name="ExpiryAI_option_chain_template.csv", mime="text/csv", key="option_chain_template")
        uploaded_chain = st.file_uploader("Upload option-chain CSV", type=["csv"], key="option_chain_csv")
    option_chain_summary = None
    if uploaded_chain is not None:
        try:
            option_chain_summary = analyze_option_chain(pd.read_csv(uploaded_chain), last)
        except (ValueError, KeyError, pd.errors.ParserError, UnicodeDecodeError):
            option_chain_summary = {"error": "Could not read CSV. Use the template columns: strike, call_oi, put_oi."}
    if option_chain_summary and "error" in option_chain_summary:
        st.warning(option_chain_summary["error"])
    elif option_chain_summary:
        oi1, oi2, oi3, oi4 = st.columns(4)
        oi1.metric("Put / Call OI ratio", f"{option_chain_summary['pcr']:.2f}")
        oi2.metric("Highest Call OI · resistance zone", f"{option_chain_summary['call_wall']:,.0f}")
        oi3.metric("Highest Put OI · support zone", f"{option_chain_summary['put_wall']:,.0f}")
        oi4.metric("Estimated max-pain strike", f"{option_chain_summary['max_pain']:,.0f}")
        st.caption(f"OI calculated from {option_chain_summary['rows']} uploaded strikes. PCR/max pain are context, not standalone trade signals.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Last price", f"{last:,.2f}", help=f"Latest available Yahoo 1-minute quote; timestamp: {price_time}")
    c2.metric("Model direction", bias, help="Direction follows the UP/DOWN chance estimate. The separate trade filter below checks historical factor performance.")
    c3.metric(f"Predicted next {prediction_timeframe} close", f"{predicted_price:,.2f}")
    c4.metric(f"Expected {prediction_timeframe} range", f"{low:,.2f} – {high:,.2f}", help="Volatility-based estimate, not a guaranteed target.")
    p1, p2, p3, p4 = st.columns(4)
    if prediction_outlook:
        p1.metric(f"Estimated UP chance · next {prediction_timeframe}", f"{prediction_outlook['up_pct']:.1f}%")
        p2.metric(f"Estimated DOWN chance · next {prediction_timeframe}", f"{prediction_outlook['down_pct']:.1f}%")
        accuracy_value = factor_backtest["hit_rate_pct"] if factor_backtest else None
        fallback_hit_label = "No factor signals" if factor_backtest and factor_backtest["signals"] == 0 else "Need more data"
        p3.metric(f"Past factor hit-rate · {prediction_timeframe}", f"{accuracy_value:.1f}%" if accuracy_value is not None else fallback_hit_label, help=f"Walk-forward checks: {factor_backtest['signals'] if factor_backtest else 0} factor signals from {factor_backtest['checks'] if factor_backtest else 0} prior candles. It is a historical result, not a future accuracy guarantee.")
        p4.metric("Paper trade filter", "VALIDATED" if validation_ready else "WAIT", help="WAIT until at least 30 historical factor signals exist and the 95% block-bootstrap lower-bound edge exceeds the majority-direction baseline.")
        if factor_backtest and accuracy_value is not None:
            up_hit_text = f"{factor_backtest['up_hit_pct']:.1f}%" if factor_backtest["up_hit_pct"] is not None else "n/a"
            down_hit_text = f"{factor_backtest['down_hit_pct']:.1f}%" if factor_backtest["down_hit_pct"] is not None else "n/a"
            st.caption(f"Selected-timeframe walk-forward: {accuracy_value:.1f}% hit-rate on {factor_backtest['signals']} fired signals; 95% moving-block bootstrap interval {factor_backtest['ci_low_pct']:.1f}–{factor_backtest['ci_high_pct']:.1f}%; coverage {factor_backtest['coverage_pct']:.1f}% of {factor_backtest['checks']} checks; majority-direction baseline {factor_backtest['baseline_pct']:.1f}%; 95% lower-bound edge vs baseline {factor_backtest['edge_ci_low_pct']:+.1f} points. UP {up_hit_text}/{factor_backtest['up_signals']} · DOWN {down_hit_text}/{factor_backtest['down_signals']}.")
        else:
            st.caption("Insufficient selected-timeframe history to compute a meaningful factor hit-rate. UP/DOWN chances remain model estimates, not calibrated guarantees.")
        st.caption("UP/DOWN chance estimate combines recent momentum and the current combined factor score; manual inputs are not part of historical backtest. Percentages are not calibrated guarantees.")
        if not validation_ready:
            st.warning(f"Model direction is {bias}; paper trade filter is WAIT because the historical factor edge is not validated. This does not cancel the model's directional estimate.")
    else:
        st.info("Prediction probability and historical hit-rate need more usable candles for the selected timeframe.")
    st.caption(f"Prediction indicators use {prediction_timeframe} candles. Latest quote and dashboard refresh every 15 seconds (Yahoo data availability permitting).")
    st.subheader(f"🗓️ Direction estimate · {outlook_label}")
    if weekly:
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Chance of moving up", f"{weekly['up_pct']:.1f}%")
        w2.metric("Chance of moving down", f"{weekly['down_pct']:.1f}%")
        w3.metric("Chance of staying sideways", f"{weekly['sideways_pct']:.1f}%")
        w4.metric(f"Estimated {outlook_label} price range", f"{weekly['low']:,.0f} – {weekly['high']:,.0f}")
        st.caption(f"Model midpoint: {weekly['expected_price']:,.2f}. Probabilities use recent daily volatility and 5/20-session momentum over the selected horizon; rough estimates, not calibrated guarantees.")
    else:
        st.info("Weekly estimate needs at least 22 daily closing prices; Yahoo returned too little daily history.")
    left, right = st.columns(2)
    with left:
        st.subheader("📊 Market factors")
        factors = pd.DataFrame(technical["factors"], columns=["Factor", "Reading"])
        factors = pd.concat([pd.DataFrame([[f"1-candle momentum ({prediction_timeframe})", f"{ret_fast*100:+.2f}%"], [f"6-candle trend ({prediction_timeframe})", f"{ret_slow*100:+.2f}%"]], columns=["Factor", "Reading"]), factors], ignore_index=True)
        st.dataframe(factors, use_container_width=True, hide_index=True)
        st.caption(f"Technical votes: {technical['score']:+d}; futures buildup: {futures_vote:+d}; lead-stock alignment: {constituent_vote:+d}; GIFT Nifty input: {gift_vote:+d}; combined factor score: {combined_score:+d}. VIX affects risk context, not direction. Not a guaranteed forecast.")
    with right:
        st.subheader("⚠️ Decision logic")
        st.write("CE / bullish case: price confirms above the recent high and momentum stays positive.")
        st.write("PE / bearish case: price confirms below the recent low and momentum stays negative.")
        if "BULLISH" in bias and validation_ready:
            st.success("Current indicators lean bullish; bearish setups are still possible if price reverses and confirms.")
        elif "BEARISH" in bias and validation_ready:
            st.error("Current indicators lean bearish; bullish setups are still possible if price reverses and confirms.")
        elif "BULLISH" in bias:
            st.warning("Model direction leans bullish; the separate historical validation filter remains WAIT.")
        elif "BEARISH" in bias:
            st.warning("Model direction leans bearish; the separate historical validation filter remains WAIT.")
        else:
            st.warning("Signals are mixed. Either direction needs a confirmed move; otherwise WAIT.")

    st.subheader("🧭 Combined market map · levels and paper plan")
    if market_levels:
        map_rows = [
            ("Current spot", last, 0.0),
            ("Next-candle model lower range", low, low - last),
            ("Next-candle model upper range", high, high - last),
            (f"Recent support · {prediction_timeframe}", market_levels["recent_support"], market_levels["recent_support"] - last),
            (f"Recent resistance · {prediction_timeframe}", market_levels["recent_resistance"], market_levels["recent_resistance"] - last),
            ("Classic pivot", market_levels["pivot"], market_levels["pivot"] - last),
            ("S1", market_levels["s1"], market_levels["s1"] - last),
            ("S2", market_levels["s2"], market_levels["s2"] - last),
            ("R1", market_levels["r1"], market_levels["r1"] - last),
            ("R2", market_levels["r2"], market_levels["r2"] - last),
        ]
        if option_chain_summary and "error" not in option_chain_summary:
            map_rows.extend([
                ("Uploaded option-chain put OI wall · support context", option_chain_summary["put_wall"], option_chain_summary["put_wall"] - last),
                ("Uploaded option-chain call OI wall · resistance context", option_chain_summary["call_wall"], option_chain_summary["call_wall"] - last),
                ("Uploaded option-chain max-pain estimate", option_chain_summary["max_pain"], option_chain_summary["max_pain"] - last),
            ])
        st.dataframe(pd.DataFrame(map_rows, columns=["Level / scenario", "Index level", "Points from spot"]).round(2), use_container_width=True, hide_index=True)
        if "BULLISH" in bias and validation_ready:
            trigger, stop, targets = market_levels["bullish_trigger"], market_levels["bullish_stop"], market_levels["bullish_targets"]
            if option_chain_summary and "error" not in option_chain_summary:
                call_wall, put_wall = option_chain_summary["call_wall"], option_chain_summary["put_wall"]
                trigger = max(trigger, call_wall) if call_wall > last else trigger
                stop = min(stop, put_wall) if put_wall < last else stop
                targets = [max(targets[0], call_wall), max(targets[1], market_levels["r2"])]
            direction_label = "Bullish setup only after a candle confirms above the trigger"
        elif "BEARISH" in bias and validation_ready:
            trigger, stop, targets = market_levels["bearish_trigger"], market_levels["bearish_stop"], market_levels["bearish_targets"]
            if option_chain_summary and "error" not in option_chain_summary:
                call_wall, put_wall = option_chain_summary["call_wall"], option_chain_summary["put_wall"]
                trigger = min(trigger, put_wall) if put_wall < last else trigger
                stop = max(stop, call_wall) if call_wall > last else stop
                targets = [min(targets[0], put_wall), min(targets[1], market_levels["s2"])]
            direction_label = "Bearish setup only after a candle confirms below the trigger"
        else:
            trigger = stop = None
            targets = []
            direction_label = f"{bias} model direction · trade filter is WAIT; use only the conditional levels below, not an active setup"
        st.markdown(f"**Paper scenario:** {direction_label}")
        if trigger is not None:
            st.write(f"Trigger level: **{trigger:,.2f}** ({trigger-last:+,.2f} points from spot) · structural stop: **{stop:,.2f}** ({stop-last:+,.2f} points) · target zones: **{targets[0]:,.2f}** ({targets[0]-last:+,.2f} pts) then **{targets[1]:,.2f}** ({targets[1]-last:+,.2f} pts).")
        else:
            bull_trigger, bull_stop = market_levels["bullish_trigger"], market_levels["bullish_stop"]
            bear_trigger, bear_stop = market_levels["bearish_trigger"], market_levels["bearish_stop"]
            bull_targets, bear_targets = market_levels["bullish_targets"], market_levels["bearish_targets"]
            if option_chain_summary and "error" not in option_chain_summary:
                call_wall, put_wall = option_chain_summary["call_wall"], option_chain_summary["put_wall"]
                bull_trigger = max(bull_trigger, call_wall) if call_wall > last else bull_trigger
                bear_trigger = min(bear_trigger, put_wall) if put_wall < last else bear_trigger
                bull_stop = min(bull_stop, put_wall) if put_wall < last else bull_stop
                bear_stop = max(bear_stop, call_wall) if call_wall > last else bear_stop
                bull_targets = [max(bull_targets[0], call_wall), max(bull_targets[1], market_levels["r2"])]
                bear_targets = [min(bear_targets[0], put_wall), min(bear_targets[1], market_levels["s2"])]
            st.write(f"Conditional bullish only after confirmation above **{bull_trigger:,.2f}** ({bull_trigger-last:+,.2f} pts): stop **{bull_stop:,.2f}** ({bull_stop-last:+,.2f} pts), targets **{bull_targets[0]:,.2f}** ({bull_targets[0]-last:+,.2f} pts) / **{bull_targets[1]:,.2f}** ({bull_targets[1]-last:+,.2f} pts).")
            st.write(f"Conditional bearish only after confirmation below **{bear_trigger:,.2f}** ({bear_trigger-last:+,.2f} pts): stop **{bear_stop:,.2f}** ({bear_stop-last:+,.2f} pts), targets **{bear_targets[0]:,.2f}** ({bear_targets[0]-last:+,.2f} pts) / **{bear_targets[1]:,.2f}** ({bear_targets[1]-last:+,.2f} pts).")
        st.caption(f"ATR ({prediction_timeframe}): {market_levels['atr']:,.2f} points · stop/target volatility multiplier: {vix_multiplier:.2f}×. Pivot values use the previous daily candle; recent support/resistance use the last 20 selected-timeframe candles. Levels are scenario markers, not guaranteed fills or advice.")
    if india_vix is not None:
        vix_regime = "High volatility" if india_vix >= 20 else ("Elevated volatility" if india_vix >= 18 else ("Low volatility" if india_vix <= 14 else "Normal volatility"))
        st.write(f"India VIX input: **{india_vix:.2f}** · {vix_regime}; volatility context only, not a direction signal.")
    else:
        st.caption("India VIX not supplied. VIX is not inferred from index candles.")
    st.write(f"Futures buildup: **{futures_build}**" + (f" · price {futures_price_change:+.2f}% / OI {futures_oi_change:+.2f}%" if futures_price_change is not None and futures_oi_change is not None else " · enter current futures price/OI changes above if available"))
    if lead_a_change is not None or lead_b_change is not None:
        st.write(f"Lead-stock context ({lead_names[0]} / {lead_names[1]}): **{lead_a_change if lead_a_change is not None else '—'}% / {lead_b_change if lead_b_change is not None else '—'}%** · alignment vote {constituent_vote:+d}.")
    if gift_change is not None:
        st.write(f"GIFT Nifty input: **{gift_change:+.2f}%** · alignment vote {gift_vote:+d}.")

    st.subheader(f"🕯️ {choice} candlestick chart · {chart_timeframe}")
    chart = chart_df.tail(120)
    fig = go.Figure(data=[go.Candlestick(x=chart.index, open=chart["Open"], high=chart["High"], low=chart["Low"], close=chart["Close"], name=choice)])
    fig.update_layout(template="plotly_dark", height=520, xaxis_title="Time", yaxis_title="Index price", xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("📰 Latest market news")
    news_queries = {
        "NIFTY": "NIFTY 50 NSE India",
        "BANKNIFTY": "Nifty Bank NSE India",
        "SENSEX": "BSE Sensex India",
    }
    news = get_market_news(news_queries[choice], "recent-7-days-v1")
    if news:
        for article in news:
            if article["link"]:
                st.link_button(article["title"], article["link"], use_container_width=True)
            st.caption(f"{article['source']} · {article['published']}")
    else:
        st.info("News feed unavailable right now. Try Refresh later.")
    st.subheader("📈 Paper options chart · theoretical premium")
    st.caption("Model estimate only, not a live option-chain quote. Adjust the contract assumptions; check live premium, OI, IV and spread with your broker.")
    opt1, opt2, opt3, opt4 = st.columns(4)
    option_type = opt1.selectbox("Option side", ["CE", "PE"], key="paper_option_side")
    strike_step = 100 if choice == "SENSEX" else (100 if choice == "BANKNIFTY" else 50)
    default_strike = round(last / strike_step) * strike_step
    strike = opt2.number_input("Strike", min_value=1.0, value=float(default_strike), step=float(strike_step), key="paper_option_strike")
    days_to_expiry = opt3.number_input("Days to expiry", min_value=1, max_value=90, value=5, step=1, key="paper_option_days")
    daily_returns = weekly_df["Close"].astype(float).pct_change().dropna() if not weekly_df.empty and "Close" in weekly_df else pd.Series(dtype=float)
    daily_vol = float(daily_returns.tail(60).std()) if len(daily_returns) > 1 else 0.20 / np.sqrt(252)
    annual_vol = float(np.sqrt(max(daily_vol, 0.000001) ** 2 * 252))
    iv_percent = opt4.number_input("Assumed IV %", min_value=1.0, max_value=200.0, value=float(np.clip(annual_vol * 100, 5, 100)), step=1.0, key="paper_option_iv")
    spot_grid = np.linspace(last * 0.90, last * 1.10, 81)
    premiums = [black_scholes_price(spot, strike, days_to_expiry, iv_percent / 100, option_type) for spot in spot_grid]
    option_fig = go.Figure()
    option_fig.add_trace(go.Scatter(x=spot_grid, y=premiums, mode="lines", name="Theoretical premium", line=dict(color="#56b4ff", width=3)))
    intrinsic = [max(spot - strike, 0) if option_type == "CE" else max(strike - spot, 0) for spot in spot_grid]
    option_fig.add_trace(go.Scatter(x=spot_grid, y=intrinsic, mode="lines", name="Expiry intrinsic value", line=dict(color="#f5a742", dash="dash")))
    option_fig.add_vline(x=last, line_dash="dot", line_color="#ffffff", annotation_text=f"Spot {last:,.2f}")
    option_fig.update_layout(template="plotly_dark", height=360, xaxis_title="Underlying index level", yaxis_title="Model premium (index points)", margin=dict(l=10, r=10, t=25, b=10), legend=dict(orientation="h"))
    st.plotly_chart(option_fig, use_container_width=True)
    model_premium = black_scholes_price(last, strike, days_to_expiry, iv_percent / 100, option_type)
    breakeven = strike + model_premium if option_type == "CE" else strike - model_premium
    st.caption(f"Model premium at current spot: {model_premium:,.2f} points · expiry breakeven (premium treated per index unit): {breakeven:,.2f}")
    greeks = calculate_option_greeks(last, strike, days_to_expiry, iv_percent / 100, option_type)
    g1, g2, g3 = st.columns(3)
    g1.metric("Model delta", f"{greeks['delta']:.3f}")
    g2.metric("Model gamma", f"{greeks['gamma']:.5f}")
    g3.metric("Model theta / calendar day", f"{greeks['theta_per_day']:.3f}")
    st.caption("Greeks are theoretical Black-Scholes estimates from the selected strike, IV and expiry, not exchange-calculated live Greeks.")
    exchange_url = "https://www.bseindia.com/markets/Derivatives/DeriReports/DeriOptionchain.html" if choice == "SENSEX" else "https://www.nseindia.com/option-chain"
    st.link_button("Open official exchange option chain", exchange_url)
    st.subheader("🧠 How to use this on expiry day")
    st.markdown("""1. Select an index and a candle chart timeframe.\n2. Choose a separate prediction timeframe.\n3. Treat the output as an estimate, not a guaranteed forecast.\n4. Either bullish or bearish setups may be considered only after price confirmation; mixed signals suggest WAIT.\n5. Verify option chain, OI, IV, spreads and expiry levels in your broker/market terminal.""")
    st.info("This build does not place real orders. A broker integration requires appropriate API access and locally stored credentials.")
    with st.sidebar:
        if st.button("✨ Generate AI summary", disabled=not api_key, use_container_width=True):
            try:
                week_summary = f"; {outlook_label} chance up/down/sideways: {weekly['up_pct']:.1f}%/{weekly['down_pct']:.1f}%/{weekly['sideways_pct']:.1f}%" if weekly else "; selected-horizon estimate unavailable"
                direction_summary = f"; selected-timeframe next-candle UP/DOWN chance {prediction_outlook['up_pct']:.1f}%/{prediction_outlook['down_pct']:.1f}%; technical factor hit-rate {factor_backtest['hit_rate_pct']:.1f}% on {factor_backtest['signals']} signals/{factor_backtest['checks']} checks (coverage {factor_backtest['coverage_pct']:.1f}%, baseline {factor_backtest['baseline_pct']:.1f}%, 95% moving-block CI {factor_backtest['ci_low_pct']:.1f}-{factor_backtest['ci_high_pct']:.1f}%, lower-bound edge {factor_backtest['edge_ci_low_pct']:+.1f} points)" if prediction_outlook and factor_backtest and factor_backtest['hit_rate_pct'] is not None and factor_backtest['edge_ci_low_pct'] is not None else "; selected-timeframe probability/history unavailable"
                market_map_text = f"; support/resistance {market_levels['recent_support']:.2f}/{market_levels['recent_resistance']:.2f}; pivot S1/S2/P/R1/R2 {market_levels['s1']:.2f}/{market_levels['s2']:.2f}/{market_levels['pivot']:.2f}/{market_levels['r1']:.2f}/{market_levels['r2']:.2f}; combined-score plan {bias}, trigger/stop/targets {trigger if trigger is not None else 'WAIT'}/{stop if stop is not None else '—'}/{targets if targets else '—'}" if market_levels else "; market levels unavailable"
                oi_text = f"; uploaded option-chain PCR {option_chain_summary['pcr']:.2f}, call wall {option_chain_summary['call_wall']:.2f}, put wall {option_chain_summary['put_wall']:.2f}, max pain {option_chain_summary['max_pain']:.2f}" if option_chain_summary and "error" not in option_chain_summary else "; no valid option-chain CSV uploaded"
                context_text = f"; India VIX {india_vix if india_vix is not None else 'not supplied'}; futures buildup {futures_build}; lead-stock changes {lead_a_change}/{lead_b_change}; GIFT Nifty change {gift_change}; context votes futures/constituents/GIFT {futures_vote}/{constituent_vote}/{gift_vote}"
                response = requests.post("https://api.openai.com/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "gpt-4o-mini", "messages": [{"role": "system", "content": "In concise Hindi, summarize these paper-analysis indicators, market levels, optional market context and news. Distinguish unavailable data from measured data. Explain uncertainty; do not promise returns or give personalized financial advice."}, {"role": "user", "content": f"Index: {choice}; chart timeframe: {chart_timeframe}; prediction timeframe: {prediction_timeframe}; latest quote: {last:.2f}; predicted next close: {predicted_price:.2f}; bias: {bias}; technical/futures vote score: {technical['score']}/{futures_vote}; factors: {technical['factors']}; one-candle return: {ret_fast*100:.2f}%; six-candle trend: {ret_slow*100:.2f}%; estimated range {low:.2f}-{high:.2f}{direction_summary}{week_summary}{market_map_text}{oi_text}{context_text}; recent headlines: {[item['title'] for item in news[:5]]}."}], "temperature": 0.3, "max_tokens": 500}, timeout=30)
                response.raise_for_status()
                st.sidebar.write(response.json()["choices"][0]["message"]["content"])
            except requests.HTTPError as exc:
                status = exc.response.status_code if exc.response is not None else 0
                if status == 429:
                    st.sidebar.error("OpenAI API credits/quota khatam hain. OpenAI Platform par billing/credits add karke dobara try karein.")
                elif status == 401:
                    st.sidebar.error("OpenAI key invalid ya revoked hai. Nayi API key set karein.")
                elif status == 403:
                    st.sidebar.error("Is project ko model/API ka access nahi hai. OpenAI project permissions check karein.")
                else:
                    st.sidebar.error(f"OpenAI request failed (HTTP {status}). Dobara try karein.")
            except requests.RequestException:
                st.sidebar.error("OpenAI tak connection nahi hua. Internet connection check karke dobara try karein.")
            except (KeyError, IndexError, ValueError):
                st.sidebar.error("AI returned an unexpected response. Please try again.")
    st.caption(f"Last dashboard update: {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")

render_dashboard()
