import os
import requests
import pandas as pd

API_KEY = os.environ["TWELVE_DATA_API_KEY"]
TELEGRAM_BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

PAIRS = {
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "USDJPY": "USD/JPY",
    "USDCHF": "USD/CHF",
    "USDCAD": "USD/CAD",
    "AUDUSD": "AUD/USD",
    "NZDUSD": "NZD/USD",
}

def get_data(symbol):
    url = "https://api.twelvedata.com/time_series"
    params = {
        "symbol": symbol,
        "interval": "15min",
        "outputsize": 100,
        "apikey": API_KEY,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    if "values" not in data:
        raise RuntimeError(f"{symbol}: {data.get('message', data)}")
    df = pd.DataFrame(data["values"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col])
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
    df = df.sort_values("datetime").reset_index(drop=True)
    return df

def rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    out = 100 - (100 / (1 + rs))
    return out.astype(float)

def atr(df, length=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False, min_periods=length).mean()

def scan_pair(symbol):
    df = get_data(symbol)

    df["ema20"] = df["close"].ewm(span=20, adjust=False, min_periods=20).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False, min_periods=50).mean()
    df["rsi14"] = rsi(df["close"], 14)
    df["atr14"] = atr(df, 14)

    # TradingView: ta.highest(high, 20)[1] / ta.lowest(low, 20)[1]
    df["hh"] = df["high"].rolling(20).max().shift(1)
    df["ll"] = df["low"].rolling(20).min().shift(1)

    row = df.iloc[-1]
    c, e20, e50 = row["close"], row["ema20"], row["ema50"]
    r = row["rsi14"]

    bull = (
        int(c > e20)
        + int(e20 > e50)
        + int(r > 55)
        + int(c > row["hh"])
        + int(c > e50)
    )
    bear = (
        int(c < e20)
        + int(e20 < e50)
        + int(r < 45)
        + int(c < row["ll"])
        + int(c < e50)
    )

    score = bull if bull >= bear else -bear

    if score >= 4:
        signal = "STRONG BUY"
    elif score >= 2:
        signal = "BUY"
    elif score <= -4:
        signal = "STRONG SELL"
    elif score <= -2:
        signal = "SELL"
    else:
        signal = "NEUTRAL"

    return {
        "pair": symbol.replace("/", ""),
        "score": int(score),
        "rsi": float(r),
        "signal": signal,
        "time": str(row["datetime"]),
    }

def send_telegram(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    r = requests.post(
        url,
        data={"chat_id": TELEGRAM_CHAT_ID, "text": message},
        timeout=30,
    )
    r.raise_for_status()

def main():
    results = []
    errors = []

    for name, symbol in PAIRS.items():
        try:
            results.append(scan_pair(symbol))
        except Exception as e:
            errors.append(f"{name}: {e}")

    strong = [x for x in results if x["score"] >= 4 or x["score"] <= -4]

    if strong:
        lines = ["🚨 FOREX STRONG SIGNAL", ""]
        for x in strong:
            lines.append(f'{x["pair"]} — {x["signal"]} (Score {x["score"]}, RSI {x["rsi"]:.1f})')
        lines.append("")
        lines.append("Timeframe: 15 minutes")
        send_telegram("\n".join(lines))

    if errors:
        send_telegram("⚠️ Forex scanner errors:\n" + "\n".join(errors))

if __name__ == "__main__":
    main()
