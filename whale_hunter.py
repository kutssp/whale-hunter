#!/usr/bin/env python3
"""
whale_watch_free.py
MVP: на бесплатных API (CoinGecko, DexScreener, Covalent) выявляет кошельки, которые
купили большие суммы ПОЧТИ перед началом роста токена, и ранжирует их по "WhaleScore".

Настрой:
 - Установи COVALENT_API_KEY (бесплатно на covalenthq.com)
 - Установи python-зависимости: requests, pandas
    pip install requests pandas

Запуск:
    python whale_hunter.py

Выход:
 - CSV: whale_candidates.csv
 - В stdout: разбивка и рекомендации (топ-кошельки к наблюдению)
"""
import os
import time
import math
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional

# ----------------- CONFIG -----------------
COVALENT_API_KEY = os.getenv("COVALENT_API_KEY", "YOUR_COVALENT_KEY_HERE")
CHAIN_ID = 1  # Ethereum mainnet
TOP_GAINERS_COUNT = 5
AMOUNT_THRESHOLD_USD = 5000           # минимальная ранняя покупка USD
EARLY_WINDOW_HOURS = 6                # сколько часов до pivot считаем "ранним"
PIVOT_RISE_PCT = 0.30                 # pivot: 30% rise within WINDOW_HOURS_CHECK
WINDOW_HOURS_CHECK = 3                 # check 3-hour rises
REQUEST_SLEEP = 1.0                   # delay between API calls to avoid rate limits
# Weights for WhaleScore (can be tuned)
WEIGHTS = {
    "wA": 0.45,
    "wT": 0.25,
    "wR": 0.15,
    "wC": 0.20,
    "wH": 0.25,
    "wL": 0.10,
}
A_MAX = 10_000_000.0
TAU_HOURS = 6.0
OUTPUT_CSV = "whale_candidates.csv"

# Minimal heuristic: addresses with many tokens/txs may be exchange-like (used to detect transfers to CEX)
EXCHANGE_LIKE_MIN_TOKEN_COUNT = 50
EXCHANGE_LIKE_MIN_TX_COUNT = 1000

# ----------------- HELPERS -----------------
def safe_request(url, params=None, headers=None, max_retries=3, backoff=1.5):
    """Robust GET with retries and backoff"""
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                return r.json()
            else:
                # handle rate limit / server busy
                print(f"[WARN] HTTP {r.status_code} for {url} (attempt {attempt+1})")
        except Exception as e:
            print(f"[WARN] Request error {e} for {url} (attempt {attempt+1})")
        time.sleep(backoff ** attempt)
    return None

def iso_to_dt(s: str) -> datetime:
    # accepts string with or without 'Z'
    if s.endswith("Z"):
        s = s[:-1]
    # some APIs return microseconds; handle
    try:
        return datetime.fromisoformat(s)
    except Exception:
        # fallback parse
        return datetime.strptime(s.split(".")[0], "%Y-%m-%dT%H:%M:%S")

# ----------------- API ACCESS -----------------
def get_top_gainers_coingecko(n=TOP_GAINERS_COUNT) -> List[Dict]:
    url = "https://api.coingecko.com/api/v3/coins/markets"
    params = {
        "vs_currency": "usd",
        "order": "market_cap_desc",
        "per_page": 250,
        "page": 1,
        "price_change_percentage": "24h"
    }
    j = safe_request(url, params=params)
    if not j:
        return []
    # sort by 24h change desc
    sorted_tokens = sorted(j, key=lambda x: x.get("price_change_percentage_24h", 0) or 0, reverse=True)
    # keep tokens that have platform data and contract address for Ethereum (or chain-specific)
    results = []
    for tok in sorted_tokens:
        # try to get ethereum contract — on CoinGecko token['platforms'] may include 'ethereum'
        platforms = tok.get("platforms") or {}
        contract = platforms.get("ethereum")
        if contract:
            results.append({
                "id": tok.get("id"),
                "symbol": tok.get("symbol"),
                "name": tok.get("name"),
                "contract": contract,
                "price_change_pct_24h": tok.get("price_change_percentage_24h"),
                "market_cap": tok.get("market_cap"),
            })
        if len(results) >= n:
            break
    return results

def get_dexscreener_token(contract_address: str) -> Optional[Dict]:
    # DexScreener endpoint: returns pairs & price points
    url = f"https://api.dexscreener.com/latest/dex/tokens/{contract_address}"
    j = safe_request(url)
    time.sleep(REQUEST_SLEEP)
    return j

def covalent_token_holders(contract_address: str, page_size=100) -> List[Dict]:
    # /v1/{chain_id}/tokens/{contract-address}/token_holders/
    url = f"https://api.covalenthq.com/v1/{CHAIN_ID}/tokens/{contract_address}/token_holders/"
    params = {"page-size": page_size, "key": COVALENT_API_KEY}
    j = safe_request(url, params=params)
    time.sleep(REQUEST_SLEEP)
    if not j:
        return []
    return j.get("data", {}).get("items", [])

def covalent_token_transfers(contract_address: str, page_size=200) -> List[Dict]:
    # We use events/address endpoint as a general way to retrieve recent events. Covalent also has token transfers endpoints.
    # Use: /v1/{chain_id}/events/address/{contract}/
    url = f"https://api.covalenthq.com/v1/{CHAIN_ID}/events/address/{contract_address}/"
    params = {"page-size": page_size, "key": COVALENT_API_KEY}
    j = safe_request(url, params=params)
    time.sleep(REQUEST_SLEEP)
    if not j:
        return []
    return j.get("data", {}).get("items", [])

def covalent_address_balances(address: str) -> List[Dict]:
    # /v1/{chain_id}/address/{address}/balances_v2/
    url = f"https://api.covalenthq.com/v1/{CHAIN_ID}/address/{address}/balances_v2/"
    params = {"key": COVALENT_API_KEY}
    j = safe_request(url, params=params)
    time.sleep(REQUEST_SLEEP)
    if not j:
        return []
    return j.get("data", {}).get("items", [])

def covalent_address_transactions(address: str, page_size=100) -> List[Dict]:
    url = f"https://api.covalenthq.com/v1/{CHAIN_ID}/address/{address}/transactions_v2/"
    params = {"page-size": page_size, "key": COVALENT_API_KEY}
    j = safe_request(url, params=params)
    time.sleep(REQUEST_SLEEP)
    if not j:
        return []
    return j.get("data", {}).get("items", [])

# ----------------- ANALYSIS LOGIC -----------------
def detect_pivot_from_price_points(price_points: List[Dict]) -> Optional[Dict]:
    """
    price_points: list of dicts as returned by DexScreener:
      [{"time": "2025-09-28T..Z", "priceUsd": 0.00123}, ...]
    We'll search for a point where price rises >= PIVOT_RISE_PCT over next WINDOW_HOURS_CHECK samples.
    DexScreener pricePoints often sampled per hour or per minute depending on pair.
    """
    if not price_points or len(price_points) < WINDOW_HOURS_CHECK + 1:
        return None
    # normalize keys
    pts = []
    for p in price_points:
        # DexScreener uses "time" and "priceUsd" or "price"
        t = p.get("time") or p.get("timestamp")
        pr = p.get("priceUsd") or p.get("price") or p.get("priceUsd")
        if t and pr is not None:
            try:
                pts.append({"time": t, "price": float(pr)})
            except:
                continue
    n = len(pts)
    for i in range(n - WINDOW_HOURS_CHECK):
        p0 = pts[i]["price"]
        pj = pts[i + WINDOW_HOURS_CHECK]["price"]
        if pj >= p0 * (1 + PIVOT_RISE_PCT):
            return {"pivot_time": pts[i]["time"], "pivot_price": p0, "index": i}
    return None

def norm_amount(a_usd: float) -> float:
    return math.log(1 + a_usd) / math.log(1 + A_MAX)

def norm_time_delta(pivot_time_iso: str, tx_time_iso: str) -> float:
    pivot = iso_to_dt(pivot_time_iso)
    tx = iso_to_dt(tx_time_iso)
    delta_hours = max(0.0, (pivot - tx).total_seconds() / 3600.0)
    return math.exp(-delta_hours / TAU_HOURS)

def norm_top10_share(top10_pct: float) -> float:
    return min(1.0, top10_pct / 60.0)

def compute_whale_score(amount_usd: float, pivot_time_iso: str, tx_time_iso: str,
                        reputation_score: float, transfer_to_cex_flag: float,
                        top10_share_pct: float, lp_locked_flag: float) -> float:
    a = norm_amount(amount_usd)
    t = norm_time_delta(pivot_time_iso, tx_time_iso)
    r = reputation_score / 100.0
    c = transfer_to_cex_flag  # 0 or 1 (or 0..1)
    h = norm_top10_share(top10_share_pct)
    l = lp_locked_flag  # 0 or 1
    w = WEIGHTS
    score = w["wA"]*a + w["wT"]*t + w["wR"]*r - w["wC"]*c - w["wH"]*h + w["wL"]*l
    return max(0.0, min(1.2, score))  # clamp

# ----------------- WALLET PROFILING (heuristics using free data) -----------------
def profile_wallet_heuristic(wallet: str) -> Dict:
    """
    Use Covalent free endpoints to compute a lightweight 'reputation' score:
      - wallet_age_score: based on first_tx timestamp
      - token_diversity_score: number of non-zero token balances (normalized)
      - tx_activity_score: based on recent tx counts
      - transfer_to_cex_flag: heuristic: whether wallet sends funds to exchange-like addresses
    Returns:
       {reputation_score (0..100), transfer_to_cex_flag (0/1), wallet_age_days, n_tokens, n_txs}
    """
    # balances
    balances = covalent_address_balances(wallet)
    n_tokens = len([b for b in balances if float(b.get("quote", 0) or 0) > 0])
    # transactions
    txs = covalent_address_transactions(wallet, page_size=50)
    n_txs = len(txs)
    # estimate first tx time -> wallet age
    first_ts = None
    for tx in txs[::-1]:  # transactions_v2 usually returns newest first; reversing to try older
        if tx.get("block_signed_at"):
            first_ts = tx["block_signed_at"]
    if first_ts:
        age_days = (datetime.utcnow() - iso_to_dt(first_ts)).days
    else:
        # fallback: assume new wallet (0 days)
        age_days = 0

    # compute sub-scores
    age_score = min(1.0, age_days / 365.0)  # normalized 0..1 for up to 1 year
    diversity_score = min(1.0, n_tokens / 20.0)  # 20 tokens -> 1.0
    tx_score = min(1.0, n_txs / 500.0)

    # naive transfer-to-CEX detection: check recent outgoing transfers' recipients for exchange-like behavior
    transfer_to_cex_flag = 0
    # Heuristic: if recipient has many tokens & many txs -> exchange-like
    for tx in txs[:20]:
        # find outgoing transfers (from this wallet)
        if tx.get("from_address","").lower() == wallet.lower():
            # look for to_address; in transactions_v2 item, there's "to_address"
            to_addr = tx.get("to_address")
            if to_addr:
                # profile recipient quickly (only few)
                rec_bal = covalent_address_balances(to_addr)
                rec_tokens = len(rec_bal)
                rec_txs = len(covalent_address_transactions(to_addr, page_size=20))
                if rec_tokens >= EXCHANGE_LIKE_MIN_TOKEN_COUNT or rec_txs >= EXCHANGE_LIKE_MIN_TX_COUNT:
                    transfer_to_cex_flag = 1
                    break
    # combine reputation: age*0.4 + diversity*0.35 + tx*0.25 -> scale 0..100
    rep = (0.4*age_score + 0.35*diversity_score + 0.25*tx_score) * 100.0
    return {
        "reputation_score": round(rep, 2),
        "transfer_to_cex_flag": transfer_to_cex_flag,
        "wallet_age_days": age_days,
        "n_tokens": n_tokens,
        "n_txs": n_txs
    }

# ----------------- CORE PIPELINE -----------------
def analyze_token_and_find_whales(contract: str, symbol: str, name: str) -> List[Dict]:
    """
    For a token contract:
      - fetch DexScreener price points -> detect pivot
      - fetch token transfers (Covalent) -> find large early buyers within EARLY_WINDOW_HOURS before pivot
      - fetch token holders -> top10 concentration
      - profile each buyer and compute whale score
    """
    print(f"\n==== Analyzing {name} ({symbol}) contract {contract} ====")
    dex = get_dexscreener_token(contract)
    if not dex or "pairs" not in dex or len(dex["pairs"]) == 0:
        print("No DexScreener data; skipping.")
        return []

    # choose primary pair (first)
    pair = dex["pairs"][0]
    price_points = pair.get("pricePoints") or pair.get("price_points") or []
    pivot = detect_pivot_from_price_points(price_points)
    if not pivot:
        print("Pivot not found for this token; skipping.")
        return []
    pivot_time = pivot["pivot_time"]
    pivot_dt = iso_to_dt(pivot_time)
    print(f"Pivot detected at {pivot_time}")

    # get token transfers (Covalent) - try to find swap-like events or transfers
    events = covalent_token_transfers(contract, page_size=500)
    if not events:
        print("No Covalent transfer events; skipping.")
        return []

    # Parse events: Covalent events structure is varied; we'll look for events with "from_address" and "raw_log_topics"/"decoded" fields
    candidate_buyers = []
    for ev in events:
        # Determine timestamp and from address; many events correspond to internal activity
        ts = ev.get("block_signed_at")
        # skip events after pivot (we want before)
        if not ts:
            continue
        ev_dt = iso_to_dt(ts)
        if ev_dt >= pivot_dt:
            continue
        # Heuristic: treat events where this contract receives value or swap events as buys.
        # Covalent's events have "decoded" with "name"/"params" sometimes; fallback to "raw_log_topics"
        # We'll conservatively treat any event where 'from_address' exists and there is value (quote) as candidate
        from_addr = ev.get("tx_from") or ev.get("from_address") or ev.get("sender_address")
        # attempt to get USD quote from ev['tx_hash'] via transaction info (Covalent sometimes provides 'value'/'log_events' with 'decoded')
        amount_usd = None
        # Try to read 'decoded' or 'log_events' amounts
        log_events = ev.get("log_events") or []
        # look for transfer logs with 'value' or 'decoded' param named 'value' or 'amount'
        for le in log_events:
            try:
                # value may be present in 'delta' or 'sender' or 'decoded'
                decoded = le.get("decoded")
                if decoded and decoded.get("name") in ("Transfer", "Swap"):
                    for p in decoded.get("params", []):
                        if p.get("name") in ("value", "amount", "amount0", "amount1"):
                            val_raw = float(p.get("value") or 0)
                            # For ERC20, we need to scale by token decimals; but Covalent sometimes provides 'delta' or 'sender_balance_change'
                            # fallback: put placeholder as None
                            amount_usd = None
            except:
                pass
        # As extracting exact USD amount from Covalent events is messy and sometimes limited on free tier,
        # we'll fallback to treating txs with 'value' > 0 (wei) or with transfer logs as candidate and later enrich
        # For MVP: we'll create a candidate with placeholder amount and later estimate using pivot price if possible.
        if from_addr:
            candidate_buyers.append({"wallet": from_addr, "time": ts, "tx": ev.get("tx_hash")})
    # Deduplicate candidate_buyers by wallet, choose earliest event per wallet
    buyers_map = {}
    for c in candidate_buyers:
        w = c["wallet"].lower()
        if w not in buyers_map or iso_to_dt(c["time"]) < iso_to_dt(buyers_map[w]["time"]):
            buyers_map[w] = c

    # We need amount_usd per buyer to score properly. Try to estimate using token balance changes around the tx or using tx-level info.
    # Simpler approach: for each wallet, check token balance snapshot before pivot and at pivot (from holders list) to compute delta.
    holders = covalent_token_holders(contract, page_size=200)
    # holders are ordered by balance desc; compute top10 share
    top10_share = 0.0
    try:
        total_supply = 0
        top10_sum = 0
        for i, h in enumerate(holders):
            bal = float(h.get("balance", 0))
            total_supply += bal
            if i < 10:
                top10_sum += bal
        if total_supply > 0:
            top10_share = (top10_sum / total_supply) * 100.0
    except Exception:
        top10_share = 0.0

    # For each wallet, estimate amount_usd by reading wallet token balance now vs historic (approx):
    results = []
    for w, info in buyers_map.items():
        wallet = w
        buy_time = info["time"]
        # Get balances of this wallet: token balance for contract
        balances = covalent_address_balances(wallet)
        token_balance = 0.0
        for b in balances:
            if b.get("contract_address", "").lower() == contract.lower():
                # Covalent returns 'balance' raw; 'quote' is USD value of the whole wallet; 'quote_rate' is price.
                token_balance = float(b.get("balance") or 0)
                token_quote = float(b.get("quote") or 0)
                # we can use quote_rate if available
                quote_rate = float(b.get("quote_rate") or 0)
                break
        # If token_quote available, use token_quote as current USD value of the wallet's holdings in this token (approx)
        # Not perfect but for MVP it gives ballpark. For amount bought before pivot, we assume they acquired at price near pivot_price.
        estimated_amount_usd = None
        if 'token_quote' in locals() and token_quote:
            estimated_amount_usd = token_quote
        else:
            # fallback: use pair price at pivot to estimate token holding USD (not accurate)
            # find price at pivot from price_points pivot index
            pivot_price = pivot["pivot_price"]
            # token_balance is in raw units; we cannot scale decimals safely here -> fallback to None
            estimated_amount_usd = None

        # Profile wallet
        profile = profile_wallet_heuristic(wallet)
        reputation_score = profile["reputation_score"]
        transfer_to_cex_flag = profile["transfer_to_cex_flag"]

        # lp_locked_flag (we don't have a free unconditional check) -> 0 for now
        lp_locked_flag = 0

        # If estimated_amount_usd is None, skip if we cannot estimate; otherwise keep if >= threshold
        if estimated_amount_usd is None:
            # To avoid losing candidates due to missing USD estimate, we give them a small placeholder
            estimated_amount_usd = AMOUNT_THRESHOLD_USD  # conservative placeholder so they enter scoring; user can refine locally
        # filter by min amount
        if estimated_amount_usd < AMOUNT_THRESHOLD_USD:
            continue

        score = compute_whale_score(
            amount_usd=estimated_amount_usd,
            pivot_time_iso=pivot_time,
            tx_time_iso=buy_time,
            reputation_score=reputation_score,
            transfer_to_cex_flag=transfer_to_cex_flag,
            top10_share_pct=top10_share,
            lp_locked_flag=lp_locked_flag
        )
        results.append({
            "token": symbol,
            "name": name,
            "contract": contract,
            "wallet": wallet,
            "estimated_amount_usd": round(estimated_amount_usd, 2),
            "buy_time": buy_time,
            "reputation_score": reputation_score,
            "transfer_to_cex_flag": transfer_to_cex_flag,
            "top10_share_pct": round(top10_share, 2),
            "lp_locked_flag": lp_locked_flag,
            "whale_score": round(score, 4),
            "tx_sample": info.get("tx")
        })

    # sort by whale_score desc
    results_sorted = sorted(results, key=lambda x: x["whale_score"], reverse=True)
    return results_sorted

def run_pipeline_and_save():
    print("Starting pipeline: CoinGecko -> DexScreener -> Covalent")
    tokens = get_top_gainers_coingecko(TOP_GAINERS_COUNT)
    print(f"Found {len(tokens)} candidate gainers from CoinGecko")
    all_candidates = []
    for t in tokens:
        contract = t["contract"]
        symbol = t["symbol"]
        name = t["name"]
        try:
            candidates = analyze_token_and_find_whales(contract, symbol, name)
            all_candidates.extend(candidates)
        except Exception as e:
            print(f"[ERROR] analyzing {symbol} {contract}: {e}")
    # Save CSV
    if all_candidates:
        df = pd.DataFrame(all_candidates)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved {len(all_candidates)} candidate rows to {OUTPUT_CSV}")
        print("\nTop candidates:")
        print(df.sort_values("whale_score", ascending=False).head(10).to_string(index=False))
    else:
        print("No candidates found in this run.")
    return all_candidates

# ----------------- MAIN -----------------
if __name__ == "__main__":
    if COVALENT_API_KEY == "YOUR_COVALENT_KEY_HERE" or not COVALENT_API_KEY:
        print("⚠️ Please set COVALENT_API_KEY environment variable or edit the script with your key.")
        print("Get free key at https://www.covalenthq.com")
    else:
        results = run_pipeline_and_save()
        # Quick recommendation: show top wallet to watch (highest whale_score)
        if results:
            top = sorted(results, key=lambda x: x["whale_score"], reverse=True)[0]
            print("\n=== РЕКОМЕНДАЦИЯ ===")
            print(f"Наблюдай кошелёк: {top['wallet']}")
            print(f"Токен: {top['token']} ({top['name']})")
            print(f"Оценка WhaleScore: {top['whale_score']}")
            print(f"Оценочная сумма покупки (USD): {top['estimated_amount_usd']}")
            print(f"Buy time: {top['buy_time']}, tx sample: {top['tx_sample']}")
            print("Добавь этот кошелёк в наблюдение и при появлении похожих ранних покупок — можешь повторять (с управлением риском).")
        else:
            print("Нет явных кошельков для наблюдения в этом запуске.")
