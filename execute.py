"""
Paper/Live trading bridge for HyperLiquid.
Connects the best strategy to the exchange.

DEFAULT MODE: PAPER (testnet). Set LIVE_MODE = True only when explicitly ready.

Usage:
    uv run execute.py                    # paper trading on testnet
    uv run execute.py --live             # live trading (requires confirmation)
    uv run execute.py --once             # single evaluation, no loop
    uv run execute.py --interval 3600    # custom interval in seconds (default: 1h)

DO NOT RUN THIS DURING THE OPTIMIZATION LOOP.
This file is for manual use only, after you have a strategy you trust.
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from prepare import (
    ASSETS, CANDLE_INTERVAL, MAX_LEVERAGE,
    INITIAL_CAPITAL, load_market_data, compute_base_features,
    download_all_data, LOOKBACK_DAYS
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LIVE_MODE = False               # PAPER by default — only change explicitly
CHECK_INTERVAL = 300            # seconds between re-evaluations (5 minutes)
LIVE_CANDLE_INTERVAL = "5m"     # candle resolution for live/paper trading
MAX_DAILY_LOSS_PCT = 5.0        # kill switch: max daily loss as % of capital
MAX_POSITION_USD = 50_000       # max notional per position
DRY_RUN = True                  # if True, log trades but don't execute

# Paper trading state file
PAPER_STATE_FILE = os.path.expanduser("~/.cache/autotrading/paper_state.json")


# ---------------------------------------------------------------------------
# Paper Trading Engine
# ---------------------------------------------------------------------------

class PaperTrader:
    """Simulated trading engine for paper mode. Tracks positions and PnL locally."""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}     # asset -> units
        self.entry_prices = {}  # asset -> avg entry price
        self.trade_log = []
        self.start_time = datetime.now(timezone.utc)
        self._load_state()

    def _state_path(self):
        return PAPER_STATE_FILE

    def _load_state(self):
        path = self._state_path()
        if os.path.exists(path):
            with open(path) as f:
                state = json.load(f)
            self.cash = state.get("cash", self.initial_capital)
            self.positions = state.get("positions", {})
            self.entry_prices = state.get("entry_prices", {})
            self.trade_log = state.get("trade_log", [])
            print(f"Loaded paper state: cash=${self.cash:.2f}, positions={self.positions}")

    def _save_state(self):
        os.makedirs(os.path.dirname(self._state_path()), exist_ok=True)
        state = {
            "cash": self.cash,
            "positions": self.positions,
            "entry_prices": self.entry_prices,
            "trade_log": self.trade_log[-1000:],  # keep last 1000 trades
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._state_path(), "w") as f:
            json.dump(state, f, indent=2)

    def get_equity(self, prices: dict[str, float]) -> float:
        equity = self.cash
        for asset, units in self.positions.items():
            if asset in prices:
                equity += units * prices[asset]
        return equity

    def execute_trade(self, asset: str, target_units: float, price: float):
        current_units = self.positions.get(asset, 0.0)
        delta = target_units - current_units

        if abs(delta) * price < 10:  # min $10 trade
            return

        # Simulate slippage
        slippage = 0.0002  # 2 bps
        exec_price = price * (1 + slippage * np.sign(delta))

        # Commission
        commission = abs(delta) * exec_price * 0.00035  # 3.5 bps

        cost = delta * exec_price + commission
        self.cash -= cost
        self.positions[asset] = target_units

        if target_units != 0:
            self.entry_prices[asset] = exec_price
        elif asset in self.entry_prices:
            del self.entry_prices[asset]

        trade = {
            "time": datetime.now(timezone.utc).isoformat(),
            "asset": asset,
            "side": "buy" if delta > 0 else "sell",
            "units": abs(delta),
            "price": exec_price,
            "commission": commission,
            "position_after": target_units,
        }
        self.trade_log.append(trade)
        self._save_state()

        side = "BUY" if delta > 0 else "SELL"
        print(f"  [PAPER] {side} {abs(delta):.6f} {asset} @ ${exec_price:.2f} | commission: ${commission:.2f}")


# ---------------------------------------------------------------------------
# Live Trading Bridge (HyperLiquid testnet/mainnet)
# ---------------------------------------------------------------------------

class LiveTrader:
    """Connects to HyperLiquid via SDK for real order execution."""

    def __init__(self, testnet: bool = True):
        import eth_account
        from hyperliquid.exchange import Exchange
        from hyperliquid.info import Info
        from hyperliquid.utils import constants

        api_url = constants.TESTNET_API_URL if testnet else constants.MAINNET_API_URL
        mode_str = "TESTNET" if testnet else "*** MAINNET ***"
        print(f"Connecting to HyperLiquid {mode_str}...")

        # Load private key from environment
        private_key = os.environ.get("HL_PRIVATE_KEY")
        if not private_key:
            raise ValueError(
                "Set HL_PRIVATE_KEY environment variable. "
                "For testnet, generate a test wallet."
            )

        self.wallet = eth_account.Account.from_key(private_key)
        self.exchange = Exchange(self.wallet, api_url)
        self.info = Info(api_url, skip_ws=True)
        self.testnet = testnet

        # Get account state
        user_state = self.info.user_state(self.wallet.address)
        print(f"  Account: {self.wallet.address}")
        print(f"  Margin: ${float(user_state.get('marginSummary', {}).get('accountValue', 0)):.2f}")

    def get_positions(self) -> dict[str, float]:
        """Get current positions as {asset: units}."""
        user_state = self.info.user_state(self.wallet.address)
        positions = {}
        for pos in user_state.get("assetPositions", []):
            p = pos.get("position", {})
            coin = p.get("coin", "")
            szi = float(p.get("szi", 0))
            if szi != 0:
                positions[coin] = szi
        return positions

    def execute_trade(self, asset: str, target_units: float, price: float):
        current_positions = self.get_positions()
        current_units = current_positions.get(asset, 0.0)
        delta = target_units - current_units

        if abs(delta) * price < 10:  # min $10
            return

        is_buy = delta > 0
        size = abs(delta)

        # Use market order with slippage protection
        slippage = 0.01  # 1% max slippage
        limit_price = price * (1 + slippage) if is_buy else price * (1 - slippage)

        if DRY_RUN:
            side = "BUY" if is_buy else "SELL"
            print(f"  [DRY RUN] Would {side} {size:.6f} {asset} @ ~${price:.2f}")
            return

        result = self.exchange.order(
            asset, is_buy, size, limit_price,
            {"limit": {"tif": "Ioc"}}  # Immediate or Cancel
        )
        print(f"  [LIVE] Order result: {result}")


# ---------------------------------------------------------------------------
# Main execution loop
# ---------------------------------------------------------------------------

def get_current_prices() -> dict[str, float]:
    """Get latest prices from cached data."""
    prices = {}
    for asset in ASSETS:
        try:
            df = load_market_data(asset)
            prices[asset] = df["close"].iloc[-1]
        except Exception:
            pass
    return prices


def get_live_prices() -> dict[str, float]:
    """Get real-time prices from HyperLiquid API."""
    from hyperliquid.info import Info
    from hyperliquid.utils import constants
    info = Info(constants.MAINNET_API_URL, skip_ws=True)

    prices = {}
    for asset in ASSETS:
        try:
            # Get latest candle
            end_time = int(time.time() * 1000)
            start_time = end_time - 3600_000  # last hour
            candles = info.candles_snapshot(asset, "1h", start_time, end_time)
            if candles:
                prices[asset] = float(candles[-1]["c"])
        except Exception as e:
            print(f"  Warning: Could not get price for {asset}: {e}")
    return prices


def run_strategy_live(trader, once: bool = False, interval: int = CHECK_INTERVAL):
    """Main loop: fetch data, generate signals, execute trades."""
    import strategy

    daily_start_equity = None
    iteration = 0

    while True:
        iteration += 1
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        print(f"\n{'='*60}")
        print(f"[{ts}] Iteration #{iteration}")
        print(f"{'='*60}")

        try:
            # Refresh data (5m candles for live trading)
            print(f"Downloading latest data ({LIVE_CANDLE_INTERVAL} candles)...")
            download_all_data(lookback_days=min(LOOKBACK_DAYS, 30),
                              max_age_hours=0, interval=LIVE_CANDLE_INTERVAL)

            # Load features
            features = {}
            prices = {}
            for asset in ASSETS:
                df = load_market_data(asset, interval=LIVE_CANDLE_INTERVAL)
                df_feat = compute_base_features(df)
                df_feat = df_feat.set_index("timestamp")
                features[asset] = df_feat
                prices[asset] = df_feat["close"].iloc[-1]

            # Generate signals
            signals_df = strategy.generate_signals(features)
            latest_signals = signals_df.iloc[-1]

            print(f"\nLatest signals:")
            for asset in ASSETS:
                sig = latest_signals.get(asset, 0.0)
                direction = "LONG" if sig > 0.01 else ("SHORT" if sig < -0.01 else "FLAT")
                print(f"  {asset}: {sig:+.4f} ({direction})")

            # Check daily loss kill switch
            equity = trader.get_equity(prices) if isinstance(trader, PaperTrader) else INITIAL_CAPITAL
            if daily_start_equity is None:
                daily_start_equity = equity

            daily_pnl_pct = (equity - daily_start_equity) / daily_start_equity * 100
            print(f"\nEquity: ${equity:,.2f} | Daily PnL: {daily_pnl_pct:+.2f}%")

            if daily_pnl_pct < -MAX_DAILY_LOSS_PCT:
                print(f"\n*** KILL SWITCH: Daily loss ({daily_pnl_pct:.1f}%) exceeds limit ({MAX_DAILY_LOSS_PCT}%). Flattening all positions. ***")
                for asset in ASSETS:
                    trader.execute_trade(asset, 0.0, prices.get(asset, 0))
                if once:
                    break
                print(f"Sleeping {interval}s...")
                time.sleep(interval)
                continue

            # Execute trades
            for asset in ASSETS:
                sig = latest_signals.get(asset, 0.0)
                price = prices.get(asset, 0)
                if price == 0:
                    continue

                target_notional = sig * equity * MAX_LEVERAGE
                # Cap notional
                target_notional = np.clip(target_notional, -MAX_POSITION_USD, MAX_POSITION_USD)
                target_units = target_notional / price

                trader.execute_trade(asset, target_units, price)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        if once:
            break

        print(f"\nSleeping {interval}s until next evaluation...")
        time.sleep(interval)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute trading strategy on HyperLiquid")
    parser.add_argument("--live", action="store_true", help="Use live trading (default: paper)")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=CHECK_INTERVAL,
                        help=f"Seconds between evaluations (default: {CHECK_INTERVAL})")
    parser.add_argument("--dry-run", action="store_true", default=True,
                        help="Log trades without executing (default: True)")
    args = parser.parse_args()

    print("=" * 60)
    if args.live:
        print("MODE: *** LIVE TRADING ***")
        print("WARNING: Real funds will be used!")
        confirm = input("Type 'YES I UNDERSTAND' to continue: ")
        if confirm != "YES I UNDERSTAND":
            print("Aborted.")
            sys.exit(0)
        trader = LiveTrader(testnet=False)
    else:
        print("MODE: PAPER TRADING (no real funds)")
        print("=" * 60)
        trader = PaperTrader()

    run_strategy_live(trader, once=args.once, interval=args.interval)
