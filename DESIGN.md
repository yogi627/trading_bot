# Event-Driven Trading Bot Design

## Scope and configuration

Python 3.12 event-driven trading bot with `backtest`, `paper`, and `live` modes. Default mode is `paper`. Market/provider, instruments, candle interval, capital, risk limits, queue capacities, feed-staleness thresholds, execution behavior, and credentials are configuration, not code constants. Credentials are read from environment variables and never persisted in configuration or logs.

The initial provider is selected by configuration from Binance, Zerodha Kite, Angel One, or Alpaca. Provider adapters implement common feed and broker interfaces. Modules are implemented one checkpoint at a time, with review between checkpoints.

## Proposed folder structure

```text
.github/
    copilot-instructions.md        # project guidance
config.yaml
.env.example
requirements.txt
README.md
pyproject.toml
DESIGN.md

src/trading_bot/
  __init__.py
  __main__.py                    # CLI and mode selection
  config.py                      # validated configuration
  domain.py                      # immutable messages, enums, scaled values
  clock.py                       # injectable monotonic and wall clocks
  ports.py                       # feed, broker, journal, and clock contracts
  runtime.py                     # startup and lifecycle orchestration
  queues.py                      # bounded queue policies and backpressure
  latency.py                     # tick-to-signal/order-sent p50/p99
  data_feed/
    __init__.py
    base.py
    binance.py
    alpaca.py
    kite.py
    angel_one.py
  candle_builder.py
  indicators/
    __init__.py
    streaming.py                 # incremental indicator state
    kernels.py                   # Numba-compatible kernels
  strategy/
    __init__.py
    base.py
    ema_atr_trend.py
  risk_manager.py
  execution/
    __init__.py
    coordinator.py               # authoritative risk gate and order lifecycle
    router.py                    # idempotent broker submission and retries
    paper.py                     # PaperBroker
    live.py
    reconciliation.py
    kill_switch.py
  logging/
    __init__.py
    async_logger.py
    journal.py
  backtester/
    __init__.py
    engine.py
    simulated_broker.py

tests/
  test_config.py
  test_clock.py
  test_queues.py
  test_data_feed.py
  test_candle_builder.py
  test_indicators.py
  test_strategy.py
  test_risk_manager.py
  test_execution.py
  test_reconciliation.py
  test_kill_switch.py
  test_logging.py
  test_backtester.py
  test_runtime.py
```

## Domain and interface contracts

Internal prices and quantities are integer counts of configured `tick_size` and `lot_size` increments. Feed adapters convert external values at the boundary using an exact conversion by default; any rounding must be an explicit configured policy and applied before constructing domain messages. `price_scale` and `quantity_scale` describe only exchange wire-format conversion and never alter internal tick/lot counts. `Tick`, `Quote`, and `Candle` constructors are data-only; feed adapters validate values before constructing them.

Each currency defines a positive `money_scale`; P&L, fees, taxes, capital, and notional risk limits are signed integer counts of that currency's money-scale unit, not necessarily ISO minor units. Each instrument defines a positive `contract_multiplier` and quote currency. Configuration requires `tick_size * lot_size * contract_multiplier * money_scale` to be an exact integer and fit the signed 64-bit money representation. Fees/taxes are computed with `Decimal` off the hot path, rounded under an explicit policy to the configured money scale, then posted as integer money units.

The `.env` file is parsed with pinned `python-dotenv`; values from the real process environment override `.env` values. Credential checks run on that final merged mapping. Secrets are never retained in config objects or included in their representations, logs, or exceptions.

Configuration uses a duplicate-key-rejecting YAML safe loader. Decimal-valued config fields must be quoted strings or integers (never YAML floats). Session times must be quoted. Sessions may be `always_open: true` with an IANA timezone, or a local open/close pair; an open time later than close represents a session crossing midnight. `daily_reset` uses the instrument session timezone, including its DST rules. `AppConfig.timeframe_ns` exposes the parsed interval in nanoseconds.

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from collections.abc import Mapping
from typing import AsyncIterator, Protocol
import asyncio


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"


class Intent(str, Enum):
    ENTRY = "entry"
    EXIT = "exit"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderStatus(str, Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    CANCEL_PENDING = "cancel_pending"
    CANCELED = "canceled"


@dataclass(slots=True, frozen=True, kw_only=True)
class Tick:
    instrument: str
    event_time_ns: int
    received_monotonic_ns: int
    price_ticks: int
    size_lots: int
    sequence: int | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class Quote:
    instrument: str
    bid_ticks: int
    ask_ticks: int
    bid_size_lots: int
    ask_size_lots: int
    event_time_ns: int
    received_monotonic_ns: int


@dataclass(slots=True, frozen=True, kw_only=True)
class Candle:
    instrument: str
    start_time_ns: int
    end_time_ns: int
    open_ticks: int
    high_ticks: int
    low_ticks: int
    close_ticks: int
    volume_lots: int


@dataclass(slots=True, frozen=True, kw_only=True)
class Signal:
    instrument: str
    side: Side
    intent: Intent
    quantity_lots: int
    stop_price_ticks: int | None
    strategy_id: str
    event_time_ns: int


@dataclass(slots=True, frozen=True, kw_only=True)
class OrderRequest:
    instrument: str
    side: Side
    quantity_lots: int
    client_order_id: str
    intent: Intent
    order_type: OrderType
    reduce_only: bool
    limit_price_ticks: int | None = None
    stop_price_ticks: int | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class RiskDecision:
    approved: bool
    reason: str | None = None
    order: OrderRequest | None = None

    # Approved decisions require an order; rejected decisions require a reason.


@dataclass(slots=True, frozen=True, kw_only=True)
class OrderUpdate:
    """One order-state event; quantity and fee are this event's fill delta only.

    Each fill has a unique, non-empty ``fill_id`` and is deduplicated by the
    pair ``(client_order_id, fill_id)``. Fees are per-fill, not cumulative.
    PARTIALLY_FILLED/FILLED statuses require positive fill quantity, fill ID,
    and fill price. Other statuses have zero fill quantity and no fill fields.
    """

    client_order_id: str
    status: OrderStatus
    filled_quantity_lots: int
    fill_id: str | None = None
    fill_price_ticks: int | None = None
    fee_money_units: int = 0
    reason: str | None = None


@dataclass(slots=True, frozen=True, kw_only=True)
class Position:
    """Signed lots: positive is long, negative is short; average price rounds half-even."""
    instrument: str
    quantity_lots: int
    average_price_ticks: int


@dataclass(slots=True, frozen=True, kw_only=True)
class PendingExposure:
    client_order_id: str
    instrument: str
    side: Side
    remaining_quantity_lots: int
    limit_price_ticks: int | None
    worst_case_price_ticks: int


@dataclass(slots=True, frozen=True, kw_only=True)
class RiskSnapshot:
    positions: tuple[Position, ...]
    pending_orders: tuple[PendingExposure, ...]
    realized_pnl_money_units: int
    unrealized_pnl_money_units: int
    daily_loss_money_units: int
    open_trade_count: int
    orders_in_rate_window: int
    reference_prices: Mapping[str, int]  # published as a read-only mapping
    published_monotonic_ns: int
    kill_latched: bool
    trading_day_id: str
    version: int


@dataclass(slots=True, frozen=True, kw_only=True)
class JournalEvent:
    kind: str
    wall_time: datetime
    payload: object


class Strategy(ABC):
    @abstractmethod
    def on_tick(self, tick: Tick) -> Signal | None: ...

    @abstractmethod
    def on_candle(self, candle: Candle) -> Signal | None: ...


class DataFeed(ABC):
    @abstractmethod
    async def run(
        self, output: asyncio.Queue[Tick], stop: asyncio.Event
    ) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class CandleBuilder(Protocol):
    def on_tick(self, tick: Tick) -> Candle | None: ...


class RiskManager(Protocol):
    def check(self, signal: Signal, snapshot: RiskSnapshot) -> RiskDecision: ...


class Broker(Protocol):
    async def submit(self, order: OrderRequest) -> OrderUpdate: ...
    async def updates(self) -> AsyncIterator[OrderUpdate]: ...
    async def positions(self) -> tuple[Position, ...]: ...
    async def open_orders(self) -> tuple[PendingExposure, ...]: ...
    async def cancel(self, client_order_id: str) -> None: ...
    async def cancel_all(self) -> None: ...


class TradeJournal(Protocol):
    async def append(self, event: JournalEvent) -> None: ...
```

## State ownership and risk

The hot path reads an immutable, atomically published `RiskSnapshot`; it does not mutate positions, P&L, pending exposure, or order state. It performs no broker query, logging, database write, JSON construction, or Decimal fee/tax computation. It validates/deduplicates input, updates preallocated candle and indicator state, evaluates strategy, reads the snapshot for a fast pre-check, and hands off an order candidate.

The execution coordinator is the authoritative serialized risk gate. Before broker submission it rechecks current limits and reserves the candidate as pending exposure, publishing a new snapshot. Pending/in-flight orders count toward position-size, capital, and open-trade limits using conservative worst-case prices. Fill events update positions and P&L on the execution side; remaining pending quantity is adjusted on partial fills and removed on final fill, reject, or confirmed cancel. Risk snapshots are published atomically so the tick path only reads a consistent version. A fail-closed or stale snapshot blocks order creation.

Risk checks include per-instrument maximum position lots and notional (notional in the instrument's quote-currency money units), daily loss, open trades, order rate, price sanity band, duplicate client order ID, pending exposure, and latched kill state. Repeated broker rejections trigger the kill switch. Daily loss is `max(0, -(realized_pnl_money_units + unrealized_pnl_money_units))` in account-currency money units for the configured trading day; it includes realized and marked unrealized P&L, net of posted fees/taxes and converted to the account currency using the execution-side FX mark. Execution-side fills and marks update these values, not broker calls from the hot path.

An `EXIT` signal requests position reduction. Every generated order declares `reduce_only`. A reduce-only order may reduce or close its current position but must be rejected if its side and quantity would cross through zero or increase exposure. This check uses the execution-owned current position and is repeated immediately before submission.

`OrderType.STOP` semantics are policy-controlled. Stops are bot-side by default in paper and backtest modes: the simulator watches quotes and submits/executes when the trigger is reached. Live mode may configure bot-side or exchange-side stops; the adapter must report which behavior is in effect and must not silently convert one into the other.

`OrderUpdate` is an event stream, not a cumulative snapshot. A fill update reports only its fill quantity delta, carries a unique `fill_id`, and reports fees for that fill only. The execution coordinator deduplicates by fill ID before changing position or P&L state. Non-fill updates carry no fill ID and a zero fill delta.

## Queue policy and hot path

All ordinary work queues have configured finite capacity:

- **Ticks:** bounded. Feed ingestion may coalesce ticks by instrument or drop stale ticks under configured policy; sequence gaps and drops are counted. Duplicate ticks are discarded. Candle aggregation receives the freshest permitted tick stream.
- **Order candidates:** bounded. A full order queue immediately latches the kill switch and blocks new orders; it never silently drops an order candidate and continues trading.
- **Broker order updates, fills, and cancels:** bounded reliable delivery. Producers apply backpressure (`await put`/provider flow control) rather than dropping events. If the provider cannot be backpressured and reliable delivery cannot be guaranteed, execution halts and reconciles before resuming.
- **Journal/log events:** bounded off-hot-path queues with an explicit durability policy. They cannot block tick processing; journal failure is surfaced to lifecycle supervision and can fail closed according to configured policy.

The only tick-to-order performance measurement is monotonic time from tick receipt through signal generation to order submission start/completion. Latency histograms report p50 and p99. The synchronous tick handler avoids per-tick history allocations and stays free of I/O. Async queue handoffs and broker network time are measured separately from internal processing.

## Kill switch

The kill switch is a latched, idempotent control path that bypasses normal queues. Activation first blocks all new order candidates, then directly invokes the broker's emergency `cancel_all()` path through a dedicated control task/API, without waiting for the ordinary order queue. It records the trigger and outcome off the hot path. Optional flattening is separately configured, uses current execution-owned state and normal risk validation, and is never implicit. Trading remains halted until an explicit recovery procedure completes reconciliation and clears the latch. Triggers include max daily loss, repeated rejections, stale feed, full order queue, inconsistent state, and unrecoverable broker disconnect.

## Lifecycle and failure handling

1. Load and validate configuration and secrets; reject invalid scales, limits, provider settings, or mode/credential conflicts.
2. Construct bounded queues, clock, state store, journal, strategy, risk manager, and broker/feed adapters.
3. Start execution event handling and perform startup reconciliation against broker positions and open orders. Seed positions and pending exposure from the reconciled result. Any unexplained mismatch, duplicate client ID, or query failure fails closed.
4. Start the feed and stale-feed watchdog. The watchdog uses the injected monotonic clock; stale data latches the kill switch. Trading is enabled only after reconciliation succeeds and fresh feed data is observed.
5. Run the event-driven tick/candle/strategy path. Execution-side order handling reserves pending exposure, submits idempotently, retries only under client-order-ID guarantees, and consumes fills/rejections/cancels with backpressure.
6. On shutdown or disconnect, block new orders, cancel outstanding orders through the kill/control path, reconcile state, flush journal work, and close adapters. Unexpected state fails closed.

Partial fills update realized exposure immediately; unfilled remainder remains pending exposure. Rejected or canceled orders release only their remaining reservation after terminal confirmation. Reconnect recovery queries/reconciles outside the tick path before resuming.

## PaperBroker

`PaperBroker` implements the same `Broker` interface as live adapters and is selected by mode, not by strategy-specific branches. It fills from `Quote` events: buys cross the ask and sells cross the bid (pessimistic side of market); configurable slippage worsens execution; injected latency delays acknowledgements/fills; configured liquidity permits partial fills; seeded random rejection probability enables reproducible rejection tests. Fees and taxes are computed using `Decimal` on the simulator/execution side and emitted as scaled integer fee amounts. It tracks open orders and positions and emits the same lifecycle updates as live adapters. The simulated broker in backtests also consumes `Quote`, not midpoint-only `Tick` data.

Every non-live mode (`paper` and `backtest`) checks the final merged environment for trading credentials and fails closed if any are present. Separately named provider-scoped read-only market-data credentials are permitted in paper mode. Tests use injected environments/config and seeded random sources.

## Backtesting and clock

Backtests replay historical events through the same strategy, risk checks, execution coordinator, and `PaperBroker`/simulated broker contracts used in paper mode. The simulator applies configured fees, taxes, spread, slippage, partial fills, rejects, and latency. Event ordering is causal: a decision can only use data available at that event time; candles are emitted only when complete. No future bars, finalized candle values, or future-derived indicators are visible to earlier decisions.

The single `Clock` protocol is defined in `clock.py`. `Clock.monotonic_ns()` is used for elapsed durations, staleness, and latency; `Clock.wall_time()` supplies journal timestamps. `SystemClock.monotonic_ns()` delegates to `time.perf_counter_ns()` and `SystemClock.wall_time()` returns timezone-aware UTC. `DeterministicClock` advances a monotonic counter and UTC wall clock by the same duration; it supports forward-only `set`, `advance(timedelta)`, `advance_ns(int)`, and replay updates from event timestamps. Equal replay timestamps are valid; earlier timestamps raise. Advancing by one nanosecond updates the monotonic counter exactly even though Python `datetime` wall-time display is microsecond-resolution. Floats, booleans, and negative advances are rejected. Backtests inject independent deterministic clock instances.

Each instrument's session timezone also governs its `daily_reset` wall time, including daylight-saving transitions. Open times later than close times describe a session crossing midnight.

Market-data environment variables must contain provider-issued read-only credentials. A token that can submit, modify, or cancel orders is a trading credential and must use the trading credential class; renaming a broadly privileged token does not make it safe for paper mode.

## Module build order and tests

Each item is a separate review checkpoint. Implement only the next module, run its focused tests, summarize the result, and wait for review before proceeding.

1. **`domain.py` and `config.py`**: tick/lot internal values, currency money scales, per-instrument contract multipliers and risk caps, and validated immutable configuration. Tests cover exact money representation, decimal parsing, dotenv precedence, credential classes, duplicate YAML keys, sessions/DST, and domain invariants.
2. **`clock.py`**: `SystemClock` using `time.perf_counter_ns()` and independent `DeterministicClock` with `advance(timedelta)` / `advance_ns(int)`. Tests cover equal/earlier event timestamps, sub-microsecond nanoseconds, rejected invalid advances, UTC wall time, and instance independence.
3. **`queues.py`**: bounded queue policies. Tests: tick coalescing/drop and duplicate handling; order-queue full latches halt; reliable update queue backpressure does not drop fill/cancel events.
4. **`candle_builder.py`**: incremental aggregation with preallocated ring buffers. Tests: boundaries, out-of-order/duplicate ticks, volume/OHLC correctness, O(1)-style bounded state, configured interval handling.
5. **`indicators/streaming.py` and `kernels.py`**: incremental EMA, RSI, ATR, VWAP. Tests: reference values, warmup, reset, incremental-vs-batch parity, no full-history dependency.
6. **`strategy/base.py` and `ema_atr_trend.py`**: strategy contract and EMA crossover with ATR stop/trend filter. Tests: crossover entries/exits, filter suppression, stop calculation, no signal before warmup.
7. **`risk_manager.py`**: read-only risk evaluation against immutable snapshots. Tests: each configured limit, pending exposure, price band, duplicates, stale/invalid snapshots, fail-closed behavior.
8. **`execution/coordinator.py`, `router.py`, and `reconciliation.py`**: authoritative risk recheck, pending reservation, idempotent orders, fills, retries, startup reconciliation. Tests: reserve-before-submit, partial fills, rejection/cancel release, duplicate client IDs, mismatch blocks startup, reconnect reconciliation.
9. **`execution/paper.py` and `simulated_broker.py`**: pessimistic fills and paper safeguards. Tests: bid/ask direction, slippage, latency via fake clock, partial fills, seeded rejects, Decimal fees/taxes, live credentials raise in paper mode.
10. **`execution/kill_switch.py`**: bypass-queue stop/cancel and optional flatten control. Tests: immediate block, cancel-all despite saturated ordinary queues, idempotence, trigger coverage, explicit recovery only.
11. **`data_feed/` adapters and watchdog**: websocket lifecycle, heartbeat, exponential backoff, event validation, stale-feed detection. Tests: reconnect/backoff, heartbeat, malformed and duplicate ticks, stale watchdog latches kill switch, provider adapter contract.
12. **`logging/`**: queued logging and SQLite journal; optional Parquet backend if selected. Tests: off-hot-path enqueue, journal serialization, timestamps from wall clock, bounded failure policy, recovery/flush.
13. **`latency.py` and `runtime.py`**: metrics and full startup/shutdown lifecycle. Tests: measured tick-to-signal/order timestamps, p50/p99 aggregation, reconciliation-before-trading, watchdog lifecycle, queue saturation, graceful and fail-closed shutdown.
14. **`backtester/engine.py`**: causal replay using shared strategy/risk/execution contracts. Tests: deterministic replay, no lookahead, fees/slippage/latency, pending exposure limits, same decision results as equivalent paper event sequence.
15. **`__main__.py`, docs, and packaging**: CLI mode selection and run instructions. Tests: CLI mode defaults to paper, invalid mode rejected, config path/env handling, smoke test for backtest startup.

## Module 1 implementation checkpoint

Module 1 consists of `domain.py`, `config.py`, `config.yaml`, `.env.example`, `pyproject.toml`, `requirements.txt`, `tests/test_config.py`, and `.github/copilot-instructions.md`. The config loader validates known keys, numeric ranges, exchange wire scales, exact tick-value money scaling, sessions/timezones, per-instrument risk limits, duplicate YAML keys, and final merged credential state. It never stores secret values in the resulting configuration object. Decimal values are YAML strings or integers only; session times must be quoted.

## Module 2 implementation checkpoint

Module 2 consists only of `clock.py` and `tests/test_clock.py`. Deterministic timestamps are UTC wall times plus an integer monotonic nanosecond counter; replay event timestamps are Unix nanoseconds.

Module 3 consists only of `queues.py` and `tests/test_queues.py`, and begins only after the Module 1-2 Python 3.12 test, strict mypy, and Ruff checks pass.

## Acceptance checks

- `pytest` covers candle building, indicators, risk, and backtesting, with focused tests for queues, execution, kill switch, reconciliation, and lifecycle as modules are added.
- Static typing/linting and Python 3.12 test run are documented in the README when implementation begins.
- Performance target is under 1 ms internal processing per tick, excluding network. Benchmark reports should distinguish tick processing, queue delay, and broker/network latency; the target is measured on a documented workload and hardware, not assumed from design alone.
- Live mode is not enabled until provider-specific credentials, reconciliation, kill-switch behavior, risk tests, and paper-mode acceptance checks pass.
