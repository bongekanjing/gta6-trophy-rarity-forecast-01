"""gta6-trophy-rarity-forecast-01: forecast Exophase GTA VI trophy earn rates.

Model (my own design, standard library only, deterministic, offline):

The page's earn rate for a trophy is a ratio: earners / players-tracked.
Both sides move for different reasons, so I forecast them separately and
divide, instead of extrapolating the ratio directly:

  * players tracked N(t): the number of players Exophase tracks grows as
    more players sync profiles. I fit log N against log-time
    x = ln(1 + hours since the release day began) by least squares and
    extrapolate to the target. Log-time is self-damping: it grows slowly,
    so a 4.5-day extrapolation does not explode.
  * earners E_t(t) per trophy: the cumulative count of tracked players who
    hold that trophy. I fit log(E_t + 0.5) against the same log-time, per
    trophy, and extrapolate. Early story trophies gain earners fast, late
    or hard trophies barely move; the per-trophy fit lets each trophy tell
    its own story instead of imposing one curve on all of them.
  * median rate = 100 * E_t(target) / N(target), clamped to 0.01..99.99.
    A trophy held by everyone (rate 100%) has E = N at every read, so its
    forecast stays at 100% automatically; a trophy nobody holds stays
    near 0%.

Stale-page refresh: Exophase refreshes a game's counts on its own
schedule, hours apart, so the last read can be stale -- on launch night
the page sat unchanged at 638 players for hours, then jumped to 3,868.
When the whole history shows players-tracked perfectly flat, the fitted
N slope is zero and would wrongly say nobody new arrives. In that case
I fall back to a launch-growth prior for N, anchored at the first read:
N grows like exp(0.9 * dx) in log-time, which reproduces the documented
launch-night jump (638 -> ~3,900 over the first ~5 hours). When the
history does show N moving, the fitted slope is used instead and no
prior is applied, so this never double-counts on the final case, whose
history is full of real movement.

Slopes are damped (x0.8 for earners, x0.85 for players) and clamped, so
noisy short histories cannot throw the median far, and the median's
logit move from the last read is clamped to [-4, +4.5].

Spread (logit sd s): a random-walk term 0.5*sqrt(horizon in days), plus
the per-trophy fit's residual spread scaled by how far the target sits
outside the observed log-time range, plus 0.25 * |log(N_target/N_last)|
for refresh uncertainty. A floor that grows with horizon keeps the band
honest on long horizons; the total is capped at 2.2 logit units.
Quantiles are inv-logit(median_logit + z_tau * s).

Edge cases: rates at 0% or 100% are clamped before any logit; reads
that lack a trophy (null) are skipped for that trophy; a flat history
falls back to the stale-page prior with a widened spread; a history of
a single read uses slope 0 plus the prior; missing history falls back
to a 1% median with a wide spread. No clock, no randomness, no network:
the same case always yields the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves the repo root off sys.path; put it back first.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Damping applied to fitted slopes (1.0 = trust the fit fully).
EARNERS_DAMP = 0.8
PLAYERS_DAMP = 0.85
# Launch-growth prior for N when the history is perfectly flat (see docstring).
FLAT_N_PRIOR_SLOPE = 0.9
# Small allowance for brand-new players earning a second trophy quickly.
NEW_PLAYER_EARN_RATE = 0.0005
# Median logit move allowed from the last read, either way.
MEDIAN_LOGIT_MIN = -4.0
MEDIAN_LOGIT_MAX = 4.5
# Spread bounds, in logit units.
SPREAD_FLOOR_BASE = 0.55
SPREAD_CAP = 2.2


def clamp_rate(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up (same as the starter's rounding)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hours_since_release(at: datetime, release_day: datetime) -> float:
    """Hours from the release day's 00:00 UTC; may be slightly negative."""
    return (at - release_day).total_seconds() / 3600.0


def log_time(hours: float) -> float:
    """Log-time coordinate: ln(1 + hours), floored so it stays defined."""
    return math.log(max(0.05, 1.0 + hours))


def fit_line(xs, ys):
    """Least-squares y ~ a + b*x. Returns (slope, intercept, resid_sd, x_mean, sxx, n).

    Returns slope 0 and resid_sd 0 when there are too few points or no
    spread in x, so callers fall back to persistence.
    """
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0, (xs[0] if xs else 0.0), 0.0, n
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, y_mean, 0.0, x_mean, sxx, n
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n > 2:
        resid_sd = math.sqrt(
            sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / (n - 2)
        )
    else:
        resid_sd = 0.0
    return slope, intercept, resid_sd, x_mean, sxx, n


def forecast_players(xs, ns, x_first, x_last, x_target):
    """Forecast players-tracked at the target.

    When N moved during the history, extrapolate its fitted log-time
    slope (damped). When N never moved, the page is stale: use the
    launch-growth prior anchored at the first read instead.
    """
    if not ns:
        return 1.0, 0.0
    n_last = ns[-1]
    if max(ns) - min(ns) > 0:
        log_ns = [math.log(max(1.0, v)) for v in ns]
        slope, _, _, _, _, _ = fit_line(xs, log_ns)
        slope = max(0.0, min(2.0, slope * PLAYERS_DAMP))
        n_target = n_last * math.exp(slope * (x_target - x_last))
        return max(1.0, n_target), slope
    # Flat history: stale page. Prior growth since the first read.
    n_target = ns[0] * math.exp(FLAT_N_PRIOR_SLOPE * (x_target - x_first))
    return max(1.0, n_target), FLAT_N_PRIOR_SLOPE


def forecast_case(case: dict, name: str) -> dict:
    if case.get("levels", LEVELS) != LEVELS:
        raise ValueError(f"{name} asks for levels {case.get('levels')}, not {LEVELS}")
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = case["horizon_minutes"] / 1440.0
    if not history:
        # No data at all: a flat 1% median with a very wide band.
        return {"trophies": [
            {"id": t["id"], "q": [round4(inv_logit(logit(1.0) + z * 2.0)) for z in Z]}
            for t in trophies
        ]}

    release_raw = (case.get("game") or {}).get("release_date", "2026-11-19")
    release_day = datetime.fromisoformat(release_raw).replace(tzinfo=timezone.utc)
    target = parse_time(case["target"])

    xs = []
    ns = []
    for read in history:
        xs.append(log_time(hours_since_release(parse_time(read["at"]), release_day)))
        ns.append(float(read["players_tracked"]))
    x_first, x_last = xs[0], xs[-1]
    x_target = log_time(hours_since_release(target, release_day))
    n_last = ns[-1]
    n_target, _ = forecast_players(xs, ns, x_first, x_last, x_target)
    n_growth = max(1.0, n_target / max(1.0, n_last))

    # Spread pieces shared by every trophy.
    rw = 0.5 * math.sqrt(horizon_days)
    refresh_unc = 0.25 * math.log(n_growth)
    spread_floor = SPREAD_FLOOR_BASE + 0.15 * horizon_days

    rows = []
    for i, trophy in enumerate(trophies):
        # Per-trophy earners series, skipping reads that lack this trophy.
        pts_x, pts_e = [], []
        last_rate = None
        for j, read in enumerate(history):
            rate = read["rates"][i]
            if rate is None:
                continue
            last_rate = rate
            pts_x.append(xs[j])
            pts_e.append(max(0.0, rate * float(read["players_tracked"]) / 100.0))
        if last_rate is None:
            last_rate = 1.0
            e_last = 0.01 * n_last
            pts_x, pts_e = [x_last], [e_last]
        else:
            e_last = pts_e[-1]

        # Fit log(E + 0.5) on log-time; flat or single-point -> slope 0.
        log_e = [math.log(e + 0.5) for e in pts_e]
        slope, _, resid_sd, x_mean, sxx, _ = fit_line(pts_x, log_e)
        slope = max(-0.5, min(3.0, slope * EARNERS_DAMP))
        e_target = (e_last + 0.5) * math.exp(slope * (x_target - x_last)) - 0.5
        # Brand-new players earn a little on top of the trend.
        e_target += NEW_PLAYER_EARN_RATE * max(0.0, n_target - n_last)
        if last_rate >= 99.0:
            # Saturation: (nearly) every tracked player holds this trophy,
            # so its earners scale with players-tracked, not with the trend.
            # Without this, a flat history would leave its earners fixed
            # while N jumps, and the median would wrongly collapse.
            e_target = n_target * (last_rate / 100.0)
        e_target = max(0.0, min(n_target, e_target))

        median_rate = clamp_rate(100.0 * e_target / n_target)
        # Keep the median's logit move from the last read inside bounds.
        med_logit = logit(median_rate)
        last_logit = logit(last_rate)
        med_logit = min(last_logit + MEDIAN_LOGIT_MAX,
                        max(last_logit + MEDIAN_LOGIT_MIN, med_logit))
        median_rate = inv_logit(med_logit)

        # Spread: random walk + fit uncertainty + refresh uncertainty.
        leverage = 0.0
        if sxx > 1e-12:
            leverage = (x_target - x_mean) ** 2 / sxx
        fit_unc = resid_sd * math.sqrt(1.0 + leverage)
        spread = math.sqrt(rw * rw + fit_unc * fit_unc + refresh_unc * refresh_unc)
        spread = max(spread_floor, min(SPREAD_CAP, spread))

        qs = []
        prev = None
        for z in Z:
            q = round4(inv_logit(med_logit + z * spread))
            q = min(100.0, max(0.0, float(q)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()

