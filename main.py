"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five
quantiles of its earn rate on Exophase at the case's target, from the house's
recording of the trophy page up to the case's checkpoint.

The model, in one paragraph. A trophy's earn rate moves for two reasons:
players already tracked earn more trophies as they play (progression, which
pushes rates up), and Exophase refreshes its counts in jumps, each one
diluting every rate toward the earn fraction of the players who just synced
(dilution, which pushes rates down, hard: on launch night one refresh took
the tracked players from 638 to 3,868 and the non-intro trophies' total
earned from 305 to 408, so their average rate fell to about 12% of what it
was). This program estimates both forces from the case's own history and
forecasts in earned-count space, where they separate cleanly: it fits the
growth of the tracked-player count and the per-trophy logit trend against
log time, measures the per-refresh logit shift of every trophy from the
refreshes the history actually shows, and projects the expected number of
refreshes before the target. When the history is flat (no refresh seen yet,
as in the practice cases) it falls back on the launch-night prior: a refresh
every ~4.5 hours, the next one due soon after the checkpoint, new players
earning almost nothing beyond the intro trophy. The spread combines the
random-walk term, the refresh-to-refresh variance and the trend fit's
residual, so a trophy whose history jumps around gets a wide band and a
steady one a narrow one.

Standard library only. Deterministic: no randomness, no clock, no network.
Reads every <id>.in in the input directory and writes <id>.out per case.
"""

import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# Launch day began 2026-11-19T00:00:00Z; all times are hours after it.
LAUNCH_EPOCH = 1795123200  # 2026-11-19T00:00:00Z as a Unix timestamp.

# Priors from the launch night the competition itself documents, used only
# when the case's history shows no refresh at all:
#   - Exophase refreshed the game's counts about 4.5 hours after the first
#     read (23:50Z -> between 04:00Z and 04:30Z).
#   - That refresh multiplied the tracked players by about 6 (638 -> 3,868)
#     while the trophies beyond the intro one gained almost nothing: the
#     new players' earn fraction is ~0.02% a trophy.
REFRESH_CADENCE_H = 4.5
FIRST_REFRESH_FACTOR = 6.0
NEW_COHORT_RATE = 0.0002  # percent, a trophy a fresh player holds: ~0.02%
RW_SIGMA = 0.5  # logit units per square-root day, the baseline's term.


def clamp_pct(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x):
    """Round to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    """ISO 8601 UTC -> hours since LAUNCH_EPOCH. Handles the trailing Z."""
    txt = s.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    from datetime import datetime

    dt = datetime.fromisoformat(txt)
    return (dt.timestamp() - LAUNCH_EPOCH) / 3600.0


def ols(xs, ys):
    """Least squares line. Returns (slope, intercept, resid_std, n)."""
    n = len(xs)
    if n < 2:
        return 0.0, (ys[0] if ys else 0.0), 0.0, n
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my, 0.0, n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    resid = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    var = sum(r * r for r in resid) / max(1, n - 2)
    return slope, intercept, math.sqrt(var), n


def median(vals):
    v = sorted(vals)
    m = len(v)
    if m == 0:
        return 0.0
    return v[m // 2] if m % 2 else 0.5 * (v[m // 2 - 1] + v[m // 2])


class History:
    """One case's reads, parsed into per-trophy series and refresh jumps."""

    def __init__(self, case):
        self.case = case
        self.reads = []
        for read in case["history"]:
            t = parse_time(read["at"])
            n = read["players_tracked"]
            rates = read["rates"]
            self.reads.append((t, n, rates))
        self.reads.sort(key=lambda r: r[0])
        self.n_trophies = len(case["trophies"])
        self.t_last, self.n_last, self.rates_last = self.reads[-1]
        self.x_last = math.log1p(max(0.0, self.t_last))
        # Refresh jumps: consecutive reads where the player count or any
        # rate moved. Each jump records (gap_hours, factor, per-trophy
        # logit shifts).
        self.jumps = []
        for (t0, n0, r0), (t1, n1, r1) in zip(self.reads, self.reads[1:]):
            moved = abs(n1 - n0) > 0.01 * max(1, n0)
            if not moved:
                for a, b in zip(r0, r1):
                    if a is not None and b is not None and abs(a - b) > 0.005:
                        moved = True
                        break
            if not moved:
                continue
            shifts = []
            for a, b in zip(r0, r1):
                if a is None or b is None:
                    shifts.append(None)
                else:
                    shifts.append(logit(b) - logit(a))
            self.jumps.append(
                {"gap": t1 - t0, "factor": n1 / max(1, n0), "shifts": shifts}
            )
        if self.jumps:
            self.cadence = median([j["gap"] for j in self.jumps])
            self.cadence = min(max(self.cadence, 0.5), 24.0)
            self.last_jump_t = self.reads[-1][0]
            for i in range(len(self.reads) - 1, 0, -1):
                if i - 1 < len(self.jumps) and self.jumps[i - 1]["gap"] > 0:
                    self.last_jump_t = self.reads[i][0]
                    break
            # The time of the last jump is the later read of the last jump.
            self.last_jump_t = self._last_jump_time()
        else:
            self.cadence = REFRESH_CADENCE_H
            # No refresh seen yet. The last refresh happened at an unknown
            # time before the first read; under a uniform phase it was on
            # average half a cadence earlier, so the next one is due sooner
            # than the time since the first read alone suggests.
            self.last_jump_t = self.reads[0][0] - self.cadence / 2.0
        self.elapsed_since_jump = max(0.0, self.t_last - self.last_jump_t)

    def _last_jump_time(self):
        # Walk the reads; a jump's later read is the one after a moved pair.
        # Recompute directly: the jumps list is in read order, so the last
        # jump ends at the read index of the last moved pair + 1.
        moved_idx = []
        for i in range(len(self.reads) - 1):
            t0, n0, r0 = self.reads[i]
            t1, n1, r1 = self.reads[i + 1]
            moved = abs(n1 - n0) > 0.01 * max(1, n0)
            if not moved:
                for a, b in zip(r0, r1):
                    if a is not None and b is not None and abs(a - b) > 0.005:
                        moved = True
                        break
            if moved:
                moved_idx.append(i + 1)
        return self.reads[moved_idx[-1]][0] if moved_idx else self.reads[0][0]

    def series(self, j):
        """(x, logit rate) points for trophy j, oldest first."""
        pts = []
        for t, _n, rates in self.reads:
            r = rates[j]
            if r is None:
                continue
            pts.append((math.log1p(max(0.0, t)), logit(r)))
        return pts


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    if not case["history"]:
        raise ValueError("case has no history")
    hist = History(case)
    horizon_h = max(1.0 / 60.0, case["horizon_minutes"] / 60.0)
    horizon_d = case["horizon_minutes"] / 1440.0
    x_target = math.log1p(max(0.0, hist.t_last + horizon_h))
    dx = x_target - hist.x_last

    # --- tracked-player growth: OLS of ln N on ln(1 + hours) ------------
    xs_n, ys_n = [], []
    seen_n = set()
    for t, n, _r in hist.reads:
        key = (round(t, 3), n)
        if key in seen_n:
            continue
        seen_n.add(key)
        xs_n.append(math.log1p(max(0.0, t)))
        ys_n.append(math.log(max(1, n)))
    slope_n, icept_n, _res_n, _nn = ols(xs_n, ys_n)
    slope_n = min(max(slope_n, 0.0), 3.0)
    ln_n_target = icept_n + slope_n * x_target
    n_target = math.exp(ln_n_target)
    n_target = min(max(n_target, hist.n_last), hist.n_last * 20.0)
    growth_factor = n_target / max(1, hist.n_last)

    # --- expected refreshes before the target ----------------------------
    wait = max(0.25, hist.cadence - hist.elapsed_since_jump)
    if horizon_h <= wait:
        lam = 0.5 * horizon_h / wait
    else:
        lam = 1.0 + (horizon_h - wait) / hist.cadence

    # --- per-refresh logit shifts, from the jumps the history shows ------
    all_shifts = []
    for j in hist.jumps:
        all_shifts.extend(s for s in j["shifts"] if s is not None)
    global_shift = median(all_shifts) if all_shifts else None
    if all_shifts:
        gm = sum(all_shifts) / len(all_shifts)
        var_shift = sum((s - gm) ** 2 for s in all_shifts) / len(all_shifts)
    else:
        var_shift = 0.64  # prior: refreshes move a trophy ~0.8 logits, 1 sd
    past_factor = median([j["factor"] for j in hist.jumps]) if hist.jumps else 1.0
    future_factor = growth_factor ** (1.0 / max(1.0, lam)) if lam > 0 else 1.0
    if future_factor <= 1.01:
        future_factor = past_factor
    shift_scale = 1.0
    if past_factor > 1.01 and future_factor > 1.01:
        shift_scale = math.log(future_factor) / math.log(past_factor)
        shift_scale = min(max(shift_scale, 0.05), 2.0)

    # --- cross-trophy fallback rate for trophies with no last rate -------
    # Exclude the intro trophy (100%): a missing rate is a normal trophy,
    # not the one everybody holds.
    last_rates = [r for r in hist.rates_last if r is not None and r < 99.99]
    fallback_rate = median(last_rates) if last_rates else 1.0

    rows = []
    for j, trophy in enumerate(case["trophies"]):
        r_last = hist.rates_last[j]
        if r_last is None:
            r_last = fallback_rate
        l_last = logit(r_last)
        is_intro = r_last >= 99.99

        # Trend: OLS of logit rate on log time over the last 24 hours.
        pts = hist.series(j)
        recent = [(x, y) for x, y in pts if x >= hist.x_last - math.log1p(24.0)]
        if len(recent) < 3:
            recent = pts
        slope, _icept, resid, npts = ols([p[0] for p in recent], [p[1] for p in recent])
        if npts < 3 or max(p[0] for p in recent) - min(p[0] for p in recent) < 1e-6:
            slope, resid = 0.0, 0.3
        shrink = npts / (npts + 3.0)
        # The fitted window spans a narrow log-time range and the target
        # sits about twice that range beyond it, so shrink the extrapolated
        # shift further: slope noise is amplified by the extrapolation.
        trend_shift = 0.7 * shrink * slope * dx
        trend_shift = min(max(trend_shift, -2.5), 2.5)

        # Refresh-process shift: the per-refresh logit move, scaled for
        # how big future refreshes will be, times the refreshes expected.
        if hist.jumps:
            raw = [jj["shifts"][j] for jj in hist.jumps if jj["shifts"][j] is not None]
            if raw:
                per_refresh = median(raw)
                # Shrink the trophy's own shift toward the global one.
                per_refresh = (len(raw) * per_refresh + 3.0 * global_shift) / (len(raw) + 3.0)
            else:
                per_refresh = global_shift
            refresh_shift = lam * per_refresh * shift_scale
        else:
            # No refresh seen yet: the launch-night prior. A refresh takes
            # the rate to (earned + new players x ~nothing) / new count.
            e_last = r_last * hist.n_last / 100.0
            q_new = 1.0 if is_intro else NEW_COHORT_RATE
            n_after = hist.n_last * FIRST_REFRESH_FACTOR
            r_after = (e_last + (n_after - hist.n_last) * q_new / 100.0) / n_after * 100.0
            prior_shift = logit(r_after) - l_last
            refresh_shift = lam * prior_shift
        refresh_shift = min(max(refresh_shift, -3.0), 3.0)

        # Median: average the two estimators of the net shift when both
        # speak; a flat history's trend is zero, so the refresh prior rules.
        if hist.jumps:
            shift = 0.5 * trend_shift + 0.5 * refresh_shift
        else:
            shift = refresh_shift
        shift = min(max(shift, -2.5), 2.5)

        # Spread: random walk + refresh variance + trend-fit residual.
        sig_rw = RW_SIGMA * math.sqrt(horizon_d)
        sig_jump = math.sqrt(max(0.0, lam) * var_shift)
        sig_fit = resid if resid > 0 else 0.3
        sigma = math.sqrt(sig_rw ** 2 + sig_jump ** 2 + sig_fit ** 2)
        sigma = min(max(sigma, 0.4), 2.5)

        if is_intro:
            # Everyone tracked holds the intro trophy; it stays ~100%.
            q = [99.0, 99.9, 99.99, 99.99, 99.99]
        else:
            l_med = l_last + shift
            q = []
            for z in Z:
                v = inv_logit(l_med + z * sigma)
                q.append(round4(clamp_pct(v)))
            # Keep the row non-decreasing after rounding.
            for k in range(1, 5):
                if q[k] < q[k - 1]:
                    q[k] = q[k - 1]
        rows.append({"id": trophy["id"], "q": q})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
