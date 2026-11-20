# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, five quantiles of its earn rate on Exophase at 2026-11-26T00:00:00Z, from the house's recording of the trophy page up to the deadline.

The model (`main.py`, pure Python 3.13 standard library, deterministic, no network): a trophy's earn rate moves for two reasons — players already tracked earn more trophies as they play (progression, pushing rates up), and Exophase refreshes its counts in jumps, each one diluting every rate toward the earn fraction of the players who just synced (dilution, pushing rates down). The program estimates both forces from the case's own history: it fits the growth of the tracked-player count and the per-trophy logit trend against log time, measures the per-refresh logit shift of every trophy from the refreshes the history shows, and projects the expected number of refreshes before the target. When the history is flat (no refresh seen yet, as in the practice cases) it falls back on the launch-night prior — a refresh every ~4.5 hours, the next one due soon after the checkpoint, new players earning almost nothing beyond the intro trophy. The spread combines the random-walk term, the refresh-to-refresh variance and the trend fit's residual.

Local checks: on the two practice cases the program runs in ~0.07s and writes valid forecasts (55 trophies, five non-decreasing quantiles each). Scored against synthetic practice answers rebuilt from the documented refresh (players 638 → 3,868, trophies earned 943 → 4,276, every rate but the 100% one falling), it scores a pinball loss of 0.23–0.38 against the house baseline's 1.45–2.00. Against a simulated final-like case (cohort model calibrated to the launch-night rates) it scores 0.38 against the baseline's 0.47. The real practice answers are not published in the kit, so these are local estimates, not server grades.

## Run it

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=204hjvsp&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
