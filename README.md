# gta6-trophy-rarity-forecast-01

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/bongekanjing/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts every Grand Theft Auto VI (PS5) trophy's earn rate on Exophase at 2026-11-26T00:00Z, as five quantiles per trophy, from the house's recorded reads up to the deadline.

Bounty: https://make.fun/bounty/gta6-trophy-rarity-forecast-01

## How it works

`main.py` (Python 3.13, standard library only) reads one case file and writes one forecast file. An earn rate is a ratio — earners over players tracked — and the two sides move for different reasons, so the model forecasts them separately and divides:

- **Players tracked** `N(t)`: log `N` is fitted against log-time `ln(1 + hours since release)` by least squares and extrapolated to the target. Log-time is self-damping, so a 4.5-day extrapolation stays sane.
- **Earners** `E(t)` per trophy: `log(E + 0.5)` is fitted against the same log-time, per trophy, damped (×0.8). Early story trophies gain earners fast, late or hard trophies barely move — each trophy tells its own story.
- **Median rate** = `100 · E(target) / N(target)`, clamped to 0.01–99.99%, with its logit move from the last read limited to [−4, +4.5].

Two details matter. First, **stale pages**: Exophase refreshes a game's counts hours apart, so a flat history can mean a stale page, not a still world — on launch night the page sat at 638 players for hours, then jumped to 3,868. When players-tracked never moves in the history, the model falls back to a launch-growth prior for `N` anchored at the first read instead of trusting the flat fit; when the history does move, the fit is used and no prior is applied. Second, **saturation**: a trophy held by (nearly) everyone has earners equal to players tracked, so its earners scale with `N` and its forecast stays near 100%.

The **spread** (logit sd) combines a random-walk term `0.5·√(horizon in days)`, the per-trophy fit's residual spread scaled by extrapolation distance, and `0.25·|log(N(target)/N(last))|` for refresh uncertainty, with a horizon-growing floor and a 2.2 cap. Quantiles are `inv-logit(median logit + z·spread)`.

The program is deterministic (fixed computation, no clock, no randomness, no network) and runs in ~0.1 s per case, well under the 8 s cap.

## Practice results

No reference answers ship in the kit, so practice loss is scored against answers reconstructed from the totals stated in the rules (3,868 players, 4,276 trophies earned at the 04:30Z refresh; the 100% trophy takes ~all new players, leaving ~2 new earnings per other trophy):

| case | loss (reconstructed) | typical miss | 90% coverage | starter baseline loss |
|------|---------------------|--------------|--------------|-----------------------|
| p00  | 0.118               | 0.007 pp     | 0.98         | 1.375                 |
| p01  | 0.116               | 0.005 pp     | 0.98         | 1.375                 |

The loss stays between 0.12 and 0.37 if the reconstruction is varied from +0 to +10 new earnings per trophy, versus 1.375 for the persistence baseline — the improvement is the stale-page refresh expectation, not a lucky split. Edge cases verified: a trophy at 0%, a trophy at 100%, a null rate in one read, a single-read history, and a synthetic moving history (early trophy rising, late trophy falling as players tracked grows). Both practice outputs are valid: exact shape, 55 trophies, ordered quantiles in range, ≤ 65,536 bytes, byte-identical across two runs.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see [LICENSE](LICENSE).
