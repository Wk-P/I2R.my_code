# v4.3.9 Remaining-service × ECU feasibility matrix F in the observation

On top of v4.3.8 (unified reward, no early stop, Maskable with EXIT), the only change is the observation (`--obs feas`):

- An M×N matrix F is appended: F[k, j] = 1 iff service k is not placed yet and could be placed on ECU j now without violating capacity or privacy; rows of placed services are 0.
- Motivation: group 13 of the base observation only says how many ECUs a future service k can go to (the mean of row k of F), not whether several services compete for the same ECUs; F says which ones.
- The base part (87 / 77 / 102 dims) and the mask position are unchanged. New dimensions: LT 237, EQ 177, GT 252.
- Pilot: `src/scripts/run_v4.3.9.py --pilot`, Maskable PPO / DQN × 3 scenarios × seed 1 × 1M steps; `pilot_report.md` puts each model next to the same model of the v4.3.8 pilot (base observation).
