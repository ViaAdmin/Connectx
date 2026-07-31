# ConnectX NNUE Agent

**English** | [简体中文](README.zh-CN.md)

A Kaggle [ConnectX](https://www.kaggle.com/competitions/connectx) competition agent: **minimax (alpha-beta) + NNUE value network + self-play reinforcement learning**. Currently around 1200 on the public leaderboard (rank 34 / 254).

## The Game

ConnectX is the classic **Connect Four**: a vertical 6×7 board where two players take turns dropping a piece into a column; the piece falls to the lowest empty cell. The first player to line up 4 pieces horizontally, vertically, or diagonally wins; a full board is a draw.

The rules are simple, but the state space holds roughly 4.5 trillion legal positions, and the first player has a proven winning strategy (the game is strongly solved). This makes it an ideal proving ground for search + evaluation architectures: small enough to dig deep, yet too large to brute-force online.

## Competition Constraints

The hard constraints of the Kaggle game environment directly shaped this project's design:

| Constraint | Design response |
|---|---|
| **CPU only**, no GPU | The evaluation network must be tiny (84→256→64→32→1); inference is pure NumPy with an incremental accumulator, not a large deep net |
| **~2 seconds per move**, timeout = loss | Iterative deepening + 1.7s wall-clock abort; a legal result from the last completed depth is always available; any exception falls back to a legal column |
| **Single-file submission**, external dependencies are awkward | Weights embedded as base64; one self-contained file, zero downloads at runtime |
| Game image ships **NumPy 2.4, incompatible with numba** | The submission-side search is pure-Python bitboards (training notebooks are unaffected and still use numba) |
| **Multi-threaded BLAS is slower** on tiny matrix products | Force single-threaded BLAS; leaf evaluation ~12x faster |

In short: the compute budget is extremely tight, and games are decided by "effective search depth per unit of CPU time" — so engineering optimizations (bitboards, transposition table, move ordering, incremental evaluation) matter as much as the model itself.

## Preface

The minimax + NNUE architecture was settled at the very start: not knowing the game well enough to hand-craft an evaluation function, self-play training was the only viable route. The network design was inspired by Stockfish's NNUE, but ConnectX has no natural anchor piece like a king to build features around, so I settled on a plain dual-perspective 84-dimensional one-hot encoding.

After several iterations, the model's win rate against heuristic search was not high enough, so I started adding hand-crafted board features to help the NNUE learn faster — only to find that the multi-feature models not only lost to the 84-dim model, but even lost to the heuristic search. I abandoned that approach and tried an 85-dim model (adding a turn dimension to convey side-to-move information), initialized losslessly from the old model via weight transfer. I had high hopes, but after many versions its win rate stayed poor: in 200 games against the 84-dim base model under the same minimax framework, playing both colors, it won only 28%.

In the end I returned to the 84-dim model and kept training, while systematically optimizing the search engine for the first Kaggle submission. This repository is what that path produced.

## Files

| File | Purpose |
|---|---|
| `submission.py` | Kaggle submission agent (self-contained single file; **model weights not included** — embed your own trained weights before use) |
| `selfplay_training.ipynb` | Self-play training notebook: the full iteration loop of data generation → training → arena evaluation |
| `model_evaluation.ipynb` | Evaluation notebook: NNUE vs NNUE at fixed depth / timed match mode (iterative deepening) |

## Tech Stack

- **Python**: everything
- **PyTorch**: NNUE training (not required at inference time)
- **NumPy**: pure-NumPy forward pass on the submission side
- **Numba**: JIT-accelerated bitboard search in the training and evaluation notebooks
- **pandas / multiprocessing**: self-play data management and parallel games
- **kaggle-environments**: local game simulation

## Algorithms & Implementation

### NNUE Value Network

- Input: 84-dim dual-perspective one-hot (42 own cells + 42 opponent cells); architecture `84 → 256 → 64 → 32 → 1 (tanh)`
- The first layer uses an **incremental accumulator**: placing a piece just adds one weight row to the accumulator, so leaf nodes only compute the last three layers
- On the submission side, weights are transposed into NNUE layout, saved as npz, and embedded as base64 in the single file; decoded at runtime with pure NumPy — no torch, no attached datasets

### Self-Play Reinforcement Learning

- **Temperature sampling throughout**: every move applies softmax temperature sampling to the exact root search scores of all 7 columns — position diversity comes from controlled randomness across the whole game, not from random openings
- **Strict separation of proven values and evaluations**: proven terminal results are encoded as `±(100+depth)`, never confused with tanh evaluations that saturate to ±1.0; proven wins are converted immediately, proven losses are masked out of sampling
- **Data pipeline**: position-level deduplication + a history replay pool (a fixed ratio of old data mixed into each round)
- **Training**: validation early stopping, iterative fine-tuning
- **Arena gating**: after each round, new and old models play both colors; repeated sub-par win rates trigger a catastrophic rollback, preventing self-play degradation

### Search (Submission Side)

- **49-bit bitboards**: 7 bits per column (6 cells + 1 sentinel); four-in-a-row detection is 4 shift operations
- **Alpha-beta + iterative deepening**: a result from the last completed depth is always available; 1.7s wall-clock abort (clock checked every 256 nodes, timeout raises an exception to unwind the recursion in one go)
- **Transposition table**: EXACT / LOWER / UPPER entries; proven results stored with depth 99 for permanent reuse; the table is cleared wholesale when it exceeds its size cap
- **Dynamic move ordering**: TT move > killers (two per ply) > history heuristic (cutoff moves get `+= depth²`); ordering applied only at deep nodes
- **LMR**: late moves are searched shallow first, re-searched on fail-high
- **Symmetry pruning**: only half the moves are searched in left-right symmetric positions
- **Immediate-win precheck**: each node first checks whether the side to move has an instant win
- **Engineering details**: force single-threaded BLAS (multi-threaded sync overhead is ~12x on tiny matrix products); no numba (the Kaggle game image ships NumPy 2.4, incompatible with numba — importing it loses the game on the spot); any exception falls back to a legal column

## Deliberately Not Included

The purpose of this repository is to contribute **ideas and algorithms** to the community, not to hand out a ready-to-upload finished product. The following components are used in the project but deliberately omitted:

- **Trained model weights**: the weight string in `submission.py` is an empty placeholder; train your own model with `selfplay_training.ipynb` and embed it (format documented in the header comments of `submission.py`)
- **Heuristic search agent**: the hand-crafted evaluation search that bootstrapped the initial dataset and served as a long-term sparring/evaluation baseline. Window scoring + center control is a well-known pattern in community tutorials — capable readers can implement their own
- **Opening book**: an opening book built via offline search is about to ship in the submission version; its construction notebook, embedding tools, and book data are not open-sourced
- **Weight embedding tool**: the small script that transposes a newly trained `.pth`, packs it, and replaces the embedded weight string in `submission.py` is not included (follow the layout notes in the file header to do it yourself)

## Quick Start

1. Training: run `selfplay_training.ipynb` in a Kaggle notebook; optionally point the config section at a pretrained `nnue_model_pretrained.pth`; it produces `/kaggle/working/nnue_model.pth`
2. Evaluation: pick a mode in the config section of `model_evaluation.ipynb` (fixed depth or timed match), fill in model paths, and run
3. Submission: embed your trained weights into `submission.py` following the layout in its header comments, then upload to Kaggle (the agent entry point is the `act` function at the end of the file)

## License

This project is released under the [MIT License](LICENSE): free to use, modify, distribute, and use commercially, provided the original copyright and license notice are retained in copies or substantial portions of the software (i.e., attribution required).
