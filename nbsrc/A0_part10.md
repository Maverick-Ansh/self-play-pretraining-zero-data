---
# Part 10 · Results

Every number below is read from `runs/*/log.jsonl` (written by the training processes) and `runs/*/final_eval.json`. The in-loop evaluation uses the first 64 windows of each corpus (511 bytes each); the final evaluation uses all 256 (136 for Mutopia).

Colours are fixed by **arm** across every figure: self-play blue, uniform prior orange, shuffled reward aqua. Model size is shown by lightness within the arm's hue (darker = larger). The grey dashed line in each panel is the untrained in-context KT counter from Part 7.3.
