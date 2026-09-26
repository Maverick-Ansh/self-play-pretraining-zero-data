---
# Part 6 · Recognising mathematics on a tape (Table 1)

> *"We allow up to 30 unrelated leading bytes before the structured portion of a tape begins. The remainder of the tape must satisfy the corresponding recurrence modulo 256 [...] we additionally require a minimal period of at least 30."* (App. C)

| family | recurrence (mod 256) | example program |
|---|---|---|
| arithmetic | constant 1st difference | `+[.++]` → 1, 3, 5, 7 |
| quadratic | constant 2nd difference | `,.[<C>>VX<RX++]` → 9, 25, 59, 111 |
| cubic | constant 3rd difference | |
| Fibonacci | $v_n = v_{n-1}+v_{n-2}$ | `,[[.C>.C>]]` → a, a, 2a, 3a, 5a |
| geometric | $v_n = r\,v_{n-1}$ | `+[.L>]` → 1, 3, 9, 27 |

The classifier is the authors' `detect.py`, **copied verbatim** so "discovered" means the same thing here as in the paper, with one repair and one speed-up:

* **Repair.** Their degenerate-tape guard computes the period of the last **2048** bytes, a number that assumes 4095-byte tapes. On a 511-byte tape that window is the whole tape, and a tape like `[5, 0, 0, 0, …]` passes as "geometric with ratio 0". On 4,096 uniform-prior programs the unrepaired detector reported **1,256 geometric discoveries, all junk**. The window is now half the tape (and still exactly 2048 at the paper's length).
* **Speed-up.** A vectorised prefilter so every round's pool can be scanned (milliseconds instead of seconds).
