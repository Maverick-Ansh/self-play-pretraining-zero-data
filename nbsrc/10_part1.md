---
# Part 1 · The universal machine $U$

> *"We use a Brainf\*ck-like Turing-complete language [...] Its primitive instructions manipulate a byte-valued tape, implement loops, and read or emit bytes."* (§2.1)

A program is a string over an alphabet $\mathcal{A}$ of **19 tokens**:

| tokens | meaning |
|---|---|
| `>` `<` | move the head right / left (the tape is **circular**) |
| `+` `-` | add / subtract 1 from the current cell, **mod 256** |
| `[` `]` | loop while the current cell $\neq 0$ (unmatched brackets are **no-ops**) |
| `.` | emit the current cell as an output byte |
| `,` | read a uniform random byte from the input tape $\omega$ into the cell |
| `Z R L N C G H W V X` | ten macros (Table 4), e.g. `C` = `[->+>+<<]` "move x into the two right cells", `X` = `[-]` + 16 × `+` "set cell to 16" |
| `F` | end of program |

Every string is a valid program. Running $x$ with random input $\omega$ gives the output $y = U(x,\omega) \in \{0..255\}^T$: the first $T$ bytes it emits, zero-padded if it stops early. It always stops: at the end of the program, after $T$ bytes, or when a **step budget** runs out.

Because `,` reads random bytes, **one program defines a distribution over byte strings**, not just one string. That is what makes a program a *data-generating process*, and the learner's training data is a mixture of such processes.

### Two numbers the paper does not give, recovered from the authors' own data

The paper says "finite memory" and "bounded step budget" without numbers. Both matter: the budget decides whether a slow Fibonacci program ever prints enough bytes to be recognised, and the tape size decides what a program sees when its head wraps around.

The authors released `hits.jsonl`: **20,045 programs their generator discovered, with the exact bytes each one printed.** 6,143 of them never read `,`, so their output is deterministic, and any machine with the right semantics must reproduce those bytes. The cells below replay them.

* **Step budget.** The released programs need up to 2031 steps per byte (at most 8,319,126 steps for 4095 bytes), just under $2^{23} = 2048 \times 4096$. So the budget is $2048\,(T+1)$ steps.
* **Tape size: an honest partial answer.** No single circular size reproduces every record. 128 and 2048 cells reproduce all 6,143 recorded first bytes, but 35 and 33 arithmetic tapes (respectively) leave their recurrence once the head wraps. 4096 keeps every tape arithmetic but misses 3 programs that walk *left* and only print the recorded numbers if they wrap around a shorter tape. So the authors' tape is not a plain circular array of one fixed size. This reproduction uses **2048**, which disagrees with theirs only for programs whose head wraps around (0.5% of the released hits).
