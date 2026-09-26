"""The universal machine U: Sec. 2.1 and App. E of arXiv:2609.30063.

    "Let x in A^{<=L} denote a program generated over the machine's instruction
     alphabet. Executing x on the universal machine U with random input tape w
     produces a byte sequence y = U(x, w) in {0, ..., 255}^T."          (Sec. 2.1)

    "Importantly, every string over A is executable. There is no syntax error:
     the only way a program can be malformed is through unmatched brackets,
     which we treat as no-ops. [...] The tape is circular, so that a head move
     off either end of the finite memory wraps around, and incrementing or
     decrementing a cell wraps modulo m. [...] Execution always terminates:
     at the step budget, the end of the program, or the T-th emitted byte,
     whichever comes first."                                              (App. E)

    "It emits y = (1, 2, 3, 0, ..., 0): one byte per iteration, then
     zero-padding once the program halts."                                (App. E)

The alphabet is the 8 Brainf*ck instructions, the 10 macros of Table 4 and the
terminator F: 19 tokens. Programs are byte strings (token id = byte value), so
the same 256-way embedding table serves programs and outputs alike.

Two numbers the paper leaves open are fixed here and flagged in REPORT.md:
    TAPE_CELLS  = 2048   The paper says only "finite memory". Replaying the 6143
                         deterministic programs in the authors' released hits.jsonl:
                           128 cells  all 6143 recorded first bytes; 35 of 5955 arithmetic
                                      tapes leave their recurrence after the head wraps
                           2048 cells all 6143 first bytes; 33 of 5955 leave it
                           4096 cells all 5955 stay arithmetic; 3 first-byte records missed
                                      (programs that walk LEFT and wrap)
                         No single circular size reproduces every record, so the authors'
                         tape is not a plain circular array of one size. 2048 disagrees
                         with it only for programs whose head wraps (0.5% of the hits).
    step budget = 2048*(T+1): the released hits need up to 2031 steps/byte (max
                 8,319,126 < 2^23 = 2048*4096), so the paper's budget is >= that
"""
import numpy as np
from numba import njit, prange

BF = b"<>+-[].,"                       # the eight primitive instructions
MACROS = {                              # Table 4: token -> pure-BF expansion
    ord("Z"): b"[-]",                   # clear current cell
    ord("R"): b"[->+<]",                # clear, add x into right neighbour
    ord("L"): b"[->+++<]",              # clear, add 3x into right neighbour
    ord("N"): b"[-<->]",                # clear, subtract x from left neighbour
    ord("C"): b"[->+>+<<]",             # clear, add x into the two right cells
    ord("G"): b"[>]",                   # scan right to next zero cell
    ord("H"): b"[<]",                   # scan left to next zero cell
    ord("W"): b"[[-]>+<]",              # if x != 0: increment right, clear current
    ord("V"): b"[.>]",                  # print stored string until a zero cell
    ord("X"): b"[-]" + b"+" * 16,       # set cell to 16
}
END = ord("F")                          # explicit end-of-program token
PROG_PREFIX, OUT_PREFIX = ord("S"), ord("O")   # row tags (Sec. 2.3)

INSTR = np.frombuffer(BF + bytes(MACROS), dtype=np.uint8)   # 18 instructions
ALPHABET = np.append(INSTR, np.uint8(END))                   # |A| = 19
TAPE_CELLS = 2048                      # recovered from the released hits, see module docstring
MODULUS = 256                            # cell values live in Z_256 (App. E, m = 256)

# 256-entry lookup: which bytes a program may contain (used to mask generator logits)
ALPHABET_MASK = np.zeros(256, dtype=bool)
ALPHABET_MASK[ALPHABET] = True

_OPS = {ord(c): i for i, c in enumerate(">" "<" "+" "-" "." "," "[" "]")}
_EXPAND = {c: MACROS.get(c, bytes([c])) for c in bytes(INSTR)}


def expand(body: bytes) -> bytes:
    """Macro tokens -> pure BF. Everything after F (if any) is not part of the program."""
    body = body.split(b"F", 1)[0]
    return b"".join(_EXPAND[c] for c in body)


def compile_batch(bodies):
    """Expand and pack N programs into flat opcode arrays for the jitted runner.

    Returns (ops int8[total], starts int64[N+1]); opcodes 0..7 = > < + - . , [ ]
    """
    codes = [expand(b) for b in bodies]
    starts = np.zeros(len(codes) + 1, dtype=np.int64)
    starts[1:] = np.cumsum([len(c) for c in codes])
    flat = b"".join(codes)
    lut = np.full(256, -1, dtype=np.int8)
    for ch, op in _OPS.items():
        lut[ch] = op
    ops = lut[np.frombuffer(flat, dtype=np.uint8)] if flat else np.zeros(0, np.int8)
    return ops.astype(np.int8), starts


@njit(cache=True)
def _match(ops, lo, hi, jump):
    """Stack-based bracket matching on ops[lo:hi]; unmatched brackets keep jump = -1."""
    stack = np.empty(hi - lo, dtype=np.int64)
    sp = 0
    for i in range(lo, hi):
        jump[i] = -1
        if ops[i] == 6:                      # '['
            stack[sp] = i
            sp += 1
        elif ops[i] == 7 and sp > 0:         # ']' with an open partner
            sp -= 1
            j = stack[sp]
            jump[i] = j
            jump[j] = i


@njit(cache=True)
def _splitmix(state):
    """One step of splitmix64: the per-program random input tape w."""
    state = (state + np.uint64(0x9E3779B97F4A7C15)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = state
    z = ((z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    z = ((z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)) & np.uint64(0xFFFFFFFFFFFFFFFF)
    return state, z ^ (z >> np.uint64(31))


@njit(cache=True, parallel=True)
def _run(ops, starts, seeds, T, max_steps, tape_cells, out, n_out, steps_used, depth_max):
    N = starts.shape[0] - 1
    jump = np.empty(ops.shape[0], dtype=np.int64)
    for p in prange(N):
        lo, hi = starts[p], starts[p + 1]
        _match(ops, lo, hi, jump)
        tape = np.zeros(tape_cells, dtype=np.int64)
        rng = np.uint64(seeds[p])
        ptr, pc, steps, n, depth, dmax = 0, lo, 0, 0, 0, 0
        while pc < hi and steps < max_steps and n < T:
            op = ops[pc]
            if op == 0:
                ptr = (ptr + 1) % tape_cells
            elif op == 1:
                ptr = (ptr - 1) % tape_cells
            elif op == 2:
                tape[ptr] = (tape[ptr] + 1) % MODULUS
            elif op == 3:
                tape[ptr] = (tape[ptr] - 1) % MODULUS
            elif op == 4:
                out[p, n] = tape[ptr]
                n += 1
            elif op == 5:                     # ',' reads an i.i.d. uniform byte from w
                rng, r = _splitmix(rng)
                tape[ptr] = r & np.uint64(255)
            elif op == 6 and jump[pc] >= 0:   # '[': skip the loop, or enter it
                if tape[ptr] == 0:
                    pc = jump[pc]
                else:
                    depth += 1
                    if depth > dmax:
                        dmax = depth
            elif op == 7 and jump[pc] >= 0:   # ']': repeat the body, or leave it
                if tape[ptr] != 0:
                    pc = jump[pc]
                else:
                    depth -= 1
            pc += 1
            steps += 1
        n_out[p] = n
        steps_used[p] = steps
        depth_max[p] = dmax


def execute(bodies, T, seed=0, max_steps=None, tape_cells=TAPE_CELLS):
    """Run N programs. `bodies` are byte strings without the S prefix (F optional).

    Returns dict with
        out    uint8[N, T]  emitted bytes, zero-padded after the program halts
        n_out  int64[N]     number of bytes actually emitted
        steps  int64[N]     instructions executed
        depth  int64[N]     maximum dynamic loop depth (MAP-Elites descriptor, App. G)
    """
    N = len(bodies)
    max_steps = 2048 * (T + 1) if max_steps is None else max_steps
    ops, starts = compile_batch(bodies)
    seeds = (np.full(N, seed, dtype=np.uint64) * np.uint64(0x2545F4914F6CDD1D)
             + np.arange(N, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15))
    out = np.zeros((N, T), dtype=np.int64)
    n_out, steps, depth = (np.zeros(N, dtype=np.int64) for _ in range(3))
    _run(ops, starts, seeds, T, max_steps, tape_cells, out, n_out, steps, depth)
    return dict(out=out.astype(np.uint8), n_out=n_out, steps=steps, depth=depth)


# ----------------------------------------------------------------------------
# The fixed prior g0 and the local edits used by the program pool
# ----------------------------------------------------------------------------
def log_g0(lengths):
    """log g0(x) = -l(x) log|A| (Sec. 2.2), l = tokens up to and including F.

    A program that hit the length cap had its F forced (probability 1), so that
    token is not charged: pass lengths that already exclude a forced F.
    """
    return -np.asarray(lengths, dtype=np.float64) * np.log(len(ALPHABET))


def sample_uniform_prior(n, max_body, rng):
    """Draw from g0: i.i.d. uniform tokens over A until F (Sec. 3.1, the control arm).

    Returns (bodies, charged_lengths). A body that reaches `max_body` tokens is
    closed by a forced F, which costs nothing under the capped prior.
    """
    k = len(ALPHABET)
    draws = rng.integers(0, k, size=(n, max_body))
    is_end = draws == k - 1
    first_end = np.where(is_end.any(1), is_end.argmax(1), max_body)
    bodies, charged = [], []
    for i in range(n):
        L = int(first_end[i])
        bodies.append(ALPHABET[draws[i, :L]].tobytes())
        charged.append(L + 1 if L < max_body else L)
    return bodies, np.array(charged)


def mutate(body: bytes, rng, max_body: int) -> bytes:
    """One local edit (App. G): single-token substitution, insertion, or deletion."""
    b = bytearray(body)
    kind = rng.integers(3)
    if kind == 2 and len(b) >= max_body:
        kind = 0
    if len(b) == 0 or (kind == 1 and len(b) == 1):
        kind = 2
    pos = int(rng.integers(max(len(b), 1)))
    tok = int(INSTR[rng.integers(len(INSTR))])
    if kind == 0:
        b[pos] = tok
    elif kind == 1:
        del b[pos]
    else:
        b.insert(int(rng.integers(len(b) + 1)), tok)
    return bytes(b)


def niche(depth: int, body_len: int) -> int:
    """MAP-Elites cell (App. G): loop depth clamped to 0..8 x length bucket <=8/16/32/>32."""
    bucket = 0 if body_len <= 8 else 1 if body_len <= 16 else 2 if body_len <= 32 else 3
    return min(int(depth), 8) * 4 + bucket          # 9 * 4 = 36 niches
