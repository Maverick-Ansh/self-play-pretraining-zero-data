"""Mathematical-structure detector for output tapes: App. C / Table 1.

    "We allow up to 30 unrelated leading bytes before the structured portion of
     a tape begins. The remainder of the tape must satisfy the corresponding
     recurrence modulo 256. [...] To eliminate degenerate matches, we
     additionally require a minimal period of at least 30."             (App. C)

`min_period` and `classify_record` are copied verbatim from the authors'
detector (acowsik/self_play_pretraining, tables/table1/detect.py) so that "discovered" means exactly the same thing here as in the
paper. `scan` adds a vectorized prefilter so every round's pool can be checked:
a row can only be a hit if its core x[30:] satisfies one of the recurrences.
"""
import numpy as np

MAX_PREFIX = 30
MIN_PERIOD = 30
FAMILIES = ("arithmetic", "fibonacci", "geometric", "quadratic", "cubic")


def _window(n):
    """Trailing window for the eventual-period test. The original hard-codes 2048,
    which assumes 4095-byte tapes; on a 511-byte tape it covers the whole tail and
    lets a transient-then-constant tape ([5, 0, 0, ...]) through. Half the tape
    keeps the original's intent; at n = 4095 the original 2048 is used unchanged."""
    return 2048 if n >= 4095 else n // 2


def min_period(t: bytes):
    """Smallest p with t[i] == t[i+p] for all i (KMP failure function)."""
    n = len(t)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k and t[i] != t[k]:
            k = pi[k - 1]
        if t[i] == t[k]:
            k += 1
        pi[i] = k
    return n - pi[-1]


def _extend_back(x, s0, ok_fn):
    """Smallest s <= s0 such that ok_fn holds on x[s:]."""
    s = s0
    while s > 0 and ok_fn(x, s - 1):
        s -= 1
    return s


def classify_record(x: np.ndarray):
    """x: uint8 1-D tape. Returns list of hit dicts."""
    n = len(x)
    if n < 200:
        return []
    core = x[MAX_PREFIX:].astype(np.int64)
    hits = []

    d1 = np.diff(core) % 256
    d2 = np.diff(d1) % 256
    d3 = np.diff(d2) % 256

    poly_order = None
    if (d1 == d1[0]).all():
        poly_order = 1
    elif (d2 == d2[0]).all():
        poly_order = 2
    elif (d3 == d3[0]).all():
        poly_order = 3

    if poly_order is not None:
        k = poly_order

        def ok(xx, s, k=k):
            seg = xx[s:].astype(np.int64)
            for _ in range(k):
                seg = np.diff(seg) % 256
            return (seg == seg[0]).all()

        s = _extend_back(x, MAX_PREFIX, ok)
        fam = {1: 'arithmetic', 2: 'quadratic', 3: 'cubic'}[k]
        hits.append((fam, s, {'diffs': [int(v) for v in
                    [d1[0], d2[0] if k >= 2 else None, d3[0] if k >= 3 else None] if v is not None]}))

    fib_ok = ((core[2:] - core[1:-1] - core[:-2]) % 256 == 0).all()
    if fib_ok:
        def okf(xx, s):
            seg = xx[s:].astype(np.int64)
            return ((seg[2:] - seg[1:-1] - seg[:-2]) % 256 == 0).all()
        s = _extend_back(x, MAX_PREFIX, okf)
        hits.append(('fibonacci', s, {}))

    # geometric: candidate r from an odd element, else brute-force 256 ratios
    cand = None
    odd_idx = np.nonzero(core[:-1] & 1)[0]
    if len(odd_idx):
        i = odd_idx[0]
        cand = [int((int(core[i + 1]) * pow(int(core[i]), -1, 256)) % 256)]
    else:
        cand = range(256)
    for r in cand:
        if ((core[1:] - r * core[:-1]) % 256 == 0).all():
            def okg(xx, s, r=r):
                seg = xx[s:].astype(np.int64)
                return ((seg[1:] - r * seg[:-1]) % 256 == 0).all()
            s = _extend_back(x, MAX_PREFIX, okg)
            hits.append(('geometric', s, {'ratio': int(r)}))
            break

    out = []
    for fam, s, params in hits:
        tail = bytes(x[s:].astype(np.uint8))
        p = min_period(tail)
        # eventual period over the trailing window: kills transient-then-cycle
        # junk (e.g. one nonzero byte then all zeros). Window 2048 >= 2x the
        # largest possible true period of these families mod 256 (<=1024).
        p_eff = min_period(tail[-_window(n):])
        if min(p, p_eff) < MIN_PERIOD:
            continue
        out.append({'family': fam, 'start_offset': int(s), 'period': p,
                    'params': params,
                    'first_terms': [int(v) for v in x[s:s + 12]],
                    'prefix': [int(v) for v in x[:s]],
                    'tape_len': int(n)})
    return out


def _candidates(X):
    """Rows of X [N, T] whose core satisfies some family's recurrence (superset of hits).

    A tape whose trailing window is constant has eventual period 1 and can never
    pass MIN_PERIOD, so those rows (including every empty tape) are skipped first.
    """
    X = X.astype(np.int64)
    W = _window(X.shape[1])
    live = ~(X[:, -W:] == X[:, -1:]).all(1)
    idx = np.nonzero(live)[0]
    core = X[idx, MAX_PREFIX:]
    d1 = np.diff(core, axis=1) % 256
    d2 = np.diff(d1, axis=1) % 256
    d3 = np.diff(d2, axis=1) % 256
    hit = (d1 == d1[:, :1]).all(1) | (d2 == d2[:, :1]).all(1) | (d3 == d3[:, :1]).all(1)
    hit |= ((core[:, 2:] - core[:, 1:-1] - core[:, :-2]) % 256 == 0).all(1)
    a, b = core[:, :-1], core[:, 1:]
    odd = (a & 1).astype(bool)
    has_odd = odd.any(1)
    j = odd.argmax(1)                                   # first odd element: the ratio is forced
    rows = np.arange(len(core))
    inv = np.array([pow(int(v), -1, 256) if v & 1 else 0 for v in range(256)])
    r = (b[rows, j] * inv[a[rows, j]]) % 256
    hit |= has_odd & ((b - r[:, None] * a) % 256 == 0).all(1)
    for i in np.nonzero(~has_odd & ~hit)[0]:            # all-even rows: try every ratio
        hit[i] = any(((b[i] - q * a[i]) % 256 == 0).all() for q in range(256))
    return idx[hit]


def scan(X):
    """[(row, hit_dict)] for every structured tape in X [N, T] uint8."""
    return [(int(i), h) for i in _candidates(X) for h in classify_record(X[i].astype(np.int64))]
