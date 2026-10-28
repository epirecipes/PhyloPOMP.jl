"""Independent high-precision reference solver for the SEIRS filter equation.

Implements the proposal-independent weighted generator tabulated in the lab wiki
(theory/seir-filter-equation.md), directly from King, Lin & Ionides (2024) eq.
seirs5/seirs23, using dense mpmath matrix exponentials (30 significant digits).
Only feasible for tiny problems (population <= ~5, <= 3 lineages).

Usage:  .venv/bin/python exact_mpmath.py cases.json  -> prints JSON {name: loglik}
The JSON case format is shared with the Julia solver (ExactSEIR):
  {"name","params":{beta,sigma,gamma,omega,psi,chi,pop},"x0":[S,E,I,R],
   "t0","T","events":[{"t","type":"root|sample|node","parent","children":[..]}]}
"""
import itertools
import json
import sys

import mpmath as mp

mp.mp.dps = 30
EXPOS, INFEC = 0, 1


def pop_states(pop):
    return [(s, e, i, r) for s, e, i, r in itertools.product(range(pop + 1), repeat=4)
            if s + e + i + r <= pop]


def colorings(lins):
    return [tuple(zip(lins, ds)) for ds in itertools.product((EXPOS, INFEC), repeat=len(lins))]


def ells(col):
    le = sum(1 for _, d in col if d == EXPOS)
    return le, len(col) - le


def feasible(x, col):
    le, li = ells(col)
    return x[1] >= le and x[2] >= li


def setcol(col, lin, deme):
    return tuple(sorted([(l, d) for l, d in col if l != lin] + [(lin, deme)]))


def generator(states, index, p):
    """Dense weighted generator G with dw/dt = G w (column = source state)."""
    n = len(states)
    G = mp.zeros(n, n)
    beta, sigma, gamma, omega, psi, chi, N = (mp.mpf(p[k]) for k in
                                              ("beta", "sigma", "gamma", "omega", "psi", "chi", "pop"))
    for j, (x, col) in enumerate(states):
        S, E, I, R = x
        le, li = ells(col)
        out = beta * S * I / N + sigma * E + gamma * I + omega * R + (psi + chi) * I
        G[j, j] -= out

        def add(x2, col2, rate):
            if rate != 0 and (x2, col2) in index:
                G[index[(x2, col2)], j] += rate
        if S >= 1 and I >= 1:
            a = beta * S * I / N
            E2 = E + 1
            add((S - 1, E2, I, R), col, a * mp.mpf(E2 - le) / E2)
            for lin, d in col:
                if d == INFEC:
                    li2 = li - 1
                    add((S - 1, E2, I, R), setcol(col, lin, EXPOS), a * mp.mpf(I - li2) / (E2 * I))
        if E >= 1:
            a = sigma * E
            I2 = I + 1
            if E - 1 >= le:
                add((S, E - 1, I2, R), col, a * mp.mpf(I2 - li) / I2)
            for lin, d in col:
                if d == EXPOS:
                    add((S, E - 1, I2, R), setcol(col, lin, INFEC), a / I2)
        if I >= 1 and I - 1 >= li:
            add((S, E, I - 1, R + 1), col, gamma * I)
        if R >= 1:
            add((S + 1, E, I, R - 1), col, omega * R)
    return G


def solve(case):
    p = case["params"]
    pop = int(p["pop"])
    xs = pop_states(pop)
    lins = []
    w = {(tuple(case["x0"]), ()): mp.mpf(1)}
    loglik = mp.mpf(0)
    t = mp.mpf(case["t0"])
    events = sorted(case["events"], key=lambda ev: ev["t"])
    times = [mp.mpf(ev["t"]) for ev in events] + [mp.mpf(case["T"])]
    beta, psi, chi, N = (mp.mpf(p[k]) for k in ("beta", "psi", "chi", "pop"))
    for k, ev in enumerate(events):
        # regular part up to the event time
        dt = times[k] - t
        if dt > 0:
            w, lc = propagate(w, lins, xs, p, dt)
            loglik += lc
        t = times[k]
        # singular part
        new = {}
        typ, par, ch = ev["type"], ev["parent"], ev["children"]

        def put(key, val):
            if val != 0:
                new[key] = new.get(key, mp.mpf(0)) + val
        for (x, col), val in w.items():
            S, E, I, R = x
            cd = dict(col)
            if typ == "root":
                c = ch[0]
                le, li = ells(col)
                if E - le >= 1:
                    put((x, setcol(col, c, EXPOS)), val)
                if I - li >= 1:
                    put((x, setcol(col, c, INFEC)), val)
            elif cd.get(par) != INFEC:
                continue
            elif typ == "sample" and len(ch) == 0:
                col2 = tuple(sorted((l, d) for l, d in col if l != par))
                le, li2 = ells(col2)
                put((x, col2), val * psi * (I - li2))
                if chi > 0 and I - 1 >= li2:
                    put(((S, E, I - 1, R), col2), val * chi * I)
            elif typ == "sample" and len(ch) == 1:
                col2 = setcol(tuple((l, d) for l, d in col if l != par), ch[0], INFEC)
                put((x, col2), val * psi)
            elif typ == "node":
                if S < 1:
                    continue
                base = tuple((l, d) for l, d in col if l != par)
                wt = beta * S / (N * (E + 1))
                x2 = (S - 1, E + 1, I, R)
                for d1, d2 in ((INFEC, EXPOS), (EXPOS, INFEC)):
                    col2 = setcol(setcol(base, ch[0], d1), ch[1], d2)
                    put((x2, col2), val * wt)
            else:
                raise ValueError(f"bad event {ev}")
        lins = sorted(set(l for l in lins if l != par) | set(ch))
        new = {key: v for key, v in new.items() if feasible(key[0], key[1])}
        s = sum(new.values())
        if s == 0:
            return -mp.inf
        loglik += mp.log(s)
        w = {key: v / s for key, v in new.items()}
    dt = times[-1] - t
    if dt > 0:
        w, lc = propagate(w, lins, xs, p, dt)
        loglik += lc
    return loglik


def propagate(w, lins, xs, p, dt):
    states = [(x, col) for x in xs for col in colorings(lins) if feasible(x, col)]
    index = {s: i for i, s in enumerate(states)}
    G = generator(states, index, p)
    v = mp.matrix(len(states), 1)
    for key, val in w.items():
        v[index[key]] = val
    v = mp.expm(G * dt) * v
    s = sum(v)
    out = {states[i]: v[i] / s for i in range(len(states)) if v[i] != 0}
    return out, mp.log(s)


if __name__ == "__main__":
    cases = json.load(open(sys.argv[1]))
    res = {c["name"]: mp.nstr(solve(c), 20) for c in cases}
    print(json.dumps(res, indent=1))
