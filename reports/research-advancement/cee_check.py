#!/usr/bin/env python3
"""cee_check.py -- standalone numerical checks for conservation of expected evidence (CEE).

Requires numpy only. Uses invented numbers and the published synthetic parameters of
extension/envs/synthetic_diagnosis/profiles.py (commit df467ea, Edwin Salcedo's synthetic study).
No patient data. This is a sandbox check, not a project experiment.

Parts
  1  Gain identity on random cases (tolerance 1e-9), KL_inc as a divergence, dual optimality.
  2  Special cases of specification section 5.3 and 5.6, and the degenerate scale rule of 5.5.
  3  Sequential statements of section 5.12 by exact enumeration (martingale, tower, T-step identity).
  4  Reproduction of the November 12 sandbox probe (same seeds, same random-call order).
  5  Scale repair by information-budget matching (section 5.11), vectorized, own seed.

Run:  python3 cee_check.py            (about 7 minutes on one CPU core)
      python3 cee_check.py --quick    (smaller samples, about 1 minute; the probe match in part 4 is not asserted)
"""
import argparse
import itertools
import sys
import time

import numpy as np

TOL = 1e-9
FLOOR = 1e-6


def floor_norm(p):
    p = np.maximum(p, FLOOR)
    return p / p.sum(-1, keepdims=True)


def project(prior, w, Q, tol=1e-13, max_iter=20000):
    """Minimum-KL projection of the joint w[o]*Q[o,y] onto joints with marginals (w, prior).

    Returns class weights u, normalizers Z, corrected posteriors Qp, KL_inc, converged flag.
    Iterative proportional fitting: u <- u * prior / (current imagined mean).
    """
    u = np.ones_like(prior)
    ok = False
    for _ in range(max_iter):
        Z = Q @ u
        mean = w @ (Q * u / Z[:, None])
        if np.abs(mean - prior).max() < tol:
            ok = True
            break
        u = u * prior / mean
        u = u / u.sum()
    Z = Q @ u
    Qp = Q * u / Z[:, None]
    kl_inc = float(prior @ np.log(u) - w @ np.log(Z))
    return u, Z, Qp, kl_inc, ok


def kl(p, q):
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


# ----------------------------------------------------------------------------------------------
# Part 1: the identity
# ----------------------------------------------------------------------------------------------
def part1(rng, n_cases):
    worst_id = worst_alg = worst_div = 0.0
    min_kl = np.inf
    dual_viol = 0.0
    skipped = 0
    for _ in range(n_cases):
        C = int(rng.integers(2, 7))
        K = int(rng.integers(2, 10))
        b = floor_norm(rng.dirichlet(np.ones(C)))
        w = floor_norm(rng.dirichlet(np.ones(K)))
        Q = floor_norm(rng.dirichlet(np.ones(C) * rng.uniform(0.3, 3.0), size=K))
        Jstar = rng.dirichlet(np.ones(K * C)).reshape(K, C)      # arbitrary truth over (o, y)
        bstar, Pstar = Jstar.sum(0), Jstar.sum(1)
        u, Z, Qp, kl_inc, ok = project(b, w, Q)
        # (i) identity with the projected weights
        direct = float((Jstar * (np.log(Qp) - np.log(Q))).sum())
        ident = kl_inc + float((bstar - b) @ np.log(u)) - float((Pstar - w) @ np.log(Z))
        worst_id = max(worst_id, abs(direct - ident))
        # (ii) the identity is algebraic: it holds for any positive class weights
        u2 = rng.uniform(0.1, 10.0, size=C)
        Z2 = Q @ u2
        Q2 = Q * u2 / Z2[:, None]
        direct2 = float((Jstar * (np.log(Q2) - np.log(Q))).sum())
        f2 = float(b @ np.log(u2) - w @ np.log(Z2))
        ident2 = f2 + float((bstar - b) @ np.log(u2)) - float((Pstar - w) @ np.log(Z2))
        worst_alg = max(worst_alg, abs(direct2 - ident2))
        if not ok:
            skipped += 1
            continue
        # (iii) with the marginal constraint met, the first term is KL(J' || J) >= 0
        J, Jp = w[:, None] * Q, w[:, None] * Qp
        worst_div = max(worst_div, abs(kl_inc - float((Jp * np.log(Jp / J)).sum())))
        min_kl = min(min_kl, kl_inc)
        # (iv) dual optimality: u maximizes F(u) = b.log u - w.log Z(u)
        dual_viol = max(dual_viol, f2 - kl_inc)
    print(f"  identity, projected weights: max |direct - identity| = {worst_id:.2e}")
    print(f"  identity, arbitrary weights: max |direct - identity| = {worst_alg:.2e}")
    print(f"  KL_inc equals KL(J'||J):     max difference          = {worst_div:.2e}")
    print(f"  smallest KL_inc over cases:  {min_kl:.3e}   (must be >= 0 up to tolerance)")
    print(f"  dual optimality: max F(random u) - F(projected u)    = {dual_viol:.2e}  (must be <= 0)")
    print(f"  cases not converged and excluded from (iii)-(iv): {skipped} of {n_cases}")
    assert worst_id < TOL and worst_alg < TOL and worst_div < TOL
    assert min_kl > -TOL and dual_viol < TOL
    assert skipped <= n_cases // 100


# ----------------------------------------------------------------------------------------------
# Part 2: special cases
# ----------------------------------------------------------------------------------------------
def rake(J, row, col, iters=5000):
    for _ in range(iters):
        J = J * (row / J.sum(1))[:, None]
        J = J * (col / J.sum(0))[None, :]
    return J


def part2(rng, n_cases):
    worst = dict(point=0.0, bayes_u=0.0, bayes_kl=0.0, calib=0.0, never=0.0)
    min_voi = np.inf
    raw_voi_neg = raw_eig_neg = 0
    for _ in range(n_cases):
        C = int(rng.integers(2, 6))
        K = int(rng.integers(2, 8))
        b = floor_norm(rng.dirichlet(np.ones(C)))
        Q = floor_norm(rng.dirichlet(np.ones(C), size=K))
        w = floor_norm(rng.dirichlet(np.ones(K)))
        # (a) exact re-exposure: the world model puts all mass on one outcome -> belief returns to prior
        k = int(rng.integers(K))
        wk = np.zeros(K)
        wk[k] = 1.0
        _, _, Qp, _, ok = project(b, wk, Q)
        worst["point"] = max(worst["point"], np.abs(Qp[k] - b).max())
        # (b) a Bayes-consistent pair is unchanged
        Jb = rng.dirichlet(np.ones(K * C)).reshape(K, C)
        bb, wb = Jb.sum(0), Jb.sum(1)
        Qb = Jb / wb[:, None]
        u, _, Qp, kl_inc, ok = project(bb, wb, Qb)
        worst["bayes_u"] = max(worst["bayes_u"], np.abs(Qp - Qb).max())
        worst["bayes_kl"] = max(worst["bayes_kl"], abs(kl_inc))
        # (c) coherent value of information is non-negative for any utility table
        u, Z, Qp, kl_inc, ok = project(b, w, Q)
        if ok:
            U = rng.normal(size=(int(rng.integers(2, 6)), C))
            phi = lambda cc: (cc @ U.T).max(-1)
            min_voi = min(min_voi, float(w @ phi(Qp) - phi(b)))
            raw_voi_neg += float(w @ phi(Q) - phi(b)) < -1e-12
            ent = lambda cc: -(cc * np.log(cc)).sum(-1)
            raw_eig_neg += float(ent(b) - w @ ent(Q)) < -1e-12
            # (d) truth with calibrated marginals (b, w): the gain equals KL_inc
            Jc = rake(rng.dirichlet(np.ones(K * C)).reshape(K, C), w, b)
            gain = float((Jc * (np.log(Qp) - np.log(Q))).sum())
            worst["calib"] = max(worst["calib"], abs(gain - kl_inc))
        # (e) the degenerate scale rule: tempering increments flat drives KL_inc to zero
        tau = 1e7
        Qt = b * (Q / b) ** (1.0 / tau)
        Qt = Qt / Qt.sum(1, keepdims=True)
        _, _, _, kl_t, _ = project(b, w, Qt)
        worst["never"] = max(worst["never"], abs(kl_t))
    print(f"  (a) point-mass predictive returns the prior: max error {worst['point']:.2e}")
    print(f"  (b) Bayes-consistent pair unchanged: max posterior change {worst['bayes_u']:.2e}, max KL_inc {worst['bayes_kl']:.2e}")
    print(f"  (c) coherent VOI minimum over cases: {min_voi:.3e}; raw score negative in {raw_voi_neg} cases, "
          f"raw entropy reduction negative in {raw_eig_neg} cases of {n_cases}")
    print(f"  (d) calibrated marginals: max |gain - KL_inc| = {worst['calib']:.2e}")
    print(f"  (e) never-updating reader is coherent: max KL_inc at tau=1e7 is {worst['never']:.2e}")
    assert worst["point"] < TOL and worst["bayes_u"] < 1e-8 and worst["bayes_kl"] < TOL
    assert min_voi > -TOL and worst["calib"] < 1e-8 and worst["never"] < 1e-8
    assert raw_voi_neg > 0 and raw_eig_neg > 0      # the raw pair does produce negative scores


# ----------------------------------------------------------------------------------------------
# Part 3: sequential statements by exact enumeration (T steps, K outcomes, C classes)
# ----------------------------------------------------------------------------------------------
def part3(rng, n_problems, T=3, K=3, C=3):
    worst = dict(mart=0.0, tower=0.0, ident=0.0, ident_raw=0.0)
    drift_raw = []
    paths = list(itertools.product(range(K), repeat=T))
    for _ in range(n_problems):
        c0 = floor_norm(rng.dirichlet(np.ones(C)))
        truth = rng.dirichlet(np.ones(C * K ** T)).reshape((C,) + (K,) * T)      # P*(y, o_1..o_T)
        rho = {}       # reader increment for outcome o after history h (arbitrary, history dependent)
        phat = {}      # world-model predictive after history h
        for t in range(T):
            for h in itertools.product(range(K), repeat=t):
                phat[h] = floor_norm(rng.dirichlet(np.ones(K)))
                rho[h] = np.exp(rng.normal(scale=0.8, size=(K, C)))
        coh, raw = {(): c0}, {(): c0}
        for t in range(T):
            for h in itertools.product(range(K), repeat=t):
                m = coh[h] * rho[h]
                m = m / m.sum(1, keepdims=True)
                _, _, Qp, _, ok = project(coh[h], phat[h], m)
                assert ok
                worst["mart"] = max(worst["mart"], np.abs(phat[h] @ Qp - coh[h]).max())
                r = raw[h] * rho[h]
                r = r / r.sum(1, keepdims=True)
                for o in range(K):
                    coh[h + (o,)] = Qp[o]
                    raw[h + (o,)] = r[o]
        pw = {p: float(np.prod([phat[p[:t]][p[t]] for t in range(T)])) for p in paths}   # world-model path prob
        for name, bel in (("coh", coh), ("raw", raw)):
            M = np.stack([pw[p] * bel[p] for p in paths])          # joint M[path, y]
            implied_prior = M.sum(0)
            if name == "coh":
                worst["mart"] = max(worst["mart"], np.abs(implied_prior - c0).max())
                for t in range(1, T):                               # tower property: c_t = M(y | o_1..o_t)
                    for h in itertools.product(range(K), repeat=t):
                        idx = [i for i, p in enumerate(paths) if p[:t] == h]
                        cond = M[idx].sum(0) / M[idx].sum()
                        worst["tower"] = max(worst["tower"], np.abs(cond - coh[h]).max())
            else:
                drift_raw.append(np.abs(implied_prior - c0).sum())
            pstar_o = np.array([truth[(slice(None),) + p].sum() for p in paths])
            bstar = truth.reshape(C, -1).sum(1)
            lhs = sum(pstar_o[i] * kl(truth[(slice(None),) + p] / pstar_o[i], bel[p]) for i, p in enumerate(paths))
            lik_err = 0.0
            for y in range(C):
                p_o_given_y = np.array([truth[(y,) + p] for p in paths]) / bstar[y]
                lik_err += bstar[y] * kl(p_o_given_y, M[:, y] / implied_prior[y])
            wm_err = kl(pstar_o, np.array([pw[p] for p in paths]))
            rhs = kl(bstar, implied_prior) + lik_err - wm_err
            key = "ident" if name == "coh" else "ident_raw"
            worst[key] = max(worst[key], abs(lhs - rhs))
    print(f"  corrected beliefs are a martingale under the world model, and the implied prior equals c_0: max error {worst['mart']:.2e}")
    print(f"  tower property, c_t = M(y | o_1..o_t) for every prefix: max error {worst['tower']:.2e}")
    print(f"  T-step identity for the corrected agent: max |lhs - rhs| = {worst['ident']:.2e}")
    print(f"  same identity for the raw reader with its drifted implied prior: max |lhs - rhs| = {worst['ident_raw']:.2e}")
    print(f"  raw reader, L1 distance between implied prior and c_0 after {T} steps: mean {np.mean(drift_raw):.3f}, max {np.max(drift_raw):.3f}")
    assert max(worst.values()) < 1e-8


# ----------------------------------------------------------------------------------------------
# Synthetic twin parameters (profiles.py at df467ea). Tests in file order; findings = 4 diseases + nonspecific.
# ----------------------------------------------------------------------------------------------
C = 4
PRIOR = np.array([957, 648, 257, 538], float)
PRIOR = PRIOR / PRIOR.sum()
HITS = np.array([
    (0.55, 0.50, 0.45, 0.40), (0.88, 0.60, 0.88, 0.80), (0.90, 0.80, 0.85, 0.80), (0.10, 0.10, 0.10, 0.12),
    (0.70, 0.88, 0.25, 0.30), (0.30, 0.25, 0.30, 0.20), (0.10, 0.10, 0.10, 0.30), (0.12, 0.20, 0.12, 0.40),
    (0.08, 0.08, 0.10, 0.25), (0.10, 0.65, 0.10, 0.40), (0.15, 0.05, 0.10, 0.05), (0.05, 0.05, 0.10, 0.20)])
CONF = np.array([0.08, 0.02, 0.02, 0.05, 0.04, 0.12, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05])
NT = len(CONF)
LIK = np.empty((NT, C, C + 1))                 # LIK[test, phenotype e, finding s]
for _t in range(NT):
    LIK[_t, :, :C] = CONF[_t]
    LIK[_t, np.arange(C), np.arange(C)] = HITS[_t]
    LIK[_t, :, C] = 1.0 - HITS[_t] - CONF[_t] * (C - 1)


def pe_given_y(p):
    M = np.full((C, C), p / (C - 1))
    np.fill_diagonal(M, 1.0 - p)
    return M


# ----------------------------------------------------------------------------------------------
# Part 4: reproduction of the November 12 probe (loop form, identical random-call order)
# ----------------------------------------------------------------------------------------------
def probe(p_atyp, reader, n=4000, steps=8, seed=0, delta=0.4, wm="oracle"):
    rng = np.random.default_rng(seed)
    PEY = pe_given_y(p_atyp)
    NB = np.einsum("ye,tes->tys", PEY, LIK)          # class-conditional independence model p(s | y)
    joint_ye = PRIOR[:, None] * PEY
    nll = {k: np.zeros(steps + 1) for k in ("raw", "cee", "bayes")}
    acc = {k: np.zeros(steps + 1) for k in nll}
    kl_sum = 0.0
    for _ in range(n):
        y = rng.choice(C, p=PRIOR)
        e = rng.choice(C, p=PEY[y])
        order = rng.permutation(NT)[:steps]
        w_ye = joint_ye.copy()
        b = PRIOR.copy()
        c = PRIOR.copy()

        def score(k, bel, t):
            nll[k][t] += -np.log(max(bel[y], 1e-12))
            acc[k][t] += (np.argmax(bel) == y)

        score("raw", b, 0); score("cee", c, 0); score("bayes", w_ye.sum(1), 0)
        for t_i, ti in enumerate(order):
            L = LIK[ti]
            s = rng.choice(C + 1, p=L[e])
            P = w_ye.sum(0) @ L if wm == "oracle" else b @ NB[ti]

            def react(bel, s_):
                q = bel * NB[ti][:, s_]
                if reader == "entrench":
                    lo = np.log(q / q.sum()); lo[np.argmax(bel)] += delta; q = np.exp(lo)
                elif reader == "fixed":
                    lo = np.log(q / q.sum()); lo[1] += delta; q = np.exp(lo)
                elif reader == "over":
                    q = bel * NB[ti][:, s_] ** 2.0
                return q / q.sum()

            Q = np.stack([react(b, s_) for s_ in range(C + 1)])
            M = c * (Q / b)
            M = M / M.sum(1, keepdims=True)
            u = np.ones(C)
            for _it in range(2000):
                Z = M @ u
                mean = P @ (M * u / Z[:, None])
                if np.abs(mean - c).max() < 1e-12:
                    break
                u = u * c / mean
                u = u / u.sum()
            Z = M @ u
            kl_sum += c @ np.log(u) - P @ np.log(Z)
            c = M[s] * u / Z[s]
            b = Q[s]
            w_ye = w_ye * L[:, s][None, :]
            w_ye = w_ye / w_ye.sum()
            score("raw", b, t_i + 1); score("cee", c, t_i + 1); score("bayes", w_ye.sum(1), t_i + 1)
    return ({k: v / n for k, v in nll.items()}, {k: v / n for k, v in acc.items()}, kl_sum / (n * steps))


REFERENCE = {   # values reported on November 12: (raw NLL@8, cee NLL@8, raw acc@8, cee acc@8)
    (0.07, "nb"): (0.543, 0.524, 0.882, 0.868),
    (0.07, "entrench"): (0.761, 0.524, 0.842, 0.868),
    (0.07, "fixed"): (0.695, 0.524, 0.817, 0.868),
    (0.07, "over"): (0.947, 1.60, 0.882, 0.796),
    (0.20, "nb"): (1.038, 0.903, 0.756, 0.723),
}


def part4(quick):
    n = 600 if quick else 4000
    ok = True
    for (p, reader), ref in REFERENCE.items():
        nll, acc, kl_mean = probe(p, reader, n=n)
        got = (nll["raw"][8], nll["cee"][8], acc["raw"][8], acc["cee"][8])
        match = all(abs(g - r) < (0.006 if r > 1.5 else 0.0015) for g, r in zip(got, ref))
        ok &= match
        print(f"  p_atyp={p:.2f} {reader:9s} NLL@8 raw {got[0]:.3f} -> projected {got[1]:.3f} | acc@8 raw {got[2]:.3f} -> "
              f"projected {got[3]:.3f} | Bayes NLL {nll['bayes'][8]:.3f} acc {acc['bayes'][8]:.3f} | mean KL_inc {kl_mean:.4f}"
              + ("" if quick else ("   matches Nov 12" if match else "   DIFFERS from Nov 12")))
    nll, acc, kl_mean = probe(0.07, "nb", n=n, wm="self")
    print(f"  control, reader's own predictive as world model: NLL@8 raw {nll['raw'][8]:.6f} projected {nll['cee'][8]:.6f} "
          f"mean KL_inc {kl_mean:.2e}")
    assert abs(nll["raw"][8] - nll["cee"][8]) < 1e-9 and abs(kl_mean) < 1e-9
    if not quick:
        assert ok, "probe numbers differ from the November 12 report"


# ----------------------------------------------------------------------------------------------
# Part 5: scale repair by information-budget matching (vectorized over patients)
# ----------------------------------------------------------------------------------------------
def project_batch(c, P, M, u=None, iters=600, tol=1e-10):
    u = np.ones_like(c) if u is None else u
    for _ in range(iters):
        Z = np.einsum("nky,ny->nk", M, u)
        mean = np.einsum("nk,nky->ny", P, M * u[:, None, :] / Z[:, :, None])
        if np.abs(mean - c).max() < tol:
            break
        u = u * c / mean
        u = u / u.sum(1, keepdims=True)
    Z = np.einsum("nky,ny->nk", M, u)
    return u, M * u[:, None, :] / Z[:, :, None]


def react_batch(reader, b, NBt, delta=0.4):
    like = np.transpose(NBt, (0, 2, 1))                    # [n, finding, class]
    if reader == "over":
        like = like ** 2.0
    elif reader == "under":
        like = like ** 0.5
    q = b[:, None, :] * like
    q = q / q.sum(2, keepdims=True)
    if reader in ("entrench", "fixed"):
        lo = np.log(q)
        cls = np.argmax(b, 1) if reader == "entrench" else np.ones(len(b), int)
        lo[np.arange(len(b)), :, cls] += delta
        q = np.exp(lo)
        q = q / q.sum(2, keepdims=True)
    return q


def scale_study(p_atyp, reader, mode, n, steps=8, seed=7, mult=1.0, shrink_only=False, noise=0.0):
    """mode: raw | proj | budget_true | budget_nb. Returns NLL@steps, acc@steps, mean tau, Bayes NLL, Bayes acc.

    shrink_only restricts the evidence scale to tau >= 1 (increments may be flattened, never amplified).
    noise multiplies each decision point's budget by an independent lognormal factor with that sigma.
    """
    rng = np.random.default_rng(seed)
    PEY = pe_given_y(p_atyp)
    NB = np.einsum("ye,tes->tys", PEY, LIK)
    y = rng.choice(C, size=n, p=PRIOR)
    e = (rng.random(n)[:, None] > np.cumsum(PEY[y], 1)).sum(1)
    order = np.argsort(rng.random((n, NT)), 1)[:, :steps]
    us = rng.random((n, steps))
    w = np.tile(PRIOR[:, None] * PEY, (n, 1, 1))          # oracle joint over (y, e)
    b = np.tile(PRIOR, (n, 1))
    c = b.copy()
    ar = np.arange(n)
    taus = []
    for t in range(steps):
        tid = order[:, t]
        Lt, NBt = LIK[tid], NB[tid]
        s = (us[:, t][:, None] > np.cumsum(Lt[ar, e], 1)).sum(1)
        Jys = np.einsum("nye,nes->nys", w, Lt)             # true joint over (y, finding) given history
        P = Jys.sum(1)
        Q = react_batch(reader, b, NBt)
        if mode != "raw":
            rho = Q / b[:, None, :]
            if mode == "proj":
                M = c[:, None, :] * rho
                M = M / M.sum(2, keepdims=True)
                _, Qp = project_batch(c, P, M)
                taus.append(np.ones(n))
            else:
                if mode == "budget_true":
                    wy = w.sum(2)
                    post = Jys / P[:, None, :]
                    budget = (Jys * np.log(post / wy[:, :, None])).sum((1, 2))
                else:                                       # class-conditional independence budget at belief c
                    Jn = c[:, :, None] * NBt
                    budget = (Jn * np.log(Jn / Jn.sum(1, keepdims=True) / c[:, :, None])).sum((1, 2))
                budget = budget * mult
                if noise > 0:
                    budget = budget * np.exp(noise * np.random.default_rng(seed + 1000 + t).normal(size=n))
                lo_t = np.full(n, 0.0 if shrink_only else np.log(1 / 8.0))
                hi_t = np.full(n, np.log(16.0))
                u = None
                for _ in range(14):
                    mid = 0.5 * (lo_t + hi_t)
                    M = c[:, None, :] * rho ** (1.0 / np.exp(mid))[:, None, None]
                    M = M / M.sum(2, keepdims=True)
                    u, Qp = project_batch(c, P, M, u)
                    resp = np.einsum("nk,nk->n", P, (Qp * np.log(Qp / c[:, None, :])).sum(2))
                    too_strong = resp > budget              # shrink increments: larger tau
                    lo_t = np.where(too_strong, mid, lo_t)
                    hi_t = np.where(too_strong, hi_t, mid)
                tau = np.exp(0.5 * (lo_t + hi_t))
                M = c[:, None, :] * rho ** (1.0 / tau)[:, None, None]
                M = M / M.sum(2, keepdims=True)
                _, Qp = project_batch(c, P, M, u)
                taus.append(tau)
            c = Qp[ar, s]
        b = Q[ar, s]
        w = w * Lt[ar, :, s][:, None, :]
        w = w / w.sum((1, 2), keepdims=True)
    bel = b if mode == "raw" else c
    bay = w.sum(2)
    out = (float(-np.log(np.maximum(bel[ar, y], 1e-12)).mean()), float((bel.argmax(1) == y).mean()),
           float(np.exp(np.mean(np.log(np.concatenate(taus))))) if taus else float("nan"),
           float(-np.log(bay[ar, y]).mean()), float((bay.argmax(1) == y).mean()))
    return out


def part5(quick):
    n = 500 if quick else 3000
    readers = ("nb", "entrench", "fixed", "over", "under")
    cols = (
        ("raw", dict(mode="raw")),
        ("projection only", dict(mode="proj")),
        ("scale+proj, true budget", dict(mode="budget_true")),
        ("true budget x2", dict(mode="budget_true", mult=2.0)),
        ("shrink-only, true budget", dict(mode="budget_true", shrink_only=True)),
        ("shrink-only, budget x0.5", dict(mode="budget_true", shrink_only=True, mult=0.5)),
        ("shrink-only, budget x2", dict(mode="budget_true", shrink_only=True, mult=2.0)),
        ("shrink-only, noisy budget", dict(mode="budget_true", shrink_only=True, noise=0.7)),
        ("shrink-only, independence budget", dict(mode="budget_nb", shrink_only=True)),
    )
    res = {}
    for p in (0.07, 0.20):
        print(f"  p_atypical = {p:.2f}, n = {n}, 8 steps, oracle world model, stateful track")
        for r in readers:
            for nm, kw in cols:
                res[(p, r, nm)] = scale_study(p, r, n=n, **kw)
        bayes = res[(p, readers[0], "raw")][3:]
        print(f"    Bayes reference: NLL {bayes[0]:.3f}, accuracy {bayes[1]:.3f}")
        print("    " + "variant".ljust(34) + "".join(r.rjust(22) for r in readers) + "    (NLL / accuracy / geometric-mean tau)")
        for nm, _ in cols:
            row = ""
            for r in readers:
                v = res[(p, r, nm)]
                row += (f"{v[0]:.3f}/{v[1]:.3f}" + ("" if np.isnan(v[2]) or nm == "projection only" else f"/{v[2]:.2f}")).rjust(22)
            print("    " + nm.ljust(34) + row)
    # asserted regularities of this probe (direction only; the numbers are sandbox numbers)
    for p in (0.07, 0.20):
        for r in readers:
            raw, proj = res[(p, r, "raw")][0], res[(p, r, "projection only")][0]
            best = res[(p, r, "scale+proj, true budget")][0]
            assert best < raw and best < proj, (p, r)
        for r in ("entrench", "fixed"):
            assert res[(p, r, "projection only")][0] < res[(p, r, "raw")][0]
        for r in ("over", "under"):
            assert res[(p, r, "projection only")][0] > res[(p, r, "raw")][0]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=20261113)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    np.seterr(all="raise", under="ignore")
    print("Part 1  gain identity on random cases")
    part1(rng, 300 if args.quick else 3000)
    print("Part 2  special cases")
    part2(rng, 200 if args.quick else 1500)
    print("Part 3  sequential statements by exact enumeration")
    part3(rng, 20 if args.quick else 200)
    print("Part 4  reproduction of the November 12 probe (twin parameters from profiles.py)")
    part4(args.quick)
    print("Part 5  scale repair by information-budget matching")
    part5(args.quick)
    print(f"ALL ASSERTED CHECKS PASSED in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    sys.exit(main())
