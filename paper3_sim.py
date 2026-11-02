# -*- coding: utf-8 -*-
"""
paper3_sim.py

Bai bao 3: Kiem chung vet can tinh an toan cua bang Q da hoc cho thiet bi bay khong nguoi lai.

Y tuong: sau khi luong tu hoa khong gian trang thai lien tuc, chinh sach tham lam lay tu bang Q
la mot ham chi phu thuoc vao o luong tu. Anh cua mot o duoi mot hanh dong co bao loi dang dong
kin, nen do thi o - o sinh ra tu phep bao loi nay la mot phep xap xi tren dung dan. Tren do thi
do, diem bat dong lon nhat cho tap o ma moi quy dao lien tuc xuat phat tu do deu khong bao gio
va cham. Tu tap nay sinh ra mat na hanh dong duoc chung nhan, dung truc tiep trong luc hoc, nen
qua trinh hoc khong con lan va cham nao va chinh sach thu duoc an toan cho moi dieu kien dau.

Ban sua sau phan bien (bai 1280, JST) bo sung:
  - kiem tra Bo de 1, Bo de 2 va Dinh ly 1 bang lay mau, vi du lam tron so dau phay dong;
  - so sanh voi ba phep truu tuong hoa khong dung dan (diem tam, 27 diem mau, o nua mo);
  - chung nhan bang Q tai cac moc huan luyen, ban do chinh sach va gia tri;
  - kiem tra tinh khong thu nguyen cua mo hinh;
  - mo rong ba chieu voi ba moi truong co do phuc tap tang dan (E1, E2, E3 dang ban do be mat so).

Chay:  python3 paper3_sim.py [--quick] [--replot]
Ket qua: ../figs/p3/results.json, hinh trong ../figs/p3/en va ../figs/p3/vi.

Tac gia: Nguyen Van Bang, Khoa Ten lua, Hoc vien Phong khong-Khong quan, Ha Noi, Viet Nam.
"""
import os
import sys
import math
import time
import json
import pickle
import argparse
from collections import deque

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

SEED = 2026
HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "figs", "p3")
os.makedirs(FIG, exist_ok=True)

# ============ 1. THAM SO ============
CFG = dict(v=1.0, dt=1.0, wmax=math.radians(30.0), safeR=2.0,
           W=60.0, H=60.0, goal=(55.0, 5.0), start=(5.0, 55.0, math.radians(-60.0)),
           goal_rad=1.5, Tmax=600,
           R_goal=100.0, R_hit=-150.0, R_timeout=-60.0,
           lam=0.05, kappa=0.08, beta=1.5,
           eta0=0.15, eta_min=0.02, eps0=1.0, eps_min=0.05)

ACT = np.array([-CFG["wmax"], 0.0, CFG["wmax"]])
NA = 3
LSTEP = CFG["v"] * CFG["dt"]
RPOLY = LSTEP / (2.0 * math.sin(CFG["wmax"] * CFG["dt"] / 2.0))

SCEN = {
    "S1": [(20.0, 40.0, 9.0), (38.0, 18.0, 10.0), (25.0, 10.0, 4.0)],
    "S2": [(18.0, 42.0, 8.0), (34.0, 30.0, 7.0), (46.0, 16.0, 6.0), (12.0, 20.0, 5.0)],
}


def wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def step(s, a):
    x, y, th = s
    t2 = wrap(th + ACT[a] * CFG["dt"])
    return (x + LSTEP * math.cos(t2), y + LSTEP * math.sin(t2), t2)


# ============ 2. BAO LOI CHINH XAC CUA ANH MOT O ============
def cos_range(a, b):
    """Gia tri nho nhat va lon nhat cua cosin tren doan [a, b]."""
    lo = min(math.cos(a), math.cos(b))
    hi = max(math.cos(a), math.cos(b))
    k = math.ceil((a - math.pi) / (2 * math.pi))
    if a <= math.pi + 2 * math.pi * k <= b:
        lo = -1.0
    k = math.ceil(a / (2 * math.pi))
    if a <= 2 * math.pi * k <= b:
        hi = 1.0
    return lo, hi


def sin_range(a, b):
    return cos_range(a - math.pi / 2.0, b - math.pi / 2.0)


class Grid:
    """Luong tu hoa deu tren (x, y, theta)."""

    def __init__(self, cell=1.0, nth=24):
        r = CFG["wmax"] * CFG["dt"] / (2.0 * math.pi / nth)
        if abs(r - round(r)) > 1e-9:
            raise ValueError("so o goc phai chia het goc quay mot buoc")
        self.cell, self.nth = cell, nth
        self.nx = int(round(CFG["W"] / cell))
        self.ny = int(round(CFG["H"] / cell))
        self.dth = 2.0 * math.pi / nth
        self.kshift = np.array([-int(round(r)), 0, int(round(r))])
        self.ncell = self.nx * self.ny * self.nth

    def phi(self, s):
        x, y, th = s
        i = min(self.nx - 1, max(0, int(math.floor(x / self.cell))))
        j = min(self.ny - 1, max(0, int(math.floor(y / self.cell))))
        k = int(math.floor((wrap(th) + math.pi) / self.dth + TOL)) % self.nth
        return i, j, k

    def offsets(self, k, a):
        """Do lech lon nhat va nho nhat cua diem cuoi so voi goc o, tinh theo met."""
        # doan goc cua o duoc noi them ETA ve hai phia de bao phu sai so lam tron
        lo = -math.pi + k * self.dth + ACT[a] * CFG["dt"] - ETA
        hi = lo + self.dth + 2.0 * ETA
        cl, ch = cos_range(lo, hi)
        sl, sh = sin_range(lo, hi)
        return LSTEP * cl, LSTEP * ch, LSTEP * sl, LSTEP * sh


TOL = 1e-9          # dung sai khi lay phan nguyen goc huong, tinh theo be rong o goc
ETA = 1e-9          # noi rong doan goc huong, rad
EPS = 1e-9          # noi rong hop anh, m


def cell_range(lo, hi, c):
    """Chi so cac o dong [ic, (i+1)c] cham doan dong [lo - EPS, hi + EPS].

    Xem o la hop dong va noi rong hop anh them EPS lam cho phep truu tuong hoa mien nhiem voi
    sai so lam tron cua so dau phay dong, ke ca khi mot canh cua hop anh roi dung len duong luoi.
    """
    a = int(math.ceil((lo - EPS) / c)) - 1
    b = int(math.floor((hi + EPS) / c))
    return list(range(a, b + 1))


def _box_hits(x0, x1, y0, y1, threats, extra):
    bad = None
    for cx, cy, r in threats:
        rho = r + extra
        dx = np.maximum(np.maximum(x0 - cx, cx - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - cy, cy - y1), 0.0)
        h = (dx * dx + dy * dy) <= rho * rho
        bad = h if bad is None else (bad | h)
    return bad


# ============ 3. BO KIEM CHUNG ============
class Verifier:
    """Do thi o - o sinh tu bao loi chinh xac, kem cac phep tinh diem bat dong."""

    def __init__(self, grid, threats, extra=None):
        self.g = grid
        self.threats = threats
        self.extra = CFG["safeR"] if extra is None else extra
        g = grid
        c = g.cell
        I = (np.arange(g.nx)[:, None]) * c
        J = (np.arange(g.ny)[None, :]) * c
        self.bad = np.zeros((g.nth, NA, g.nx, g.ny), dtype=bool)
        self.off = {}
        for k in range(g.nth):
            for a in range(NA):
                dxl, dxh, dyl, dyh = g.offsets(k, a)
                self.bad[k, a] = _box_hits(I + min(0.0, dxl) - EPS, I + c + max(0.0, dxh) + EPS,
                                           J + min(0.0, dyl) - EPS, J + c + max(0.0, dyh) + EPS,
                                           threats, self.extra)
                di = cell_range(dxl, c + dxh, c)
                dj = cell_range(dyl, c + dyh, c)
                self.off[(k, a)] = (di, dj, (k + g.kshift[a]) % g.nth)
        self.pad = max(4, 2 + int(math.ceil(LSTEP / c)))
        self.edges = sum(len(v[0]) * len(v[1]) for v in self.off.values()) * g.nx * g.ny

    def _all_in(self, S, k, a):
        di, dj, kn = self.off[(k, a)]
        p, g = self.pad, self.g
        P = np.zeros((g.nx + 2 * p, g.ny + 2 * p), dtype=bool)
        P[p:p + g.nx, p:p + g.ny] = S[:, :, kn]
        out = np.ones((g.nx, g.ny), dtype=bool)
        for u in di:
            for w in dj:
                out &= P[p + u:p + u + g.nx, p + w:p + w + g.ny]
        return out

    def verify_policy(self, PI, maxit=400):
        """Tap o lon nhat vua bat bien tien duoi PI vua khong bao gio va cham."""
        g = self.g
        S = np.ones((g.nx, g.ny, g.nth), dtype=bool)
        hist = []
        for _ in range(maxit):
            new = np.zeros_like(S)
            for k in range(g.nth):
                for a in range(NA):
                    sel = (PI[:, :, k] == a)
                    if not sel.any():
                        continue
                    new[:, :, k] |= sel & (~self.bad[k, a]) & self._all_in(S, k, a)
            new &= S
            hist.append(int(new.sum()))
            if np.array_equal(new, S):
                break
            S = new
        return S, hist

    def max_invariant(self, maxit=400):
        """Tap bat bien co dieu khien lon nhat tren luoi o."""
        g = self.g
        S = np.ones((g.nx, g.ny, g.nth), dtype=bool)
        hist = []
        for _ in range(maxit):
            new = np.zeros_like(S)
            for k in range(g.nth):
                acc = np.zeros((g.nx, g.ny), dtype=bool)
                for a in range(NA):
                    acc |= (~self.bad[k, a]) & self._all_in(S, k, a)
                new[:, :, k] = acc
            new &= S
            hist.append(int(new.sum()))
            if np.array_equal(new, S):
                break
            S = new
        return S, hist

    def mask(self, S):
        """Mat na hanh dong duoc chung nhan tren toan bo luoi o."""
        g = self.g
        M = np.zeros((g.nx, g.ny, g.nth, NA), dtype=bool)
        for k in range(g.nth):
            for a in range(NA):
                M[:, :, k, a] = (~self.bad[k, a]) & self._all_in(S, k, a)
        return M

    def repair(self, PI, S, Q=None):
        """Giu nguyen hanh dong da hoc khi no an toan, neu khong thi chon thay the tot nhat."""
        g = self.g
        PR = PI.copy()
        changed = np.zeros((g.nx, g.ny, g.nth), dtype=bool)
        ok = {(k, a): (~self.bad[k, a]) & self._all_in(S, k, a)
              for k in range(g.nth) for a in range(NA)}
        for k in range(g.nth):
            cur = np.zeros((g.nx, g.ny), dtype=bool)
            for a in range(NA):
                cur |= (PI[:, :, k] == a) & ok[(k, a)]
            need = S[:, :, k] & ~cur
            if Q is None:
                for a in range(NA):
                    take = need & ok[(k, a)]
                    PR[:, :, k][take] = a
                    changed[:, :, k] |= take
                    need = need & ~take
            else:
                M = np.full((g.nx, g.ny, NA), -np.inf)
                for a in range(NA):
                    M[:, :, a] = np.where(ok[(k, a)], Q[:, :, k, a], -np.inf)
                best = np.argmax(M, axis=2).astype(np.int8)
                take = need & np.isfinite(M.max(axis=2))
                PR[:, :, k][take] = best[take]
                changed[:, :, k] |= take
        return PR, changed

    def reachable(self, M, cell0):
        """Cac o co the den duoc tu cell0 khi chi dung cac hanh dong trong mat na M."""
        g = self.g
        vis = np.zeros((g.nx, g.ny, g.nth), dtype=bool)
        dq = deque([cell0])
        vis[cell0] = True
        while dq:
            i, j, k = dq.popleft()
            for a in range(NA):
                if not M[i, j, k, a]:
                    continue
                di, dj, kn = self.off[(k, a)]
                for u in di:
                    for w in dj:
                        n = (i + u, j + w, kn)
                        if 0 <= n[0] < g.nx and 0 <= n[1] < g.ny and not vis[n]:
                            vis[n] = True
                            dq.append(n)
        return vis


# ============ 4. HINH HOC CHINH XAC VA BO LOC TRUC TUYEN ============
def seg_hits(p0, p1, o, rho):
    dx = p1[0] - p0[0]; dy = p1[1] - p0[1]
    ax = p0[0] - o[0]; ay = p0[1] - o[1]
    dd = dx * dx + dy * dy
    t = 0.0 if dd == 0.0 else -(ax * dx + ay * dy) / dd
    t = min(1.0, max(0.0, t))
    qx = ax + t * dx; qy = ay + t * dy
    return (qx * qx + qy * qy) <= rho * rho


def clear_exact(p0, p1, threats, extra):
    if not (0.0 <= p1[0] <= CFG["W"] and 0.0 <= p1[1] <= CFG["H"]):
        return False
    for cx, cy, r in threats:
        if seg_hits(p0, p1, (cx, cy), r + extra):
            return False
    return True


def orbit_c(s, sign):
    x, y, th = s
    a = th + sign * math.pi / 2.0
    return (x + RPOLY * math.cos(a), y + RPOLY * math.sin(a))


def viable(s, threats):
    for sg in (+1, -1):
        cx, cy = orbit_c(s, sg)
        if not (RPOLY <= cx <= CFG["W"] - RPOLY and RPOLY <= cy <= CFG["H"] - RPOLY):
            continue
        ok = True
        for ox, oy, r in threats:
            if math.hypot(cx - ox, cy - oy) < RPOLY + r + CFG["safeR"]:
                ok = False
                break
        if ok:
            return True
    return False


def allowed(s, threats, mode):
    if mode == "off":
        return [0, 1, 2]
    out = []
    for a in range(NA):
        sn = step(s, a)
        if not clear_exact(s[:2], sn[:2], threats, CFG["safeR"]):
            continue
        if mode == "cert" and not viable(sn, threats):
            continue
        out.append(a)
    return out


# ============ 5. HUAN LUYEN ============
def train(grid, threats, mode="none", episodes=6000, seed=SEED, mask=None, log=0, snap=0):
    """Hoc Q dang bang. snap > 0: luu ban sao bang Q sau moi snap tap (khong dung toi bo sinh
    so ngau nhien, nen ket qua trung khop tung chu so voi khi khong luu)."""
    rng = np.random.default_rng(seed)
    snaps = []
    Q = np.zeros((grid.nx, grid.ny, grid.nth, NA), dtype=np.float32)
    seen = np.zeros((grid.nx, grid.ny, grid.nth), dtype=np.int32)
    gx, gy = CFG["goal"]
    eps, eta = CFG["eps0"], CFG["eta0"]
    ed = math.exp(math.log(CFG["eps_min"] / CFG["eps0"]) / (0.75 * episodes))
    nd = math.exp(math.log(CFG["eta_min"] / CFG["eta0"]) / (0.85 * episodes))
    succ = np.zeros(episodes); coll = np.zeros(episodes); steps = np.zeros(episodes)
    dead = 0
    t0 = time.time()
    for ep in range(episodes):
        s = CFG["start"]
        d0 = math.hypot(s[0] - gx, s[1] - gy)
        hit = reach = False
        t = 0
        for t in range(1, CFG["Tmax"] + 1):
            key = grid.phi(s)
            seen[key] += 1
            if mode == "mask":
                A = np.nonzero(mask[key])[0].tolist()
            else:
                A = allowed(s, threats, mode)
            if not A:
                A = list(range(NA))
                dead += 1
            if rng.random() < eps:
                a = int(A[rng.integers(len(A))])
            else:
                q = Q[key][A]
                m = q.max()
                cand = [A[u] for u in range(len(A)) if q[u] >= m - 1e-9]
                a = int(cand[rng.integers(len(cand))])
            sn = step(s, a)
            ok = clear_exact(s[:2], sn[:2], threats, CFG["safeR"])
            r = -CFG["lam"] - CFG["kappa"] * abs(ACT[a]) / CFG["wmax"]
            done = False
            if not ok:
                r = CFG["R_hit"]; done = True; hit = True
            else:
                d1 = math.hypot(sn[0] - gx, sn[1] - gy)
                r += CFG["beta"] * (d0 - d1)
                d0 = d1
                if d1 <= CFG["goal_rad"]:
                    r = CFG["R_goal"]; done = True; reach = True
            if (not done) and t == CFG["Tmax"]:
                r += CFG["R_timeout"]; done = True
            if done:
                y = r
            else:
                kn = grid.phi(sn)
                An = (np.nonzero(mask[kn])[0].tolist() if mode == "mask"
                      else allowed(sn, threats, mode)) or list(range(NA))
                y = r + float(Q[kn][An].max())
            Q[key][a] += eta * (y - Q[key][a])
            if done:
                break
            s = sn
        succ[ep] = reach; coll[ep] = hit; steps[ep] = t
        if snap and (ep + 1) % snap == 0:
            snaps.append((ep + 1, Q.copy()))
        eps = max(CFG["eps_min"], eps * ed)
        eta = max(CFG["eta_min"], eta * nd)
        if log and (ep + 1) % log == 0:
            print("    ep %5d succ=%.3f coll=%.3f"
                  % (ep + 1, succ[max(0, ep - 999):ep + 1].mean(),
                     coll[max(0, ep - 999):ep + 1].mean()), flush=True)
    return dict(Q=Q, seen=seen, succ=succ, coll=coll, steps=steps, dead=dead,
                wall=time.time() - t0, snaps=snaps)


def greedy(Q, tie=1, mask=None):
    """Chinh sach tham lam tren o. Neu co mat na thi chi lay arg max trong tap duoc phep."""
    if mask is None:
        PI = np.argmax(Q, axis=3).astype(np.int8)
        PI[Q.max(axis=3) == Q.min(axis=3)] = tie
        return PI
    M = np.where(mask, Q.astype(np.float64), -np.inf)
    PI = np.argmax(M, axis=3).astype(np.int8)
    PI[~mask.any(axis=3)] = tie
    return PI


def run_cell_policy(PI, grid, threats, s0=None, Tmax=None):
    """Chay chinh sach o thuan tuy, khong dung bat ky phep tinh hinh hoc nao khi bay."""
    s = s0 or CFG["start"]
    gx, gy = CFG["goal"]
    Tmax = Tmax or CFG["Tmax"]
    P = [s[:2]]
    hit = False
    mc = 1e9
    for _ in range(Tmax):
        a = int(PI[grid.phi(s)])
        sn = step(s, a)
        if not clear_exact(s[:2], sn[:2], threats, 0.0):
            hit = True
        for ox, oy, r in threats:
            mc = min(mc, math.hypot(sn[0] - ox, sn[1] - oy) - r)
        s = sn
        P.append(s[:2])
        if hit:
            break
        if math.hypot(s[0] - gx, s[1] - gy) <= CFG["goal_rad"]:
            break
    P = np.array(P)
    return dict(path=P, hit=hit,
                reached=math.hypot(P[-1][0] - gx, P[-1][1] - gy) <= CFG["goal_rad"] and not hit,
                length=float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
                minclear=float(mc), steps=len(P) - 1)


def run_online(Q, grid, threats, mode, s0=None, Tmax=None):
    """Chay chinh sach tham lam kem bo loc hinh hoc truc tuyen."""
    s = s0 or CFG["start"]
    gx, gy = CFG["goal"]
    Tmax = Tmax or CFG["Tmax"]
    P = [s[:2]]
    hit = False
    mc = 1e9
    for _ in range(Tmax):
        A = allowed(s, threats, mode) or list(range(NA))
        a = int(A[int(np.argmax(Q[grid.phi(s)][A]))])
        sn = step(s, a)
        if not clear_exact(s[:2], sn[:2], threats, 0.0):
            hit = True
        for ox, oy, r in threats:
            mc = min(mc, math.hypot(sn[0] - ox, sn[1] - oy) - r)
        s = sn
        P.append(s[:2])
        if hit:
            break
        if math.hypot(s[0] - gx, s[1] - gy) <= CFG["goal_rad"]:
            break
    P = np.array(P)
    return dict(path=P, hit=hit,
                reached=math.hypot(P[-1][0] - gx, P[-1][1] - gy) <= CFG["goal_rad"] and not hit,
                length=float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
                minclear=float(mc), steps=len(P) - 1)


# ============ 6. KIEM CHUNG BANG LAY MAU DE DOI CHIEU ============
def _free_start(x, y, threats):
    for cx, cy, r in threats:
        if math.hypot(x - cx, y - cy) <= r + CFG["safeR"]:
            return False
    return True


def monte_carlo(PI, grid, threats, S, n=20000, horizon=400, seed=7, inside=True):
    """Lay mau trang thai dau khong nam trong hinh tron noi rong, chay chinh sach o thuan tuy.

    Mot lan vi pham la khi day cung cham hinh tron da noi rong hoac roi khoi hinh vuong nhiem vu,
    tuc dung tinh chat ma chung nhan bao dam.
    """
    rng = np.random.default_rng(seed)
    bad = 0
    tot = 0
    while tot < n:
        x = rng.uniform(0.0, CFG["W"]); y = rng.uniform(0.0, CFG["H"])
        th = rng.uniform(-math.pi, math.pi)
        c = grid.phi((x, y, th))
        if bool(S[c]) != inside or not _free_start(x, y, threats):
            continue
        tot += 1
        s = (x, y, th)
        for _ in range(horizon):
            a = int(PI[grid.phi(s)])
            sn = step(s, a)
            if not clear_exact(s[:2], sn[:2], threats, CFG["safeR"]):
                bad += 1
                break
            s = sn
    return bad, tot


def sampling_curve(PI, grid, threats, S, budgets, horizon=400, seed=11):
    """So diem xuat phat khong an toan ma phep lay mau phan bo deu tim duoc theo ngan sach."""
    rng = np.random.default_rng(seed)
    out = []
    found = 0
    drawn = 0
    marks = sorted(budgets)
    mi = 0
    while drawn < marks[-1]:
        x = rng.uniform(0.0, CFG["W"]); y = rng.uniform(0.0, CFG["H"])
        th = rng.uniform(-math.pi, math.pi)
        if not _free_start(x, y, threats):
            continue
        drawn += 1
        s = (x, y, th)
        for _ in range(horizon):
            a = int(PI[grid.phi(s)])
            sn = step(s, a)
            if not clear_exact(s[:2], sn[:2], threats, CFG["safeR"]):
                found += 1
                break
            s = sn
        while mi < len(marks) and drawn >= marks[mi]:
            out.append((marks[mi], found))
            mi += 1
    return out


# ============ 7. QUY CHUAN HINH VE ============
CM = 1.0 / 2.54
SERIF = "Liberation Serif"
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [SERIF, 'Nimbus Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'custom', 'mathtext.rm': SERIF, 'mathtext.it': SERIF + ':italic',
    'mathtext.bf': SERIF + ':bold', 'mathtext.cal': SERIF + ':italic',
    'font.size': 10, 'axes.labelsize': 10, 'axes.titlesize': 9.5,
    'xtick.labelsize': 9, 'ytick.labelsize': 9, 'axes.linewidth': 0.8,
    'lines.linewidth': 1.4, 'legend.fontsize': 8.0, 'savefig.dpi': 600, 'figure.dpi': 110,
})
OK = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#56B4E9', '#000000']
LANG = "en"

_LEG = {
    "raw": ("No action mask", "Không có mặt nạ hành động"),
    "step1": ("One-step exact filter", "Bộ lọc một bước chính xác"),
    "online": ("Backup-orbit filter", "Bộ lọc quỹ đạo dự phòng"),
    "mask": ("Certified mask, proposed", "Mặt nạ có chứng nhận, đề xuất"),
    "repair": ("Repaired table", "Bảng đã sửa"),
    "inv": ("Maximal invariant set", "Tập bất biến lớn nhất"),
}


def L(en, vi):
    return vi if LANG == "vi" else en


def LEG(k):
    e, v = _LEG.get(k, (k, k))
    return v if LANG == "vi" else e


def save(fig, name):
    d = os.path.join(FIG, LANG)
    os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, name), bbox_inches=None, pad_inches=0.0)
    plt.close(fig)
    print("  luu %s/%s" % (LANG, name))


def draw_threats(ax, threats, lab=True):
    for i, (cx, cy, r) in enumerate(threats):
        ax.add_patch(Circle((cx, cy), r, fc='#D55E00', ec='#8c3d00', alpha=0.30, lw=0.9,
                            label=L('Threat zone', 'Vùng nguy hiểm') if (i == 0 and lab) else None))
        ax.add_patch(Circle((cx, cy), r + CFG["safeR"], fc='none', ec='#8c3d00', ls=':', lw=0.8,
                            label=L('Safety margin', 'Biên an toàn') if (i == 0 and lab) else None))


def fig_hull(grid, name="Fig2_hull.png", k=15, a=1, i0=26, j0=33):
    """Minh hoa bao loi chinh xac cua anh mot o, hop quet va tap o ke tiep."""
    fig, ax = plt.subplots(figsize=(8.6 * CM, 8.5 * CM))
    c = grid.cell
    dxl, dxh, dyl, dyh = grid.offsets(k, a)
    x0, y0 = i0 * c, j0 * c
    di = cell_range(dxl, c + dxh, c)
    dj = cell_range(dyl, c + dyh, c)
    first = True
    for u in di:
        for w in dj:
            ax.add_patch(Rectangle(((i0 + u) * c, (j0 + w) * c), c, c, fc='#FDF0E3',
                                   ec='none', zorder=0,
                                   label=L('Successor cells', 'Các ô kế tiếp') if first else None))
            first = False
    ax.add_patch(Rectangle((x0, y0), c, c, fc='#EAF2F8', ec=OK[0], lw=1.3, zorder=2,
                           label=L('Source cell', 'Ô nguồn')))
    ax.add_patch(Rectangle((x0 + dxl, y0 + dyl), c + dxh - dxl, c + dyh - dyl,
                           fc='none', ec=OK[2], lw=1.3, ls='--', zorder=3,
                           label=L('Exact image hull', 'Bao lồi chính xác của ảnh')))
    sx0 = x0 + min(0.0, dxl); sy0 = y0 + min(0.0, dyl)
    sx1 = x0 + c + max(0.0, dxh); sy1 = y0 + c + max(0.0, dyh)
    ax.add_patch(Rectangle((sx0, sy0), sx1 - sx0, sy1 - sy0, fc='none', ec=OK[1], lw=1.1,
                           ls=':', zorder=3, label=L('Swept box', 'Hộp quét')))
    rng = np.random.default_rng(3)
    P = []
    for _ in range(200):
        st = (x0 + rng.random() * c, y0 + rng.random() * c,
              -math.pi + (k + rng.random()) * grid.dth)
        P.append(step(st, a)[:2])
    P = np.array(P)
    ax.plot(P[:, 0], P[:, 1], '.', ms=2.0, color='#333333', zorder=4,
            label=L('Sampled successors', 'Trạng thái kế tiếp lấy mẫu'))
    gx0 = math.floor(min(sx0, x0) / c) * c - 0.35 * c
    gx1 = math.ceil(max(sx1, x0 + c) / c) * c + 0.35 * c
    gy0 = math.floor(min(sy0, y0) / c) * c - 0.35 * c
    gy1 = math.ceil(max(sy1, y0 + c) / c) * c + 0.35 * c
    for xx in np.arange(gx0, gx1 + 0.5 * c, c):
        ax.axvline(xx, color='0.80', lw=0.5, zorder=1)
    for yy in np.arange(gy0, gy1 + 0.5 * c, c):
        ax.axhline(yy, color='0.80', lw=0.5, zorder=1)
    ax.set_xlim(gx0, gx1); ax.set_ylim(gy0, gy1)
    ax.set_aspect('equal')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=3, fontsize=6.6, framealpha=1.0,
               borderaxespad=0.2, handlelength=1.8, columnspacing=0.8, labelspacing=0.4,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.3, rect=(0.0, 0.145, 1.0, 1.0))
    save(fig, name)


def fig_region(grid, threats, S, ks, name="Fig3_region.png"):
    """Lat cat cua tap duoc chung nhan theo ba o goc huong."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    cmap = ListedColormap(['#4D4D4D', '#E8F0F7'])
    fig, axs = plt.subplots(1, len(ks), figsize=(14.8 * CM, 6.4 * CM))
    c = grid.cell
    for ax, k in zip(axs, ks):
        ax.imshow(S[:, :, k].T.astype(float), origin='lower', cmap=cmap, vmin=0, vmax=1,
                  extent=(0, grid.nx * c, 0, grid.ny * c), interpolation='nearest')
        for cx, cy, r in threats:
            ax.add_patch(Circle((cx, cy), r, fc='#D55E00', ec='#8c3d00', alpha=0.55, lw=0.8))
            ax.add_patch(Circle((cx, cy), r + CFG["safeR"], fc='none', ec='#8c3d00',
                                ls=':', lw=0.8))
        th = math.degrees(-math.pi + (k + 0.5) * grid.dth)
        ax.set_title(L('heading %.0f°' % th, 'hướng %.0f°' % th))
        ax.set_xlabel('x (m)')
        ax.set_aspect('equal')
    axs[0].set_ylabel('y (m)')
    hs = [Patch(fc='#E8F0F7', ec='0.4', label=L('Certified', 'Được chứng nhận')),
          Patch(fc='#4D4D4D', ec='0.4', label=L('Not certified', 'Không được chứng nhận')),
          Patch(fc='#D55E00', ec='#8c3d00', alpha=0.55, label=L('Threat zone', 'Vùng nguy hiểm'))]
    fig.legend(handles=hs, loc='lower center', ncol=3, fontsize=7.6, framealpha=1.0,
               borderaxespad=0.2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.4, rect=(0.0, 0.12, 1.0, 1.0))
    save(fig, name)


def fig_fixpoint(hists, name="Fig4_fixpoint.png"):
    fig, ax = plt.subplots(figsize=(8.6 * CM, 6.0 * CM))
    ls = ['-', '--', '-.', ':']
    for i, (tag, h, tot) in enumerate(hists):
        ax.plot(np.arange(1, len(h) + 1), 100.0 * np.array(h) / tot, ls[i % 4],
                color=OK[i % 7], marker='o', ms=2.4, label=LEG(tag))
    ax.set_xlabel(L('Fixed-point iteration', 'Vòng lặp điểm bất động'))
    ax.set_ylabel(L('Certified cells (%)', 'Ô được chứng nhận (%)'))
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(framealpha=0.95)
    fig.tight_layout(pad=0.4)
    save(fig, name)


def fig_resolution(rows, name="Fig8_resolution.png"):
    fig, axs = plt.subplots(1, 2, figsize=(14.8 * CM, 5.8 * CM))
    cells = sorted(set(r["cell"] for r in rows))
    nths = sorted(set(r["nth"] for r in rows))
    mk = ['o', 's', '^', 'd']
    for i, nt in enumerate(nths):
        sub = [r for r in rows if r["nth"] == nt]
        sub.sort(key=lambda r: r["cell"])
        axs[0].plot([r["cell"] for r in sub], [100.0 * r["frac"] for r in sub],
                    '-', marker=mk[i % 4], ms=4, color=OK[i % 7],
                    label=L('%d heading bins' % nt, '%d ô góc hướng' % nt))
        axs[1].plot([r["ncell"] for r in sub], [r["time"] for r in sub],
                    ' ', marker=mk[i % 4], ms=4.5, color=OK[i % 7],
                    label=L('%d heading bins' % nt, '%d ô góc hướng' % nt))
    axs[0].set_xlabel(L('Cell size (m)', 'Kích thước ô (m)'))
    axs[0].set_ylabel(L('Certified fraction (%)', 'Tỷ lệ được chứng nhận (%)'))
    axs[1].set_xlabel(L('Number of cells', 'Số ô'))
    axs[1].set_ylabel(L('Verification time (s)', 'Thời gian kiểm chứng (s)'))
    nn = np.array([r["ncell"] for r in rows], dtype=float)
    tt = np.array([r["time"] for r in rows], dtype=float)
    kfit = float(np.sum(tt) / np.sum(nn))
    xs = np.array([nn.min(), nn.max()])
    axs[1].plot(xs, kfit * xs, '-', color='0.35', lw=1.0,
                label=L('Linear reference', 'Tham chiếu tuyến tính'))
    axs[1].set_xscale('log'); axs[1].set_yscale('log')
    for ax in axs:
        ax.grid(True, alpha=0.3, ls=':')
    h, lb = axs[1].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=4, fontsize=7.4, framealpha=1.0,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    fig.set_size_inches(14.8 * CM, 6.8 * CM)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.12, 1.0, 1.0))
    save(fig, name)


def fig_tracks(threats, tracks, name="Fig6_tracks.png"):
    fig, ax = plt.subplots(figsize=(8.6 * CM, 11.2 * CM))
    draw_threats(ax, threats)
    ls = ['-', '--', '-.', ':', '-']
    for i, (lab, P) in enumerate(tracks.items()):
        thin = (lab == "repair")
        ax.plot(P[:, 0], P[:, 1], ls[i % 5], color=OK[i % 7], lw=0.8 if thin else 1.4,
                alpha=0.75 if thin else 1.0, label=LEG(lab), zorder=2 if thin else 3)
    ax.plot(CFG["start"][0], CFG["start"][1], 's', color='#009E73', ms=7, mec='k', mew=0.6,
            label=L('Start', 'Điểm xuất phát'))
    ax.plot(CFG["goal"][0], CFG["goal"][1], '*', color='#CC0000', ms=12, mec='k', mew=0.5,
            label=L('Goal', 'Điểm đích'))
    ax.set_xlim(0, CFG["W"]); ax.set_ylim(0, CFG["H"]); ax.set_aspect('equal')
    ax.grid(True, alpha=0.3, ls=':')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=2, fontsize=7.0, framealpha=1.0,
               borderaxespad=0.2, handlelength=2.2, columnspacing=0.9, labelspacing=0.5,
               bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(pad=0.35, rect=(0.0, 0.25, 1.0, 1.0))
    save(fig, name)


def fig_sampling(curve, name="Fig7_sampling.png"):
    fig, ax = plt.subplots(figsize=(8.6 * CM, 6.0 * CM))
    n = [c[0] for c in curve]
    f = [c[1] for c in curve]
    ax.plot(n, f, '-o', ms=3.2, color=OK[1],
            label=L('Unsafe starts found by sampling',
                    'Số điểm không an toàn tìm được nhờ lấy mẫu'))
    ax.set_xscale('log')
    ax.set_xlabel(L('Random initial states drawn', 'Số trạng thái đầu lấy ngẫu nhiên'))
    ax.set_ylabel(L('Unsafe starts found', 'Số điểm không an toàn'))
    ax.grid(True, alpha=0.3, ls=':')
    ax.legend(framealpha=0.95, fontsize=7.4, loc='upper left')
    fig.tight_layout(pad=0.4)
    save(fig, name)


def fig_learning(curves, name="Fig5_learning.png", win=200):
    fig, axs = plt.subplots(1, 2, figsize=(14.8 * CM, 5.8 * CM))
    ls = ['-', '--', '-.']
    for i, (lab, d) in enumerate(curves.items()):
        for ax, key in zip(axs, ("succ", "coll")):
            y = np.asarray(d[key])
            m = len(y) // win * win
            ym = y[:m].reshape(-1, win).mean(1)
            ax.plot((np.arange(len(ym)) + 1) * win, 100 * ym, ls[i % 3], color=OK[i % 7],
                    label=LEG(lab))
    axs[0].set_ylabel(L('Success rate (%)', 'Tỷ lệ thành công (%)'))
    axs[1].set_ylabel(L('Collision rate (%)', 'Tỷ lệ va chạm (%)'))
    for ax in axs:
        ax.set_xlabel(L('Training episode', 'Tập huấn luyện'))
        ax.grid(True, alpha=0.3, ls=':')
    axs[0].set_ylim(-3, 103); axs[1].set_ylim(-3, 103)
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=2, fontsize=7.6, framealpha=1.0,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.2)
    fig.set_size_inches(14.8 * CM, 7.0 * CM)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.17, 1.0, 1.0))
    save(fig, name)


def render_all(PAY, lang):
    global LANG
    LANG = lang
    print("  --- ve hinh cho ngon ngu %s ---" % lang)
    g = Grid(PAY["cell"], PAY["nth"])
    fig_hull(g)
    fig_region(g, PAY["threats"], PAY["S"], PAY["ks"])
    fig_fixpoint(PAY["hists"])
    fig_resolution(PAY["res"])
    fig_tracks(PAY["threats"], PAY["tracks"])
    fig_sampling(PAY["curve"])
    fig_learning(PAY["curves"])
    if "ck" in PAY:
        fig_ckpt(PAY["ck"], PAY["colls"])
        fig_policy(g, PAY["threats"], PAY["PI_mask"], PAY["Q_mask"], PAY["S"], 8)
        fig_3d(PAY["R3"])
    LANG = "en"


def replot(lang=None):
    with open(os.path.join(FIG, "plotdata.pkl"), "rb") as f:
        PAY = pickle.load(f)
    for lg in ([lang] if lang else ("en", "vi")):
        render_all(PAY, lg)


# ============ 8. CAC THI NGHIEM BO SUNG CHO VONG PHAN BIEN (2D) ============
def roundoff_example():
    """Chuoi hanh dong ngan nhat tu diem xuat phat danh nghia cho toa do roi dung len duong luoi
    trong so hoc chinh xac, nhung so dau phay dong lai cho gia tri nho hon mot chut."""
    frontier = [(CFG["start"], ())]
    for depth in range(1, 13):
        nf, seen = [], set()
        for s, acts in frontier:
            for a in range(NA):
                sn = step(s, a)
                for idx in (0, 1):
                    v = sn[idx]
                    r = round(v)
                    if v < r and r - v < 1e-9:
                        return dict(steps=depth, actions=list(acts + (a,)), coord="xy"[idx],
                                    value=repr(v), exact=float(r), floor=int(math.floor(v)),
                                    heading_start=repr(CFG["start"][2]),
                                    heading_boundary=repr(-math.pi + 8 * 2.0 * math.pi / 24))
                key = (round(sn[0], 6), round(sn[1], 6), round(sn[2], 6))
                if key not in seen:
                    seen.add(key)
                    nf.append((sn, acts + (a,)))
        frontier = nf
    return None


def lemma_example(grid, k=15, a=2):
    """So lieu cu the cua Bo de 1 va Bo de 2 cho mot cap (o goc, hanh dong)."""
    lo = -math.pi + k * grid.dth + ACT[a] * CFG["dt"] - ETA
    hi = lo + grid.dth + 2.0 * ETA
    cl, ch = cos_range(lo, hi)
    sl, sh = sin_range(lo, hi)
    c = grid.cell
    di = cell_range(LSTEP * cl, c + LSTEP * ch, c)
    dj = cell_range(LSTEP * sl, c + LSTEP * sh, c)
    return dict(k=k, a=int(a - 1), kn=int((k + grid.kshift[a]) % grid.nth),
                th_lo=math.degrees(lo + ETA), th_hi=math.degrees(hi - ETA),
                cm=cl, cp=ch, sm=sl, sp=sh, di=[di[0], di[-1]], dj=[dj[0], dj[-1]],
                nsucc=len(di) * len(dj))


def lemma_check(grid, vf, threats, n=400000, seed=5):
    """Kiem tra Bo de 1, Bo de 2 va Dinh ly 1 bang lay mau: trang thai ngau nhien, trang thai nam
    dung tren mat o, va cac trang thai sinh tu diem xuat phat danh nghia."""
    rng = np.random.default_rng(seed)
    c, g = grid.cell, grid
    S_lem1 = S_lem2 = S_thm = 0
    hit = {key: set() for key in vf.off}
    tot = 0
    states = []
    for _ in range(n):
        x = rng.uniform(0, CFG["W"]); y = rng.uniform(0, CFG["H"])
        th = rng.uniform(-math.pi, math.pi)
        u = rng.random()
        if u < 0.25:
            x = c * round(x / c)
        elif u < 0.5:
            y = c * round(y / c)
        elif u < 0.6:
            th = -math.pi + g.dth * round((th + math.pi) / g.dth)
        states.append((x, y, th))
    s = CFG["start"]
    for _ in range(4000):                  # quy dao that tu diem xuat phat, du hanh dong ngau nhien
        states.append(s)
        s = step(s, int(rng.integers(NA)))
        if not (0.0 <= s[0] <= CFG["W"] and 0.0 <= s[1] <= CFG["H"]):
            s = CFG["start"]
    for st in states:
        i, j, k = g.phi(st)
        if not (0 <= i < g.nx and 0 <= j < g.ny):
            continue
        for a in range(NA):
            tot += 1
            sn = step(st, a)
            di, dj, kn = vf.off[(k, a)]
            i2 = int(math.floor(sn[0] / c)); j2 = int(math.floor(sn[1] / c))
            k2 = int(math.floor((wrap(sn[2]) + math.pi) / g.dth + TOL)) % g.nth
            if k2 != kn:
                S_lem1 += 1
            if (i2 - i) not in di or (j2 - j) not in dj:
                S_lem2 += 1
            else:
                hit[(k, a)].add((i2 - i, j2 - j))
            if not vf.bad[k, a, i, j] and not clear_exact(st[:2], sn[:2], threats, CFG["safeR"]):
                # chu y: clear_exact con kiem tra ca hinh vuong nhiem vu
                if (0.0 <= sn[0] <= CFG["W"] and 0.0 <= sn[1] <= CFG["H"]):
                    S_thm += 1
    used = sum(len(v) for v in hit.values())
    allc = sum(len(vf.off[key][0]) * len(vf.off[key][1]) for key in vf.off)
    return dict(pairs=tot, viol_lemma1=S_lem1, viol_lemma2=S_lem2, viol_thm1=S_thm,
                succ_listed=allc, succ_observed=used, tight=used / allc)


class SampledVerifier(Verifier):
    """Truu tuong hoa KHONG dung dan dung de doi chung: anh cua o chi lay tu mot so diem mau.

    pts: danh sach diem mau tuong doi trong o (u, v, w) thuoc [0, 1]^3 theo x, y, goc huong.
    Anh cua o la hop cac o chua anh cua diem mau, va cap (o, hanh dong) bi coi la nguy hiem khi
    mot day cung xuat phat tu diem mau cham hinh tron noi rong.
    """

    def __init__(self, grid, threats, pts):
        self.g = grid
        self.threats = threats
        g = grid
        c = g.cell
        I = (np.arange(g.nx)[:, None]) * c + np.zeros((1, g.ny))
        J = (np.arange(g.ny)[None, :]) * c + np.zeros((g.nx, 1))
        self.bad = np.zeros((g.nth, NA, g.nx, g.ny), dtype=bool)
        self.off = {}
        for k in range(g.nth):
            for a in range(NA):
                offs = set()
                bad = np.zeros((g.nx, g.ny), dtype=bool)
                for (u, v, w) in pts:
                    th = -math.pi + (k + w) * g.dth
                    t2 = th + ACT[a] * CFG["dt"]
                    dx, dy = LSTEP * math.cos(t2), LSTEP * math.sin(t2)
                    offs.add((int(math.floor(u + dx / c)), int(math.floor(v + dy / c))))
                    x0 = I + u * c; y0 = J + v * c
                    x1 = x0 + dx; y1 = y0 + dy
                    out = (x1 < 0) | (x1 > CFG["W"]) | (y1 < 0) | (y1 > CFG["H"])
                    for cx, cy, r in threats:
                        rho = r + CFG["safeR"]
                        ax = x0 - cx; ay = y0 - cy
                        dd = dx * dx + dy * dy
                        t = np.clip(-(ax * dx + ay * dy) / dd, 0.0, 1.0)
                        qx = ax + t * dx; qy = ay + t * dy
                        out |= (qx * qx + qy * qy) <= rho * rho
                    bad |= out
                self.bad[k, a] = bad
                self.off[(k, a)] = sorted(offs)
        self.pad = max(4, 2 + int(math.ceil(LSTEP / c)))
        self.edges = sum(len(v) for v in self.off.values()) * g.nx * g.ny

    def _all_in(self, S, k, a):
        g, p = self.g, self.pad
        kn = (k + g.kshift[a]) % g.nth
        P = np.zeros((g.nx + 2 * p, g.ny + 2 * p), dtype=bool)
        P[p:p + g.nx, p:p + g.ny] = S[:, :, kn]
        out = np.ones((g.nx, g.ny), dtype=bool)
        for (u, w) in self.off[(k, a)]:
            out &= P[p + u:p + u + g.nx, p + w:p + w + g.ny]
        return out


def open_cell_verifier(grid, threats):
    """Truu tuong hoa dung o nua mo, khong co le TOL, ETA, EPS (doi chung cho quy uoc o dong)."""
    global TOL, ETA, EPS
    old = (TOL, ETA, EPS)
    TOL = ETA = EPS = 0.0
    try:
        vf = Verifier(grid, threats)
        c = grid.cell
        for key, (di, dj, kn) in list(vf.off.items()):
            k, a = key
            dxl, dxh, dyl, dyh = grid.offsets(k, a)
            # o nua mo [n c, (n+1) c): chi lay o co phan trong giao voi doan anh
            ii = list(range(int(math.floor(dxl / c)), int(math.ceil((c + dxh) / c))))
            jj = list(range(int(math.floor(dyl / c)), int(math.ceil((c + dyh) / c))))
            vf.off[key] = (ii, jj, kn)
        vf.edges = sum(len(v[0]) * len(v[1]) for v in vf.off.values()) * grid.nx * grid.ny
    finally:
        TOL, ETA, EPS = old
    return vf


def lattice_consistency(grid, vf, tol, trials=3000, steps=60, seed=3):
    """Ty le chuyen trang thai ma o ke tiep do bo luong tu hoa gan KHONG nam trong danh sach o ke
    tiep cua phep truu tuong hoa. Trang thai dau dat tren mat o (vi tri nguyen, goc huong dung
    bien o goc), dung nhu trang thai xuat phat danh nghia, roi bay theo hanh dong ngau nhien."""
    rng = np.random.default_rng(seed)
    c = grid.cell

    def ph(s):
        return (int(math.floor(s[0] / c)), int(math.floor(s[1] / c)),
                int(math.floor((wrap(s[2]) + math.pi) / grid.dth + tol)) % grid.nth)

    def listed(k, a, du, dv, k2):
        o = vf.off[(k, a)]
        if isinstance(o, tuple):
            di, dj, kn = o
            return du in di and dv in dj and k2 == kn
        return (du, dv) in o and k2 == (k + grid.kshift[a]) % grid.nth

    bad = tot = 0
    for trial in range(trials):
        s = (float(rng.integers(5, 55)), float(rng.integers(5, 55)),
             -math.pi + grid.dth * int(rng.integers(0, grid.nth)))
        if trial == 0:
            s = CFG["start"]
        for _ in range(steps):
            a = int(rng.integers(NA))
            sn = step(s, a)
            if not (0.0 < sn[0] < CFG["W"] and 0.0 < sn[1] < CFG["H"]):
                break
            i, j, k = ph(s); i2, j2, k2 = ph(sn)
            tot += 1
            if not listed(k, a, i2 - i, j2 - j, k2):
                bad += 1
            s = sn
    return bad, tot


def unsound_study(grid, threats, Qmask, S_ref, vf_ref, n=20000):
    """So sanh truu tuong hoa dung dan voi ba truu tuong hoa thuong gap nhung khong dung dan."""
    global TOL
    rows = []
    cases = [
        ("center", [(0.5, 0.5, 0.5)]),
        ("sample27", [(u, v, w) for u in (0.0, 0.5, 1.0) for v in (0.0, 0.5, 1.0)
                      for w in (0.0, 0.5, 1.0)]),
    ]
    for tag, pts in cases:
        t0 = time.time()
        vv = SampledVerifier(grid, threats, pts)
        Sx, hx = vv.max_invariant()
        Mx = vv.mask(Sx)
        PIx = greedy(Qmask, mask=Mx)
        bad, tot = monte_carlo(PIx, grid, threats, Sx, n=n, inside=True)
        lb, lt = lattice_consistency(grid, vv, TOL)
        rows.append(dict(tag=tag, frac=float(Sx.mean()), edges=int(vv.edges), it=len(hx),
                         time=time.time() - t0, viol=int(bad), draws=int(tot),
                         extra=int((Sx & ~S_ref).sum()), lat_bad=lb, lat_tot=lt))
        print("   %-9s bat bien %.2f%%, %d canh, vi pham %d/%d" % (tag, 100 * Sx.mean(),
                                                                   vv.edges, bad, tot))
    t0 = time.time()
    vo = open_cell_verifier(grid, threats)
    So, ho = vo.max_invariant()
    Mo = vo.mask(So)
    PIo = greedy(Qmask, mask=Mo)
    keep, TOL = TOL, 0.0                 # bo luong tu hoa cua phuong an nay cung khong co le
    try:
        bad, tot = monte_carlo(PIo, grid, threats, So, n=n, inside=True)
    finally:
        TOL = keep
    lb, lt = lattice_consistency(grid, vo, 0.0)
    rows.append(dict(tag="open", frac=float(So.mean()), edges=int(vo.edges), it=len(ho),
                     time=time.time() - t0, viol=int(bad), draws=int(tot),
                     extra=int((So & ~S_ref).sum()), lat_bad=lb, lat_tot=lt))
    lb, lt = lattice_consistency(grid, vf_ref, TOL)
    bad, tot = monte_carlo(greedy(Qmask, mask=vf_ref.mask(S_ref)), grid, threats, S_ref, n=n,
                           inside=True)
    rows.append(dict(tag="sound", frac=float(S_ref.mean()), edges=int(vf_ref.edges),
                     it=None, time=None, viol=int(bad), draws=int(tot), extra=0,
                     lat_bad=lb, lat_tot=lt))
    for r in rows[-2:]:
        print("   %-9s bat bien %.2f%%, %d canh, vi pham %d/%d, luoi %d/%d"
              % (r["tag"], 100 * r["frac"], r["edges"], r["viol"], r["draws"], r["lat_bad"],
                 r["lat_tot"]))
    return rows


def checkpoint_cert(vf, snaps, mask=None):
    """Ty le duoc chung nhan cua chinh sach tham lam o tung moc huan luyen."""
    out = []
    for ep, Q in snaps:
        PI = greedy(Q, mask=mask)
        Sp, _ = vf.verify_policy(PI)
        out.append((ep, float(Sp.mean())))
    return out


def scale_check(lam=20.0):
    """Mo hinh khong thu nguyen: nhan moi do dai va van toc voi lam thi tap duoc chung nhan
    giu nguyen (cung chu ky lay mau va toc do quay)."""
    global LSTEP
    keep = (dict(CFG), LSTEP)
    try:
        CFG["v"] *= lam; CFG["W"] *= lam; CFG["H"] *= lam; CFG["safeR"] *= lam
        LSTEP = CFG["v"] * CFG["dt"]
        g = Grid(1.0 * lam, 24)
        th = [(x * lam, y * lam, r * lam) for (x, y, r) in SCEN["S1"]]
        vf = Verifier(g, th)
        S, h = vf.max_invariant()
        out = dict(lam=lam, v=CFG["v"], cell=g.cell, frac=float(S.mean()), cells=int(S.sum()),
                   it=len(h), bank_deg=math.degrees(math.atan(CFG["v"] * CFG["wmax"] / 9.81)),
                   radius=CFG["v"] / CFG["wmax"])
    finally:
        CFG.clear(); CFG.update(keep[0]); LSTEP = keep[1]
    return out


def fig_policy(grid, threats, PI, Q, S, k, name="Fig_policy.png"):
    """Chinh sach tham lam va gia tri cua bang Q de xuat tren mot lat goc huong."""
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    fig, axs = plt.subplots(1, 2, figsize=(14.8 * CM, 7.8 * CM))
    c = grid.cell
    ext = (0, grid.nx * c, 0, grid.ny * c)
    A = PI[:, :, k].astype(float)
    A[~S[:, :, k]] = 3
    cmap = ListedColormap(['#0072B2', '#F0E442', '#009E73', '#4D4D4D'])
    axs[0].imshow(A.T, origin='lower', cmap=cmap, vmin=-0.5, vmax=3.5, extent=ext,
                  interpolation='nearest')
    V = Q[:, :, k, :].max(axis=2).astype(float)
    V = np.where(S[:, :, k], V, np.nan)
    im = axs[1].imshow(V.T, origin='lower', cmap='viridis', extent=ext, interpolation='nearest')
    cb = fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.03)
    cb.set_label(L('Greedy value', 'Giá trị tham lam'), fontsize=9)
    th = math.degrees(-math.pi + (k + 0.5) * grid.dth)
    for ax, ttl in zip(axs, (L('(a) Greedy action', '(a) Hành động tham lam'),
                             L('(b) Greedy value', '(b) Giá trị tham lam'))):
        for cx, cy, r in threats:
            ax.add_patch(Circle((cx, cy), r, fc='#D55E00', ec='#8c3d00', alpha=0.55, lw=0.8))
        ax.plot(CFG["goal"][0], CFG["goal"][1], '*', color='#CC0000', ms=9, mec='k', mew=0.5)
        ax.set_title(ttl + L(', heading %.0f°' % th, ', hướng %.0f°' % th))
        ax.set_xlabel('x (m)')
        ax.set_aspect('equal')
    axs[0].set_ylabel('y (m)')
    hs = [Patch(fc='#0072B2', label=L('Turn right', 'Quay phải')),
          Patch(fc='#F0E442', label=L('Straight', 'Bay thẳng')),
          Patch(fc='#009E73', label=L('Turn left', 'Quay trái')),
          Patch(fc='#4D4D4D', label=L('Not certified', 'Không được chứng nhận')),
          Patch(fc='#D55E00', alpha=0.55, label=L('Threat zone', 'Vùng nguy hiểm'))]
    fig.legend(handles=hs, loc='lower center', ncol=5, fontsize=7.4, framealpha=1.0,
               bbox_to_anchor=(0.5, 0.0), columnspacing=0.9, handlelength=1.4)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.10, 1.0, 0.97))
    save(fig, name)


def fig_ckpt(ck, colls, name="Fig_ckpt.png"):
    """(a) Ty le chung nhan cua bang Q theo moc huan luyen; (b) so va cham tich luy."""
    fig, axs = plt.subplots(1, 2, figsize=(14.8 * CM, 6.6 * CM))
    ls = ['-', '--', '-.', ':']
    mk = ['o', 's', '^', 'd']
    for i, tag in enumerate(("raw", "step1", "online", "mask")):
        e = [p[0] for p in ck[tag]]
        f = [100.0 * p[1] for p in ck[tag]]
        axs[0].plot(e, f, ls[i], marker=mk[i], ms=3.2, color=OK[i], label=LEG(tag))
        cc = np.cumsum(colls[tag])
        axs[1].plot(np.arange(1, len(cc) + 1), cc, ls[i], color=OK[i], label=LEG(tag))
    axs[0].set_ylabel(L('Certified fraction (%)', 'Tỷ lệ được chứng nhận (%)'))
    axs[1].set_ylabel(L('Cumulative collisions', 'Số va chạm tích lũy'))
    axs[0].set_title(L('(a) Certified fraction', '(a) Tỷ lệ được chứng nhận'))
    axs[1].set_title(L('(b) Collisions during training', '(b) Va chạm trong huấn luyện'))
    axs[0].set_ylim(-3, 70)
    for ax in axs:
        ax.set_xlabel(L('Training episode', 'Tập huấn luyện'))
        ax.grid(True, alpha=0.3, ls=':')
    h, lb = axs[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=4, fontsize=7.4, framealpha=1.0,
               bbox_to_anchor=(0.5, 0.0), columnspacing=1.0)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.11, 1.0, 1.0))
    save(fig, name)


# ============ 9. MO RONG BA CHIEU ============
# Trang thai (x, y, z, theta). Lenh gom toc do quay va toc do len xuong:
# a = (a_theta, a_z), a_theta, a_z thuoc {-1, 0, 1}. Do cao bi chia thanh lop day h_z = v_z dt nen
# chi so lop do cao doi dung a_z sau moi buoc, giong nhu chi so goc huong (Bo de 3).
CFG3 = dict(z0=1.0, z1=11.0, hz=0.5, vz=0.5, safeV=1.0, z_start=5.0, kz=0.04, gamma=1.0)
ACT3 = [(at, az) for at in range(NA) for az in (-1, 0, 1)]
NA3 = len(ACT3)


def env3(name):
    """Ba moi truong co do phuc tap tang dan. Tru tron: (cx, cy, r, h); khoi nha: (x0, x1, y0, y1, h).
    Do cao h tinh tu mat dat, tran vung bay la z1 = 11 m, nen vat can cao hon 11 m phai di vong."""
    if name == "E1":
        cyl = [(20.0, 40.0, 9.0, 6.0), (38.0, 18.0, 10.0, 14.0), (25.0, 10.0, 4.0, 3.0)]
        return dict(cyl=cyl, blk=[])
    if name == "E2":
        cyl = [(18.0, 42.0, 7.0, 8.0), (34.0, 30.0, 6.0, 14.0), (46.0, 15.0, 6.0, 5.0),
               (12.0, 22.0, 5.0, 6.5), (45.0, 45.0, 5.0, 4.0), (30.0, 50.0, 3.5, 14.0),
               (54.0, 30.0, 3.5, 9.0), (28.0, 12.0, 4.5, 7.0)]
        return dict(cyl=cyl, blk=[])
    if name == "E3":
        rng = np.random.default_rng(SEED)
        blk = []
        for ix in range(5):
            for iy in range(5):
                x0 = 6.0 + 11.0 * ix + rng.uniform(0.0, 1.5)
                y0 = 6.0 + 11.0 * iy + rng.uniform(0.0, 1.5)
                w = rng.uniform(5.0, 7.0); d = rng.uniform(5.0, 7.0)
                if (ix == 0 and iy == 4) or (ix == 4 and iy == 0):
                    continue            # de trong goc xuat phat va goc dich
                h = float(rng.choice([3.0, 5.0, 7.0, 9.0, 13.0]))
                blk.append((round(x0, 2), round(x0 + w, 2), round(y0, 2), round(y0 + d, 2), h))
        cyl = [(30.5, 30.5, 3.0, 20.0)]          # vung cam bay (tru tram ra da)
        return dict(cyl=cyl, blk=blk)
    raise ValueError(name)


def dsm_raster(env, res=0.5):
    """Mo hinh be mat so (DSM) dang luoi diem anh suy ra tu danh sach vat can, dung de ve hinh va
    minh hoa buoc chuyen ban do so thanh dau vao cua bo kiem chung."""
    nx = int(round(CFG["W"] / res)); ny = int(round(CFG["H"] / res))
    xc = (np.arange(nx) + 0.5) * res; yc = (np.arange(ny) + 0.5) * res
    X, Y = np.meshgrid(xc, yc, indexing='ij')
    Hm = np.zeros((nx, ny))
    for cx, cy, r, h in env["cyl"]:
        Hm = np.where((X - cx) ** 2 + (Y - cy) ** 2 <= r * r, np.maximum(Hm, h), Hm)
    for x0, x1, y0, y1, h in env["blk"]:
        Hm = np.where((X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1), np.maximum(Hm, h), Hm)
    return Hm


class Grid3(Grid):
    def __init__(self, cell=1.0, nth=24, hz=None):
        super().__init__(cell, nth)
        self.hz = CFG3["hz"] if hz is None else hz
        if abs(CFG3["vz"] * CFG["dt"] / self.hz - round(CFG3["vz"] * CFG["dt"] / self.hz)) > 1e-12:
            raise ValueError("be day lop do cao phai chia het buoc len xuong")
        self.zs = int(round(CFG3["vz"] * CFG["dt"] / self.hz))
        self.nz = int(round((CFG3["z1"] - CFG3["z0"]) / self.hz))
        self.ncell = self.nx * self.ny * self.nz * self.nth

    def phi3(self, s):
        x, y, z, th = s
        i, j, k = self.phi((x, y, th))
        m = int(math.floor((z - CFG3["z0"]) / self.hz + TOL))
        return i, j, min(self.nz - 1, max(0, m)), k


def step3(s, a3):
    at, az = ACT3[a3]
    x, y, z, th = s
    t2 = wrap(th + ACT[at] * CFG["dt"])
    return (x + LSTEP * math.cos(t2), y + LSTEP * math.sin(t2), z + az * CFG3["vz"] * CFG["dt"], t2)


class Verifier3:
    """Do thi o - o ba chieu va hai diem bat dong, cung cau truc voi Verifier."""

    def __init__(self, grid, env):
        self.g = g = grid
        self.env = env
        c = g.cell
        I = (np.arange(g.nx)[:, None]) * c + np.zeros((1, g.ny))
        J = (np.arange(g.ny)[None, :]) * c + np.zeros((g.nx, 1))
        zl = CFG3["z0"] + np.arange(g.nz) * g.hz
        R, RV = CFG["safeR"], CFG3["safeV"]
        self.bad = np.zeros((g.nth, NA3, g.nx, g.ny, g.nz), dtype=bool)
        self.off = {}
        for k in range(g.nth):
            for at in range(NA):
                dxl, dxh, dyl, dyh = g.offsets(k, at)
                X0 = I + min(0.0, dxl) - EPS; X1 = I + c + max(0.0, dxh) + EPS
                Y0 = J + min(0.0, dyl) - EPS; Y1 = J + c + max(0.0, dyh) + EPS
                hits = []
                for cx, cy, r, h in env["cyl"]:
                    dx = np.maximum(np.maximum(X0 - cx, cx - X1), 0.0)
                    dy = np.maximum(np.maximum(Y0 - cy, cy - Y1), 0.0)
                    hits.append(((dx * dx + dy * dy) <= (r + R) ** 2, h))
                for bx0, bx1, by0, by1, h in env["blk"]:
                    dx = np.maximum(np.maximum(bx0 - X1, X0 - bx1), 0.0)
                    dy = np.maximum(np.maximum(by0 - Y1, Y0 - by1), 0.0)
                    hits.append(((dx * dx + dy * dy) <= R * R, h))
                di = cell_range(dxl, c + dxh, c)
                dj = cell_range(dyl, c + dyh, c)
                for az in (-1, 0, 1):
                    a3 = ACT3.index((at, az))
                    Z0 = zl + min(0.0, az * g.hz) - EPS          # day hop quet theo do cao
                    bad = np.zeros((g.nx, g.ny, g.nz), dtype=bool)
                    for hx, h in hits:
                        zc = Z0 <= h + RV
                        if zc.any() and hx.any():
                            bad |= hx[:, :, None] & zc[None, None, :]
                    self.bad[k, a3] = bad
                    self.off[(k, a3)] = (di, dj, az * g.zs, (k + g.kshift[at]) % g.nth)
        self.pad = max(4, 2 + int(math.ceil(LSTEP / c)))
        self.edges = sum(len(v[0]) * len(v[1]) for v in self.off.values()) * g.nx * g.ny * g.nz

    def _all_in(self, S, k, a3):
        di, dj, dz, kn = self.off[(k, a3)]
        p, g = self.pad, self.g
        P = np.zeros((g.nx + 2 * p, g.ny + 2 * p, g.nz), dtype=bool)
        if dz == 0:
            P[p:p + g.nx, p:p + g.ny, :] = S[:, :, :, kn]
        elif dz > 0:
            P[p:p + g.nx, p:p + g.ny, :g.nz - dz] = S[:, :, dz:, kn]
        else:
            P[p:p + g.nx, p:p + g.ny, -dz:] = S[:, :, :g.nz + dz, kn]
        out = np.ones((g.nx, g.ny, g.nz), dtype=bool)
        for u in di:
            for w in dj:
                out &= P[p + u:p + u + g.nx, p + w:p + w + g.ny, :]
        return out

    def max_invariant(self, maxit=600):
        g = self.g
        S = np.ones((g.nx, g.ny, g.nz, g.nth), dtype=bool)
        hist = []
        for _ in range(maxit):
            new = np.zeros_like(S)
            for k in range(g.nth):
                acc = np.zeros((g.nx, g.ny, g.nz), dtype=bool)
                for a3 in range(NA3):
                    acc |= (~self.bad[k, a3]) & self._all_in(S, k, a3)
                new[:, :, :, k] = acc
            new &= S
            hist.append(int(new.sum()))
            if np.array_equal(new, S):
                break
            S = new
        return S, hist

    def verify_policy(self, PI, maxit=600):
        g = self.g
        S = np.ones((g.nx, g.ny, g.nz, g.nth), dtype=bool)
        hist = []
        for _ in range(maxit):
            new = np.zeros_like(S)
            for k in range(g.nth):
                for a3 in range(NA3):
                    sel = (PI[:, :, :, k] == a3)
                    if not sel.any():
                        continue
                    new[:, :, :, k] |= sel & (~self.bad[k, a3]) & self._all_in(S, k, a3)
            new &= S
            hist.append(int(new.sum()))
            if np.array_equal(new, S):
                break
            S = new
        return S, hist

    def _any_in(self, S, k, a3):
        di, dj, dz, kn = self.off[(k, a3)]
        p, g = self.pad, self.g
        P = np.zeros((g.nx + 2 * p, g.ny + 2 * p, g.nz), dtype=bool)
        if dz == 0:
            P[p:p + g.nx, p:p + g.ny, :] = S[:, :, :, kn]
        elif dz > 0:
            P[p:p + g.nx, p:p + g.ny, :g.nz - dz] = S[:, :, dz:, kn]
        else:
            P[p:p + g.nx, p:p + g.ny, -dz:] = S[:, :, :g.nz + dz, kn]
        out = np.zeros((g.nx, g.ny, g.nz), dtype=bool)
        for u in di:
            for w in dj:
                out |= P[p + u:p + u + g.nx, p + w:p + w + g.ny, :]
        return out

    def steps_to_goal(self, S, M, goal, maxit=2000):
        """So buoc it nhat tu moi o toi dich tren do thi o, chi dung hanh dong cua mat na M va chi di
        qua cac o cua S (ton tai mot o ke tiep gan dich hon). Day la uoc luong lac quan, dung lam the
        nang dinh hinh phan thuong; -1 voi o khong noi duoc toi dich."""
        g = self.g
        R = goal & S
        rank = np.where(R, 0, -1).astype(np.int32)
        for n in range(1, maxit):
            new = np.zeros_like(R)
            for k in range(g.nth):
                acc = np.zeros((g.nx, g.ny, g.nz), dtype=bool)
                for a3 in range(NA3):
                    acc |= M[:, :, :, k, a3] & self._any_in(R, k, a3)
                new[:, :, :, k] = acc
            new = (new & S) | R
            add = new & ~R
            if not add.any():
                break
            rank[add] = n
            R = new
        return rank

    def attractor(self, S, M, goal, maxit=2000):
        """Diem bat dong NHO NHAT doi ngau voi (18): tap o tu do co the BAO DAM toi dich ma van o
        trong S, bang cac hanh dong cua mat na M. Tra ve so buoc bao dam toi dich (hang) cua moi o,
        -1 voi o khong bao dam duoc."""
        g = self.g
        R = goal & S
        rank = np.where(R, 0, -1).astype(np.int32)
        for n in range(1, maxit):
            new = np.zeros_like(R)
            for k in range(g.nth):
                acc = np.zeros((g.nx, g.ny, g.nz), dtype=bool)
                for a3 in range(NA3):
                    acc |= M[:, :, :, k, a3] & self._all_in(R, k, a3)
                new[:, :, :, k] = acc
            new = (new & S) | R
            add = new & ~R
            if not add.any():
                break
            rank[add] = n
            R = new
        return rank

    def mask(self, S):
        g = self.g
        M = np.zeros((g.nx, g.ny, g.nz, g.nth, NA3), dtype=bool)
        for k in range(g.nth):
            for a3 in range(NA3):
                M[:, :, :, k, a3] = (~self.bad[k, a3]) & self._all_in(S, k, a3)
        return M


def _obs_arrays(env):
    C = np.array(env["cyl"], dtype=float).reshape(-1, 4)
    B = np.array(env["blk"], dtype=float).reshape(-1, 5)
    return C, B


def _seg_hits3(P0, P1, C, B):
    """Day cung ba chieu P0 -> P1 (mang N x 3) co cham vat can da noi rong khong.

    Tap tham so t thuoc [0, 1] ma hinh chieu ngang nam trong hinh da noi rong la mot doan (vi tap
    do loi). Day cung cham vat can khi doan do khac rong va do cao nho nhat tren doan do khong
    vuot qua dinh vat can cong le dung."""
    R, RV = CFG["safeR"], CFG3["safeV"]
    N = P0.shape[0]
    hit = np.zeros(N, dtype=bool)
    dx = P1[:, 0] - P0[:, 0]; dy = P1[:, 1] - P0[:, 1]; dz = P1[:, 2] - P0[:, 2]
    dd = dx * dx + dy * dy

    def disc_interval(ox, oy, rho):
        ax = P0[:, 0] - ox; ay = P0[:, 1] - oy
        b = ax * dx + ay * dy
        cc = ax * ax + ay * ay - rho * rho
        disc = b * b - dd * cc
        ok = disc >= 0.0
        sq = np.sqrt(np.where(ok, disc, 0.0))
        t1 = (-b - sq) / np.where(dd > 0, dd, 1.0)
        t2 = (-b + sq) / np.where(dd > 0, dd, 1.0)
        return ok, t1, t2

    def slab(x0, x1, y0, y1):
        with np.errstate(divide='ignore', invalid='ignore'):
            tx1 = (x0 - P0[:, 0]) / dx; tx2 = (x1 - P0[:, 0]) / dx
            ty1 = (y0 - P0[:, 1]) / dy; ty2 = (y1 - P0[:, 1]) / dy
        inx = (P0[:, 0] >= x0) & (P0[:, 0] <= x1)
        iny = (P0[:, 1] >= y0) & (P0[:, 1] <= y1)
        lo_x = np.where(dx != 0, np.minimum(tx1, tx2), np.where(inx, -np.inf, np.inf))
        hi_x = np.where(dx != 0, np.maximum(tx1, tx2), np.where(inx, np.inf, -np.inf))
        lo_y = np.where(dy != 0, np.minimum(ty1, ty2), np.where(iny, -np.inf, np.inf))
        hi_y = np.where(dy != 0, np.maximum(ty1, ty2), np.where(iny, np.inf, -np.inf))
        return np.maximum(lo_x, lo_y), np.minimum(hi_x, hi_y)

    def zcheck(ok, t1, t2, top):
        a = np.clip(t1, 0.0, 1.0); b = np.clip(t2, 0.0, 1.0)
        ok = ok & (t2 >= 0.0) & (t1 <= 1.0) & (a <= b)
        zmin = np.minimum(P0[:, 2] + a * dz, P0[:, 2] + b * dz)
        return ok & (zmin <= top + RV)

    for ox, oy, r, h in C:
        ok, t1, t2 = disc_interval(ox, oy, r + R)
        hit |= zcheck(ok, t1, t2, h)
    for x0, x1, y0, y1, h in B:
        # hinh chu nhat bo tron = hop cua hai hinh chu nhat noi rong va bon hinh tron o goc
        lo = np.full(N, np.inf); hi = np.full(N, -np.inf)
        for (a0, a1, b0, b1) in ((x0 - R, x1 + R, y0, y1), (x0, x1, y0 - R, y1 + R)):
            l, u = slab(a0, a1, b0, b1)
            m = l <= u
            lo = np.where(m, np.minimum(lo, l), lo); hi = np.where(m, np.maximum(hi, u), hi)
        for (ox, oy) in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):
            ok, t1, t2 = disc_interval(ox, oy, R)
            lo = np.where(ok, np.minimum(lo, t1), lo); hi = np.where(ok, np.maximum(hi, t2), hi)
        hit |= zcheck(lo <= hi, lo, hi, h)
    return hit


def _inside3(P):
    return ((P[:, 0] >= 0.0) & (P[:, 0] <= CFG["W"]) & (P[:, 1] >= 0.0) & (P[:, 1] <= CFG["H"])
            & (P[:, 2] >= CFG3["z0"]) & (P[:, 2] <= CFG3["z1"]))


def _point_in3(P, C, B):
    """Diem P (N x 3) nam trong vat can da noi rong."""
    R, RV = CFG["safeR"], CFG3["safeV"]
    hit = np.zeros(P.shape[0], dtype=bool)
    for ox, oy, r, h in C:
        hit |= (np.hypot(P[:, 0] - ox, P[:, 1] - oy) <= r + R) & (P[:, 2] <= h + RV)
    for x0, x1, y0, y1, h in B:
        dx = np.maximum(np.maximum(x0 - P[:, 0], P[:, 0] - x1), 0.0)
        dy = np.maximum(np.maximum(y0 - P[:, 1], P[:, 1] - y1), 0.0)
        hit |= (np.hypot(dx, dy) <= R) & (P[:, 2] <= h + RV)
    return hit


def clear3(s, sn, C, B):
    P0 = np.array([s[:3]]); P1 = np.array([sn[:3]])
    return bool(_inside3(P1)[0]) and not bool(_seg_hits3(P0, P1, C, B)[0])


def _seg_clear_fast(p0, p1, obs):
    """Phien ban vo huong cua _seg_hits3 cho mot day cung va danh sach vat can o gan (nhanh hon
    nhieu khi so vat can nho). Tra ve True khi day cung khong cham vat can nao."""
    R, RV = CFG["safeR"], CFG3["safeV"]
    x0, y0, z0 = p0; x1, y1, z1 = p1
    dx = x1 - x0; dy = y1 - y0; dz = z1 - z0
    dd = dx * dx + dy * dy

    def disc(ox, oy, rho):
        ax = x0 - ox; ay = y0 - oy
        b = ax * dx + ay * dy
        cc = ax * ax + ay * ay - rho * rho
        D = b * b - dd * cc
        if D < 0.0:
            return None
        q = math.sqrt(D)
        return (-b - q) / dd, (-b + q) / dd

    def slab(a0, a1, b0, b1):
        lo, hi = -math.inf, math.inf
        for p, d, u0, u1 in ((x0, dx, a0, a1), (y0, dy, b0, b1)):
            if d == 0.0:
                if p < u0 or p > u1:
                    return None
            else:
                t1 = (u0 - p) / d; t2 = (u1 - p) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                lo = max(lo, t1); hi = min(hi, t2)
        return (lo, hi) if lo <= hi else None

    for o in obs:
        if o[0] == 0:
            iv = disc(o[1], o[2], o[3] + R)
            top = o[4]
        else:
            _, bx0, bx1, by0, by1, top = o
            lo, hi = math.inf, -math.inf
            for iv2 in (slab(bx0 - R, bx1 + R, by0, by1), slab(bx0, bx1, by0 - R, by1 + R),
                        disc(bx0, by0, R), disc(bx0, by1, R), disc(bx1, by0, R), disc(bx1, by1, R)):
                if iv2 is not None:
                    lo = min(lo, iv2[0]); hi = max(hi, iv2[1])
            iv = (lo, hi) if lo <= hi else None
        if iv is None:
            continue
        a = min(1.0, max(0.0, iv[0])); b = min(1.0, max(0.0, iv[1]))
        if iv[1] < 0.0 or iv[0] > 1.0 or a > b:
            continue
        if min(z0 + a * dz, z0 + b * dz) <= top + RV:
            return False
    return True


class NearIndex:
    """Danh sach vat can co the cham mot day cung xuat phat tu o (i, j) cua luoi."""

    def __init__(self, grid, env):
        c = grid.cell
        reach = LSTEP + CFG["safeR"] + 0.01
        self.idx = {}
        obs = [(0,) + tuple(o) for o in env["cyl"]] + [(1,) + tuple(o) for o in env["blk"]]
        for i in range(grid.nx):
            for j in range(grid.ny):
                L_ = []
                for o in obs:
                    if o[0] == 0:
                        dx = max(i * c - o[1], o[1] - (i + 1) * c, 0.0)
                        dy = max(j * c - o[2], o[2] - (j + 1) * c, 0.0)
                        if math.hypot(dx, dy) <= o[3] + reach:
                            L_.append(o)
                    else:
                        dx = max(o[1] - (i + 1) * c, i * c - o[2], 0.0)
                        dy = max(o[3] - (j + 1) * c, j * c - o[4], 0.0)
                        if math.hypot(dx, dy) <= reach:
                            L_.append(o)
                self.idx[(i, j)] = L_
        self.c, self.nx, self.ny = c, grid.nx, grid.ny

    def clear(self, s, sn):
        if not (0.0 <= sn[0] <= CFG["W"] and 0.0 <= sn[1] <= CFG["H"]
                and CFG3["z0"] <= sn[2] <= CFG3["z1"]):
            return False
        i = min(self.nx - 1, max(0, int(s[0] // self.c)))
        j = min(self.ny - 1, max(0, int(s[1] // self.c)))
        return _seg_clear_fast(s[:3], sn[:3], self.idx[(i, j)])


def _near(s, C, B, reach):
    """Loc truoc cac vat can o gan de kiem tra nhanh trong luc huan luyen."""
    x, y = s[0], s[1]
    Cn = C[np.hypot(C[:, 0] - x, C[:, 1] - y) <= C[:, 2] + reach] if len(C) else C
    if len(B):
        dx = np.maximum(np.maximum(B[:, 0] - x, x - B[:, 1]), 0.0)
        dy = np.maximum(np.maximum(B[:, 2] - y, y - B[:, 3]), 0.0)
        Bn = B[np.hypot(dx, dy) <= reach]
    else:
        Bn = B
    return Cn, Bn


def train3(grid, env, mode="off", episodes=10000, seed=SEED, mask=None, log=0, rank=None):
    """Hoc Q dang bang trong khong gian ba chieu. mode: off, step1 (loc mot buoc chinh xac), mask.
    rank: so buoc toi dich tren do thi o (Verifier3.steps_to_goal), dung lam the nang dinh hinh phan
    thuong; khoang cach duong thang tao ngo cut sau cac toa nha cao va lam viec hoc ket trong vong lap."""
    rng = np.random.default_rng(seed)
    C, B = _obs_arrays(env)
    Q = np.zeros((grid.nx, grid.ny, grid.nz, grid.nth, NA3), dtype=np.float32)
    seen = np.zeros((grid.nx, grid.ny, grid.nz, grid.nth), dtype=bool)
    gx, gy = CFG["goal"]
    eps, eta = CFG["eps0"], CFG["eta0"]
    ed = math.exp(math.log(CFG["eps_min"] / CFG["eps0"]) / (0.75 * episodes))
    nd = math.exp(math.log(CFG["eta_min"] / CFG["eta0"]) / (0.85 * episodes))
    succ = np.zeros(episodes); coll = np.zeros(episodes)
    dead = 0
    NI = NearIndex(grid, env)
    s0 = (CFG["start"][0], CFG["start"][1], CFG3["z_start"], CFG["start"][2])
    allA = list(range(NA3))

    if rank is not None:
        RK = np.where(rank >= 0, rank, rank.max() + 20).astype(np.float32) * LSTEP

    def dist(s):
        """Thanh phan tien bo cua phan thuong: so buoc toi dich tren do thi o (nhan voi buoc v dt)
        neu co, nguoc lai la khoang cach ngang duong thang toi dich."""
        if rank is not None:
            return float(RK[grid.phi3(s)])
        return math.hypot(s[0] - gx, s[1] - gy)

    def adm(s):
        if mode == "mask":
            return np.nonzero(mask[grid.phi3(s)])[0].tolist()
        if mode == "off":
            return allA
        return [a for a in allA if NI.clear(s, step3(s, a))]

    t0 = time.time()
    for ep in range(episodes):
        s = s0
        d0 = dist(s)
        hit = rch = False
        A = adm(s)
        for t in range(1, CFG["Tmax"] + 1):
            key = grid.phi3(s)
            seen[key] = True
            if not A:
                A = allA
                dead += 1
            if rng.random() < eps:
                a = int(A[rng.integers(len(A))])
            else:
                q = Q[key][A]
                m = q.max()
                cand = [A[u] for u in range(len(A)) if q[u] >= m - 1e-9]
                a = int(cand[rng.integers(len(cand))])
            sn = step3(s, a)
            ok = NI.clear(s, sn)
            at, az = ACT3[a]
            r = -CFG["lam"] - CFG["kappa"] * abs(ACT[at]) / CFG["wmax"] - CFG3["kz"] * abs(az)
            done = False
            if not ok:
                r = CFG["R_hit"]; done = True; hit = True
            else:
                d1 = dist(sn)
                r += CFG["beta"] * (d0 - d1)
                d0 = d1
                if math.hypot(sn[0] - gx, sn[1] - gy) <= CFG["goal_rad"]:
                    r = CFG["R_goal"]; done = True; rch = True
            if (not done) and t == CFG["Tmax"]:
                r += CFG["R_timeout"]; done = True
            if done:
                y = r
            else:
                An = adm(sn)
                y = r + CFG3["gamma"] * float(Q[grid.phi3(sn)][An or allA].max())
                A = An
            Q[key][a] += eta * (y - Q[key][a])
            if done:
                break
            s = sn
        succ[ep] = rch; coll[ep] = hit
        eps = max(CFG["eps_min"], eps * ed)
        eta = max(CFG["eta_min"], eta * nd)
        if log and (ep + 1) % log == 0:
            print("    ep %5d succ=%.3f coll=%.3f  %.0fs"
                  % (ep + 1, succ[max(0, ep - 999):ep + 1].mean(),
                     coll[max(0, ep - 999):ep + 1].mean(), time.time() - t0), flush=True)
    return dict(Q=Q, seen=seen, succ=succ, coll=coll, dead=dead, wall=time.time() - t0)


def greedy3(Q, mask=None, tie=1):
    if mask is None:
        PI = np.argmax(Q, axis=4).astype(np.int8)
        PI[Q.max(axis=4) == Q.min(axis=4)] = ACT3.index((tie, 0))
        return PI
    M = np.where(mask, Q.astype(np.float32), -np.inf)
    PI = np.argmax(M, axis=4).astype(np.int8)
    PI[~mask.any(axis=4)] = ACT3.index((tie, 0))
    return PI


def run3(PI, grid, env, Tmax=None, online=None):
    """Bay theo chinh sach o thuan tuy (online=None) hoac kem bo loc mot buoc (online=Q)."""
    C, B = _obs_arrays(env)
    s = (CFG["start"][0], CFG["start"][1], CFG3["z_start"], CFG["start"][2])
    gx, gy = CFG["goal"]
    P = [s[:3]]
    hit = False
    NI = NearIndex(grid, env)
    for _ in range(Tmax or CFG["Tmax"]):
        if online is None:
            a = int(PI[grid.phi3(s)])
        else:
            A = [u for u in range(NA3) if NI.clear(s, step3(s, u))] or list(range(NA3))
            a = int(A[int(np.argmax(online[grid.phi3(s)][A]))])
        sn = step3(s, a)
        if not clear3(s, sn, C, B):
            hit = True
        s = sn
        P.append(s[:3])
        if hit or math.hypot(s[0] - gx, s[1] - gy) <= CFG["goal_rad"]:
            break
    P = np.array(P)
    return dict(path=P, hit=hit,
                reached=(not hit) and math.hypot(P[-1][0] - gx, P[-1][1] - gy) <= CFG["goal_rad"],
                length=float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))),
                steps=len(P) - 1, zmin=float(P[:, 2].min()), zmax=float(P[:, 2].max()))


def monte_carlo3(PI, grid, env, S, n=10000, horizon=400, seed=7, inside=True):
    """Lay mau trang thai dau tu do trong hoac ngoai tap duoc chung nhan, bay song song n quy dao."""
    rng = np.random.default_rng(seed)
    C, B = _obs_arrays(env)
    X = []
    while len(X) < n:
        m = 4 * n
        cand = np.column_stack([rng.uniform(0, CFG["W"], m), rng.uniform(0, CFG["H"], m),
                                rng.uniform(CFG3["z0"], CFG3["z1"], m),
                                rng.uniform(-math.pi, math.pi, m)])
        free = ~_point_in3(cand[:, :3], C, B)
        for row, f in zip(cand, free):
            if not f:
                continue
            if bool(S[grid.phi3(tuple(row))]) != inside:
                continue
            X.append(row)
            if len(X) >= n:
                break
    X = np.array(X)
    alive = np.ones(n, dtype=bool)
    bad = np.zeros(n, dtype=bool)
    c, hz = grid.cell, grid.hz
    for _ in range(horizon):
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            break
        x, y, z, th = X[idx, 0], X[idx, 1], X[idx, 2], X[idx, 3]
        i = np.clip(np.floor(x / c).astype(int), 0, grid.nx - 1)
        j = np.clip(np.floor(y / c).astype(int), 0, grid.ny - 1)
        mz = np.clip(np.floor((z - CFG3["z0"]) / hz + TOL).astype(int), 0, grid.nz - 1)
        k = np.floor((((th + math.pi) % (2 * math.pi)) - math.pi + math.pi) / grid.dth + TOL)
        k = k.astype(int) % grid.nth
        a = PI[i, j, mz, k]
        at = np.array([ACT3[u][0] for u in range(NA3)])[a]
        az = np.array([ACT3[u][1] for u in range(NA3)])[a]
        t2 = (th + ACT[at] * CFG["dt"] + math.pi) % (2 * math.pi) - math.pi
        N1 = np.column_stack([x + LSTEP * np.cos(t2), y + LSTEP * np.sin(t2),
                              z + az * CFG3["vz"] * CFG["dt"]])
        viol = (~_inside3(N1)) | _seg_hits3(X[idx, :3], N1, C, B)
        bad[idx[viol]] = True
        alive[idx[viol]] = False
        X[idx, 0], X[idx, 1], X[idx, 2], X[idx, 3] = N1[:, 0], N1[:, 1], N1[:, 2], t2
    return int(bad.sum()), n


def study3(names=("E1", "E2", "E3"), episodes=10000, nmc=10000):
    out = {}
    for nm in names:
        env = env3(nm)
        g = Grid3(1.0, 24)
        print("\n[3D %s] %d tru, %d khoi nha, luoi %d x %d x %d x %d = %d o"
              % (nm, len(env["cyl"]), len(env["blk"]), g.nx, g.ny, g.nz, g.nth, g.ncell))
        t0 = time.time(); vf = Verifier3(g, env); tb = time.time() - t0
        t0 = time.time(); S, h = vf.max_invariant(); tf = time.time() - t0
        M = vf.mask(S)
        s0 = (CFG["start"][0], CFG["start"][1], CFG3["z_start"], CFG["start"][2])
        c0 = g.phi3(s0)
        gx, gy = CFG["goal"]
        goal = np.zeros(S.shape, dtype=bool)
        for i in range(g.nx):
            for j in range(g.ny):
                far = max(math.hypot(x - gx, y - gy) for x in (i * g.cell, (i + 1) * g.cell)
                          for y in (j * g.cell, (j + 1) * g.cell))
                if far <= CFG["goal_rad"]:
                    goal[i, j] = True
        t0 = time.time(); rank = vf.steps_to_goal(S, M, goal); trk = time.time() - t0
        print("   canh %d, dung %.2f s, diem bat dong %.2f s (%d vong), bat bien %.2f%%, start=%s"
              % (vf.edges, tb, tf, len(h), 100 * S.mean(), bool(S[c0])))
        res = dict(ncyl=len(env["cyl"]), nblk=len(env["blk"]), ncell=g.ncell, edges=int(vf.edges),
                   build=tb, fix=tf, it=len(h), frac=float(S.mean()), cells=int(S.sum()),
                   start_in=bool(S[c0]), mean_act=float(M.sum(4)[S].mean()),
                   frac_by_z=[float(S[:, :, m, :].mean()) for m in range(g.nz)], train={},
                   rank_time=trk, rank_start=int(rank[c0]), rank_conn=float((rank >= 0).sum() / S.sum()))
        tracks = {}
        Qmask = None
        for mode in ("off", "step1", "mask"):
            r = train3(g, env, mode, episodes, mask=(M if mode == "mask" else None), log=2000, rank=rank)
            PI = greedy3(r["Q"], mask=(M if mode == "mask" else None))
            Sp, hp = vf.verify_policy(PI)
            res["train"][mode] = dict(succ=float(r["succ"][-500:].mean()),
                                      coll=float(r["coll"][-500:].mean()),
                                      coll_total=int(r["coll"].sum()), dead=int(r["dead"]),
                                      seen=int(r["seen"].sum()), wall=r["wall"],
                                      cert=float(Sp.mean()), start_cert=bool(Sp[c0]))
            tr = run3(PI, g, env, online=(r["Q"] if mode == "step1" else None))
            tracks[mode] = tr
            res["train"][mode].update(reached=bool(tr["reached"]), hit=bool(tr["hit"]),
                                      length=tr["length"], zmin=tr["zmin"], zmax=tr["zmax"])
            print("   %-6s succ=%.3f coll=%.3f vc_tong=%d cert=%.2f%% tuyen: den=%s cham=%s L=%.0f"
                  % (mode, res["train"][mode]["succ"], res["train"][mode]["coll"],
                     res["train"][mode]["coll_total"], 100 * Sp.mean(), tr["reached"], tr["hit"],
                     tr["length"]), flush=True)
            if mode == "mask":
                Qmask = r["Q"]
                PImask = PI
            del r
        bi, ti = monte_carlo3(PImask, g, env, S, n=nmc, inside=True)
        bo, to = monte_carlo3(PImask, g, env, S, n=nmc, inside=False)
        res["mc"] = dict(inside=[bi, ti], outside=[bo, to])
        print("   Monte Carlo: trong %d/%d, ngoai %d/%d" % (bi, ti, bo, to))
        res["tracks"] = {k: v["path"].tolist() for k, v in tracks.items()}
        out[nm] = res
        del Qmask, M, S, vf
    return out


def fig_3d(R3, name="Fig_3d.png"):
    """Hang tren: ban do be mat so va tuyen bay tham lam; hang duoi: tuyen theo do cao."""
    from matplotlib.colors import Normalize
    fig = plt.figure(figsize=(14.8 * CM, 11.6 * CM))
    names = list(R3.keys())
    norm = Normalize(vmin=0.0, vmax=14.0)
    lab = {"off": LEG("raw"), "step1": LEG("step1"), "mask": LEG("mask")}
    sty = {"off": ('-', OK[0]), "step1": ('--', OK[1]), "mask": (':', OK[3])}
    for col, nm in enumerate(names):
        env = env3(nm)
        ax = fig.add_subplot(2, 3, col + 1)
        Hm = dsm_raster(env)
        im = ax.imshow(np.where(Hm > 0, Hm, np.nan).T, origin='lower', cmap='YlOrBr', norm=norm,
                       extent=(0, CFG["W"], 0, CFG["H"]), interpolation='nearest')
        for mode in ("off", "step1", "mask"):
            P = np.array(R3[nm]["tracks"][mode])
            ls, cl = sty[mode]
            ax.plot(P[:, 0], P[:, 1], ls, color=cl, lw=1.6 if mode == "mask" else 1.1,
                    label=lab[mode])
        ax.plot(CFG["start"][0], CFG["start"][1], 's', color='#009E73', ms=5, mec='k', mew=0.5)
        ax.plot(CFG["goal"][0], CFG["goal"][1], '*', color='#CC0000', ms=8, mec='k', mew=0.5)
        ax.set_xlim(0, CFG["W"]); ax.set_ylim(0, CFG["H"]); ax.set_aspect('equal')
        ax.set_title(L('(%s) Environment %s' % ("abc"[col], nm),
                       '(%s) Môi trường %s' % ("abc"[col], nm)))
        ax.set_xlabel('x (m)')
        if col == 0:
            ax.set_ylabel('y (m)')
        ax2 = fig.add_subplot(2, 3, col + 4)
        for mode in ("off", "step1", "mask"):
            P = np.array(R3[nm]["tracks"][mode])
            ls, cl = sty[mode]
            ax2.plot(np.arange(len(P)), P[:, 2], ls, color=cl, lw=1.6 if mode == "mask" else 1.1)
        ax2.axhline(CFG3["z0"], color='0.4', lw=0.8, ls='-')
        ax2.axhline(CFG3["z1"], color='0.4', lw=0.8, ls='-')
        ax2.set_ylim(0, 12)
        ax2.set_xlabel(L('Step', 'Bước'))
        if col == 0:
            ax2.set_ylabel(L('Altitude z (m)', 'Độ cao z (m)'))
        ax2.grid(True, alpha=0.3, ls=':')
    h, lb = fig.axes[0].get_legend_handles_labels()
    fig.legend(h, lb, loc='lower center', ncol=3, fontsize=7.6, framealpha=1.0,
               bbox_to_anchor=(0.45, 0.0), columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0.0, 0.07, 0.86, 1.0))
    cax = fig.add_axes([0.875, 0.58, 0.016, 0.34])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(L('Obstacle height (m)', 'Chiều cao vật cản (m)'), fontsize=8.5)
    save(fig, name)


# ============ 10. CHUONG TRINH CHINH ============
def main(quick=False):
    EP = 2000 if quick else 10000
    NMC = 4000 if quick else 20000
    TH = SCEN["S1"]
    g = Grid(1.0, 24)
    print("Ban kinh vong luon du phong = %.4f m, buoc = %.2f m" % (RPOLY, LSTEP))
    print("Luoi o: %d x %d x %d = %d o" % (g.nx, g.ny, g.nth, g.ncell))

    print("\n[1] Dung do thi o - o va tim tap bat bien lon nhat")
    t0 = time.time()
    vf = Verifier(g, TH)
    tbuild = time.time() - t0
    t0 = time.time()
    S, hist = vf.max_invariant()
    tfix = time.time() - t0
    M = vf.mask(S)
    print("   canh do thi = %d, dung do thi %.2f s, diem bat dong %.2f s, %d vong"
          % (vf.edges, tbuild, tfix, len(hist)))
    print("   tap bat bien lon nhat = %d o, %.1f%%" % (S.sum(), 100.0 * S.mean()))
    print("   so hanh dong duoc phep trung binh trong tap = %.2f" % M.sum(3)[S].mean())

    print("\n[2] Huan luyen ba phuong an")
    runs = {}
    runs["raw"] = train(g, TH, "off", EP, log=1000, snap=500)
    print("   %-8s succ=%.3f coll=%.3f dead=%d %.0fs"
          % ("raw", runs["raw"]["succ"][-500:].mean(), runs["raw"]["coll"][-500:].mean(),
             runs["raw"]["dead"], runs["raw"]["wall"]))
    runs["step1"] = train(g, TH, "none", EP, log=1000, snap=500)
    print("   %-8s succ=%.3f coll=%.3f dead=%d %.0fs"
          % ("step1", runs["step1"]["succ"][-500:].mean(),
             runs["step1"]["coll"][-500:].mean(), runs["step1"]["dead"],
             runs["step1"]["wall"]))
    runs["online"] = train(g, TH, "cert", EP, log=1000, snap=500)
    print("   %-8s succ=%.3f coll=%.3f dead=%d %.0fs"
          % ("online", runs["online"]["succ"][-500:].mean(),
             runs["online"]["coll"][-500:].mean(), runs["online"]["dead"],
             runs["online"]["wall"]))
    runs["mask"] = train(g, TH, "mask", EP, mask=M, log=1000, snap=500)
    print("   %-8s succ=%.3f coll=%.3f dead=%d %.0fs"
          % ("mask", runs["mask"]["succ"][-500:].mean(), runs["mask"]["coll"][-500:].mean(),
             runs["mask"]["dead"], runs["mask"]["wall"]))

    print("\n[3] Kiem chung chinh sach o cua tung bang Q")
    ver = {}
    PIs = {}
    for tag in ("raw", "step1", "online", "mask"):
        PI = greedy(runs[tag]["Q"], mask=(M if tag == "mask" else None))
        PIs[tag] = PI
        t0 = time.time()
        Sp, hp = vf.verify_policy(PI)
        ver[tag] = dict(frac=float(Sp.mean()), cells=int(Sp.sum()), it=len(hp),
                        time=time.time() - t0)
        seen = runs[tag]["seen"] > 0
        c0 = g.phi(CFG["start"])
        ver[tag].update(start=bool(Sp[c0]),
                        unvisited_in_S=float(((~seen) & S).sum() / S.sum()),
                        cert_seen=float((Sp & seen).sum() / max(1, seen.sum())))
        if tag != "mask":
            # thay quy tac pha hoa: o chua tham thi quay het toc do (0: phai, 2: trai)
            best_t = 0.0
            for tie in (0, 2):
                St, _ = vf.verify_policy(greedy(runs[tag]["Q"], tie=tie))
                best_t = max(best_t, float(St.mean()))
            ver[tag]["frac_turn_tie"] = best_t
        print("   %-8s chung nhan %6.2f%%  (%d o, %d vong, %.2f s) start=%s unvisitedS=%.2f "
              "turn_tie=%.4f"
              % (tag, 100.0 * ver[tag]["frac"], ver[tag]["cells"], ver[tag]["it"],
                 ver[tag]["time"], ver[tag]["start"], ver[tag]["unvisited_in_S"],
                 ver[tag].get("frac_turn_tie", -1)))

    print("\n[4] Sua chinh sach da hoc trong tap bat bien")
    PR, ch = vf.repair(PIs["online"], S, runs["online"]["Q"])
    Sr, hr = vf.verify_policy(PR)
    seen_online = runs["online"]["seen"] > 0
    rep = dict(changed=int(ch.sum()), frac_changed=float(ch.sum() / S.sum()),
               frac=float(Sr.mean()),
               changed_seen=int((ch & seen_online).sum()),
               seen_in_S=int((S & seen_online).sum()))
    PIs["repair"] = PR
    print("   so o bi sua = %d, chiem %.1f%% cua tap bat bien, sau khi sua chung nhan %.2f%%"
          % (rep["changed"], 100.0 * rep["frac_changed"], 100.0 * rep["frac"]))

    print("\n[5] Do nhay theo do phan giai cua bo luong tu hoa")
    res = []
    reps = 1 if quick else 5
    for cell in (2.0, 1.5, 1.0, 0.75, 0.5, 0.375, 0.25):
        for nth in (12, 24, 36):
            gg = Grid(cell, nth)
            best = None
            for _ in range(reps):
                t0 = time.time()
                vv = Verifier(gg, TH)
                Sx, hx = vv.max_invariant()
                dt_ = time.time() - t0
                best = dt_ if best is None else min(best, dt_)
            res.append(dict(cell=cell, nth=nth, ncell=gg.ncell, frac=float(Sx.mean()),
                            it=len(hx), time=best, edges=int(vv.edges)))
            print("   o %.2f m, %2d o goc: %8d o, bat bien %5.2f%%, %5.2f s"
                  % (cell, nth, gg.ncell, 100.0 * res[-1]["frac"], res[-1]["time"]))

    print("\n[6] Doi chieu voi kiem chung bang lay mau")
    mc_in = monte_carlo(PIs["mask"], g, TH, S, n=NMC, inside=True)
    mc_out = monte_carlo(PIs["mask"], g, TH, S, n=NMC, inside=False)
    print("   trong tap: %d va cham tren %d diem xuat phat" % mc_in)
    print("   ngoai tap: %d va cham tren %d diem xuat phat" % mc_out)
    curve = sampling_curve(PIs["online"], g, TH, S,
                           [100, 300, 1000, 3000, 10000] if quick
                           else [100, 300, 1000, 3000, 10000, 30000])
    print("   duong cong lay mau:", curve)

    print("\n[7] Chi phi tinh toan khi bay")
    t0 = time.time()
    s = CFG["start"]
    for _ in range(20000):
        allowed(s, TH, "cert")
    t_online = (time.time() - t0) / 20000.0
    t0 = time.time()
    for _ in range(20000):
        _ = int(PIs["mask"][g.phi(s)])
    t_mask = (time.time() - t0) / 20000.0
    print("   bo loc truc tuyen %.1f us moi buoc, tra bang %.1f us moi buoc"
          % (1e6 * t_online, 1e6 * t_mask))

    tracks = {}
    tracks["raw"] = run_cell_policy(PIs["raw"], g, TH)
    tracks["step1"] = run_online(runs["step1"]["Q"], g, TH, "none")
    tracks["online"] = run_online(runs["online"]["Q"], g, TH, "cert")
    tracks["mask"] = run_cell_policy(PIs["mask"], g, TH)
    tracks["repair"] = run_cell_policy(PIs["repair"], g, TH)
    for k, v in tracks.items():
        print("   tuyen %-8s reach=%s hit=%s L=%.1f mc=%.2f"
              % (k, v["reached"], v["hit"], v["length"], v["minclear"]))

    reach_mask = vf.reachable(M, g.phi(CFG["start"]))

    print("\n[8] Thi nghiem bo sung cho vong phan bien")
    rnd = roundoff_example()
    print("   vi du lam tron:", rnd)
    lex = lemma_example(g)
    print("   vi du Bo de 1 va 2:", lex)
    t0 = time.time()
    lchk = lemma_check(g, vf, TH, n=100000 if quick else 400000)
    lchk["time"] = time.time() - t0
    print("   kiem tra Bo de 1, 2 va Dinh ly 1:", lchk)
    uns = unsound_study(g, TH, runs["mask"]["Q"], S, vf, n=NMC)
    ck = {}
    for tag in ("raw", "step1", "online", "mask"):
        ck[tag] = checkpoint_cert(vf, runs[tag]["snaps"], mask=(M if tag == "mask" else None))
        print("   moc huan luyen %-7s:" % tag, [(e, round(100 * f, 2)) for e, f in ck[tag]])
    scl = scale_check(20.0)
    print("   kiem tra ty le:", scl)
    nbytes = dict(mask_bits=int(M.size), Q_bytes=int(runs["mask"]["Q"].nbytes))

    print("\n[9] Mo rong ba chieu")
    R3 = study3(episodes=2 * EP, nmc=NMC // 2)

    PAY = dict(cell=1.0, nth=24, threats=TH, S=S, ks=[0, 8, 16],
               hists=[("inv", hist, S.size)],
               res=res,
               tracks={k: v["path"] for k, v in tracks.items()},
               curve=curve,
               curves={k: dict(succ=runs[k]["succ"], coll=runs[k]["coll"])
                       for k in ("raw", "step1", "online", "mask")})
    for tag in ("raw", "step1", "online"):
        _, hp = vf.verify_policy(PIs[tag])
        PAY["hists"].append((tag, hp, S.size))
    PAY.update(ck=ck, colls={k: runs[k]["coll"] for k in ("raw", "step1", "online", "mask")},
               PI_mask=PIs["mask"], Q_mask=runs["mask"]["Q"], R3=R3)
    with open(os.path.join(FIG, "plotdata.pkl"), "wb") as f:
        pickle.dump(PAY, f)
    for lg in ("en", "vi"):
        render_all(PAY, lg)

    summ = dict(
        grid=dict(cell=1.0, nth=24, nx=g.nx, ny=g.ny, ncell=g.ncell, edges=int(vf.edges),
                  build=tbuild, fix=tfix, it=len(hist)),
        inv=dict(cells=int(S.sum()), frac=float(S.mean()),
                 mean_actions=float(M.sum(3)[S].mean()),
                 reach=int(reach_mask.sum()), reach_frac=float(reach_mask.sum() / S.sum())),
        train={k: dict(succ=float(runs[k]["succ"][-500:].mean()),
                       coll=float(runs[k]["coll"][-500:].mean()),
                       coll_total=int(runs[k]["coll"].sum()),
                       dead=int(runs[k]["dead"]),
                       seen=int((runs[k]["seen"] > 0).sum()),
                       wall=float(runs[k]["wall"])) for k in runs},
        ver=ver, repair=rep, res=res,
        mc=dict(inside=mc_in, outside=mc_out, curve=curve),
        cost=dict(online_us=1e6 * t_online, mask_us=1e6 * t_mask),
        tracks={k: dict(reached=bool(v["reached"]), hit=bool(v["hit"]),
                        length=float(v["length"]), minclear=float(v["minclear"]),
                        steps=int(v["steps"])) for k, v in tracks.items()},
        RPOLY=RPOLY, episodes=EP,
        roundoff=rnd, lemma_example=lex, lemma_check=lchk, unsound=uns,
        checkpoints={k: [(e, f) for e, f in v] for k, v in ck.items()}, scale=scl,
        memory=nbytes,
        threeD={k: {kk: vv for kk, vv in v.items() if kk != "tracks"} for k, v in R3.items()})
    with open(os.path.join(FIG, "results.json"), "w") as f:
        json.dump(summ, f, indent=1, default=float)
    print("\nDa luu ket qua vao figs/p3/results.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="chay nhanh de thu (it tap hon)")
    ap.add_argument("--replot", action="store_true", help="chi ve lai hinh tu plotdata.pkl")
    ap.add_argument("--only3d", action="store_true", help="chi chay lai phan ba chieu, giu phan phang")
    a = ap.parse_args()
    if a.replot:
        replot()
    elif a.only3d:
        R3 = study3(episodes=20000, nmc=10000)
        with open(os.path.join(FIG, "plotdata.pkl"), "rb") as f:
            PAY = pickle.load(f)
        PAY["R3"] = R3
        with open(os.path.join(FIG, "plotdata.pkl"), "wb") as f:
            pickle.dump(PAY, f)
        with open(os.path.join(FIG, "results.json")) as f:
            summ = json.load(f)
        summ["threeD"] = {k: {kk: vv for kk, vv in v.items() if kk != "tracks"} for k, v in R3.items()}
        with open(os.path.join(FIG, "results.json"), "w") as f:
            json.dump(summ, f, indent=1, default=float)
        for lg in ("en", "vi"):
            render_all(PAY, lg)
    else:
        main(quick=a.quick)
