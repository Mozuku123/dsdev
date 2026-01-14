import time
import warnings
from itertools import product

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, kpss


# ----------------------------------------------------------------------
# 1.  差分次数の自動判定 ―― pmdarima.ndiffs / nsdiffs 互換
# ----------------------------------------------------------------------
def ndiffs(y, *, max_d=2, alpha=0.05, test="kpss"):
    """
    Return the minimum d (0…max_d) that passes the chosen unit-root test.
    test='kpss' (pmdarima デフォルト)│'adf'│'kpss-adf'
    """
    y = np.asarray(y, dtype=float)
    for d in range(max_d + 1):
        series = np.diff(y, d) if d else y
        if len(series) < 10:
            break
        try:
            if test == "kpss":
                if kpss(series, regression="c")[1] > alpha:
                    return d
            elif test == "adf":
                if adfuller(series, regression="c")[1] < alpha:
                    return d
            else:                               # 'kpss-adf'
                adf_p = adfuller(series, regression="c")[1]
                kpss_p = kpss(series, regression="c")[1]
                if adf_p < alpha and kpss_p > alpha:
                    return d
        except Exception:                       # 値が足りない / 特異行列 など
            continue
    return max_d


def nsdiffs(y, m, *, max_D=1, alpha=0.05):
    """
    Very small surrogate for pmdarima.nsdiffs (Osborn seasonality test)  0‒max_D
    """
    y = np.asarray(y, dtype=float)
    for D in range(max_D + 1):
        series = np.diff(y, D * m) if D else y
        if len(series) < m + 1:
            break
        # “季節差分後” に分散が大きく減れば季節成分があるとみなす
        seas = np.diff(series, m)
        if np.std(seas) < np.std(series) * 0.8:
            return D
    return 0


# ----------------------------------------------------------------------
# 2.  単一モデルをフィットして IC を返すヘルパ
# ----------------------------------------------------------------------
def _fit(y, order, seas_order, trend, ic):
    """Fit a single ARIMA / SARIMA model and return (results, IC)."""
    if seas_order is None:                       # ===== 非季節 ARIMA =====
        mod = ARIMA(
            y, order=order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        # ARIMA は既定の 'innovations_mle' で OK
        res = mod.fit()
    else:                                        # ===== SARIMA =====
        mod = SARIMAX(
            y, order=order, seasonal_order=seas_order, trend=trend,
            enforce_stationarity=False, enforce_invertibility=False
        )
        # SARIMAX は従来通り lbfgs を明示
        res = mod.fit(method="lbfgs", disp=False)
    return res, getattr(res, ic)

# ----------------------------------------------------------------------
# 3.  auto_arima 本体
# ----------------------------------------------------------------------
def auto_arima(
    y,
    *,
    # -------- 基本オプション --------
    seasonal=False,
    m=12,
    stepwise=True,
    trace=False,
    suppress_warnings=True,
    information_criterion="aic",
    # -------- 差分次数探索 --------
    max_d=2,
    max_D=1,
    d_test="kpss",  # 'kpss'|'adf'|'kpss-adf'
    # -------- パラメータ上限 --------
    max_p=5,
    max_q=5,
    max_P=2,
    max_Q=2,
    max_order=None,  # ← stepwise 中は “None で制限なし” が pmdarima 流
    # -------- ステップワイズ初期点 --------
    start_p=2,
    start_q=2,
    start_P=1,
    start_Q=1,
    # -------- その他 --------
    trend=None,
):
    """
    Drop-in replacement for pmdarima.auto_arima using statsmodels.
    Only `ic` (=information_criterion) and `trace` are fully supported.
    """

    if suppress_warnings:
        warnings.filterwarnings("ignore")

    # -------- 0. データ整形 --------
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.squeeze()               # Series にして index を保持
        if not np.issubdtype(y.dtype, np.number):
            y = y.astype(float)
    else:
        # 配列が来た場合だけ ndarray に変換
        y = np.asarray(y, dtype=float)
        if y.ndim != 1:
            raise ValueError("y must be 1-d array-like")

    # -------- 1. d, D を自動決定 --------
    d = ndiffs(y, max_d=max_d, test=d_test)
    D = nsdiffs(y, m, max_D=max_D) if seasonal else 0

    # -------- 2. デフォルト trend（定数項） --------
    trend = trend if trend is not None else ("c" if (d + D) == 0 else None)

    # -------- 3. ステップワイズ探索 --------
    def candidates(p, q, P, Q):
        """
        Hyndman-Khandakar 近傍集合。pmdarima と同じ 4/8 方向。
        max_p/q/P/Q は常に尊重するが、max_order は stepwise では無視。
        """
        cand = {
            (p + 1, q,     P, Q),
            (p,     q + 1, P, Q),
            (p - 1, q,     P, Q) if p else None,
            (p,     q - 1, P, Q) if q else None,
            (p + 1, q + 1, P, Q),
            (p - 1, q - 1, P, Q) if p and q else None,
            (p + 1, q - 1, P, Q) if q else None,
            (p - 1, q + 1, P, Q) if p else None,
        }
        if seasonal:
            cand |= {
                (p, q, P + 1, Q),
                (p, q, P, Q + 1),
                (p, q, P - 1, Q) if P else None,
                (p, q, P, Q - 1) if Q else None,
            }
        res = {
            c
            for c in cand
            if c
            and c[0] <= max_p
            and c[1] <= max_q
            and c[2] <= max_P
            and c[3] <= max_Q
        }
        return res

    # -- 3-A. 初期モデル ----------------------------------------------
    order0 = (start_p, d, start_q)
    seas0 = (start_P, D, start_Q, m) if seasonal else None
    best_res, best_ic = _fit(y, order0, seas0, trend, information_criterion)
    best_order, best_seas = order0, seas0
    tried = {(*best_order, *(best_seas[:3] if seasonal else ()))}
    if trace:
        seas_s = f"{best_seas}" if best_seas else ""
        print(f"START  {best_order}{seas_s:<15} {information_criterion.upper()}={best_ic: .3f}")

    # -- 3-B. 近傍をたどるステップワイズ -------------------------------
    improving = True
    while improving:
        improving = False
        p, _, q = best_order
        P = best_seas[0] if seasonal else 0
        Q = best_seas[2] if seasonal else 0

        # -----------------------------------------------------------------
        # ★1) まず八近傍
        # -----------------------------------------------------------------
        neigh = sorted(candidates(p, q, P, Q))
        # ★2) その次に “遠い” 候補を追加
        #     (p+2,q), (p,q+2), (p+2,q+2) などを試して脱局所最適
        jump = {
            (p + 2, q,     P, Q),
            (p,     q + 2, P, Q),
            (p + 2, q + 2, P, Q),
            (p + 2, q + 1, P, Q),
            (p + 1, q + 2, P, Q),
        }
        neigh += [c for c in jump if c in candidates(p+2, q+2, P, Q)]
 
        for cand in neigh:
            np_, nq_, nP_, nQ_ = cand
            if (*cand,) in tried:
                continue
            tried.add((*cand,))

            order = (np_, d, nq_)
            seas = (nP_, D, nQ_, m) if seasonal else None

            # stepwise 中は max_order を強制しない（pmdarima 仕様）
            t0 = time.time()
            try:
                res, ic_val = _fit(y, order, seas, trend, information_criterion)
            except Exception:
                continue

            if trace:
                sec = time.time() - t0
                seas_s = f"{seas}" if seas else ""
                print(f"  {order}{seas_s:<15} {information_criterion.upper()}={ic_val: .3f} "
                      f"Δ={ic_val - best_ic: .3f}  ({sec: .2f}s)")

            if ic_val + 1e-8 < best_ic:      # 改善があれば
                best_ic, best_res = ic_val, res
                best_order, best_seas = order, seas
                improving = True            # さらに近傍を探す
                break                       # 新しい近傍集合へ

    # -- 3-C. 総当たり（stepwise=False） -------------------------------
    if not stepwise:
        rng_p = range(max_p + 1)
        rng_q = range(max_q + 1)
        rng_P = range(max_P + 1) if seasonal else (0,)
        rng_Q = range(max_Q + 1) if seasonal else (0,)

        for p, q, P, Q in product(rng_p, rng_q, rng_P, rng_Q):
            if max_order is not None and (p + q + P + Q) > max_order:
                continue
            if (p, d, q, P, D, Q) in tried:
                continue
            tried.add((p, d, q, P, D, Q))
            order = (p, d, q)
            seas = (P, D, Q, m) if seasonal else None
            try:
                res, ic_val = _fit(y, order, seas, trend, information_criterion)
            except Exception:
                continue
            if ic_val + 1e-8 < best_ic:
                best_ic, best_res = ic_val, res
                best_order, best_seas = order, seas
                if trace:
                    print(f"* NEW BEST {order}{seas or ''}  {information_criterion.upper()}={ic_val: .3f}")

    if trace:
        seas_s = f"{best_seas}" if seasonal else ""
        print(f"\n==> BEST  ARIMA{best_order}{seas_s}   "
              f"{information_criterion.upper()}={best_ic: .3f}")

    return best_res
