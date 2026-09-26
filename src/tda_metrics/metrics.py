"""Метрики схожести двух облаков точек (P — данные, Q — модель).

Тяжелые зависимости (ripser++, MTopDiv, RTD, tensorflow) устанавливаются
внешне — см. ноутбук notebooks/topology_metrics.ipynb.
"""
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_v2_behavior()

import mtd
import rtd
from precision_recall import knn_precision_recall_features
from scipy.cluster.hierarchy import linkage
from scipy.linalg import sqrtm
from scipy.special import digamma, gammaln
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist, pdist, squareform

__all__ = ['TopologyMetrics', 'np_random_seed']


from contextlib import contextmanager


@contextmanager
def np_random_seed(seed):
    """Детерминирует глобальный numpy RNG внутри блока.

    mtd и rtd используют глобальный np.random для подсэмплирования — фиксируем
    состояние на время вызова, чтобы метрики не зависели от порядка выполнения.
    """
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class TopologyMetrics:
    """Единый интерфейс к метрикам схожести двух облаков точек.

    P — «данные» (reference), Q — «модель».

    Асимметричные: mtd (направления P->Q и Q->P), improved precision/recall.
    Симметричные: rtd, mmd, frechet, js.

    Параметры:
        seed: сид для стохастических внутренних подсэмплирований (mtd, rtd);
        device: 'cpu' | 'cuda:N' — где считать попарные расстояния.
    """

    def __init__(self, seed=42, device='cpu'):
        self.seed = seed
        self.device = device
        self.ntd_repeats = 5

    # ---------- топологические ----------

    def mtd(self, P, Q, batch_size1=None, batch_size2=None, n=1):
        """Manifold Topology Divergence: сумма длин H1-баров кросс-баркода.

        Асимметрична: считайте mtd(P, Q) и mtd(Q, P).
        batch_size1 / batch_size2 — сколько точек брать из P и Q (по умолчанию все);
        n — число повторов со случайным подсэмплированием (нужно только если
        батчи меньше облаков).
        """
        bs1 = len(P) if batch_size1 is None else min(batch_size1, len(P))
        bs2 = len(Q) if batch_size2 is None else min(batch_size2, len(Q))
        with np_random_seed(self.seed):
            value = mtd.mtopdiv(
                P, Q, batch_size1=bs1, batch_size2=bs2, n=n, pdist_device=self.device,
            )
        return float(value)

    def mtd_homology(self, P, Q):
        """Суммы длин баров кросс-баркода по гомологиям: H0 — связность, H1 — циклы.

        mtd(P, Q) эквивалентен mtd_homology(P, Q)['H1'] при полных батчах.
        """
        with np_random_seed(self.seed):
            barcodes = mtd.calc_cross_barcodes(
                P, Q,
                batch_size1=len(P), batch_size2=len(Q),
                pdist_device=self.device, dim=1, is_plot=False,
            )
        scores = {}
        for level, bars in enumerate(barcodes):
            bars = np.asarray(bars, dtype=float).reshape(-1, 2)
            scores[f'H{level}'] = float(np.sum(bars[:, 1] - bars[:, 0]))
        return scores

    def mtd0(self, P, Q):
        """mtd^0: средняя длина H0-баров кросс-баркода (Dmitriev et al., IEEE Access 2025).

        Внутренние расстояния Q обнулены (Q предслит в один блок), H0-бары —
        на каких дистанциях точки P подключаются к многообразию Q; среднее вместо
        суммы снимает зависимость от размера облака. H0-баркод фильтрации
        вырождается в merge-дистанции single-linkage кластеризации, поэтому
        считается напрямую через scipy (без ripser++/joblib: детерминированно и
        быстро на любых размерах).
        """
        P = np.asarray(P, dtype=np.float64)
        Q = np.asarray(Q, dtype=np.float64)
        pooled = np.vstack([Q, P])
        full = squareform(pdist(pooled))
        n_q = len(Q)
        full[:n_q, :n_q] = 0.0
        condensed = squareform(full, checks=False)
        merges = linkage(condensed, method='single')
        return float(merges[:, 2].sum() / len(P))

    def ntd(self, X, Y, n_repeats=10, subsample_size=50, factor=3):
        """Normalized Topological Divergence (Dmitriev et al., IEEE Access 2025).

        NTD(X, Y) = E[mtd^0(Xs, Yb)] / E[mtd^0(Ys, Yb)]: кросс-расходимость X
        против Y, нормированная на шумовой пол референса Y (две независимые
        подвыборки Y). X — «модель», Y — «данные».

        Смысл: ~1 — выборки одного закона; < 1 — точки X близки к многообразию Y
        (X «внутри» Y, precision-подобно); > 1 — у X есть моды вне Y
        (mode invention). NTD(Y, X) — обратное направление (покрытие, recall-подобно).
        Подвыборки: Xs, Ys по subsample_size точек, Yb — в factor раз больше
        (правило 3 * n_s <= n_b из статьи).
        """
        if len(Y) < 2 * subsample_size:
            raise ValueError('ntd требует |Y| >= 2 * subsample_size')
        m = min(subsample_size, len(X), len(Y) // factor)
        rng = np.random.default_rng(self.seed)
        nominator = []
        denominator = []
        for _ in range(n_repeats):
            Xs = X[rng.choice(len(X), size=m, replace=False)]
            Ys = Y[rng.choice(len(Y), size=m, replace=False)]
            Yb = Y[rng.choice(len(Y), size=factor * m, replace=False)]
            nominator.append(self.mtd0(Xs, Yb))
            denominator.append(self.mtd0(Ys, Yb))
        return float(np.mean(nominator) / np.mean(denominator))

    def rtd(self, P, Q, trials=10, batch=500):
        """Representation Topology Divergence. Симметрична, требует |P| == |Q|."""
        if len(P) != len(Q):
            raise ValueError('rtd определен для облаков одинакового размера')
        if trials < 1:
            raise ValueError('trials должен быть >= 1')
        with np_random_seed(self.seed):
            value = rtd.rtd(
                P, Q, pdist_device=self.device, trials=trials, batch=min(batch, len(P)),
            )
        return float(value)

    # ---------- precision / recall ----------

    def improved_precision_recall(self, P, Q, nhood_sizes=(1, 3, 10)):
        """Improved precision and recall (Kynkäänniemi et al., 2019).

        precision@k — доля точек Q внутри kNN-многообразия P;
        recall@k — доля точек P внутри kNN-многообразия Q.
        Возвращает плоский словарь {'precision@k': ..., 'recall@k': ...}.
        """
        config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=config):
            state = knn_precision_recall_features(
                P, Q,
                nhood_sizes=list(nhood_sizes),
                row_batch_size=10000,
                col_batch_size=50000,
                num_gpus=1,
            )
        tf.reset_default_graph()
        return {
            **{f'precision@{k}': float(v) for k, v in zip(nhood_sizes, state['precision'])},
            **{f'recall@{k}': float(v) for k, v in zip(nhood_sizes, state['recall'])},
        }

    # ---------- симметричные статистические ----------

    def mmd(self, P, Q, gamma='median'):
        """Maximum Mean Discrepancy с RBF-ядром; несмещенная оценка MMD^2, возвращается корень.

        gamma='median' — медианная эвристика: gamma = 1 / median^2, где median —
        медиана попарных расстояний объединенной выборки (т.е. sigma = median / sqrt(2)).
        """
        pooled = np.vstack([P, Q])
        if gamma == 'median':
            median = float(np.median(pdist(pooled)))
            gamma = 1.0 / max(median ** 2, 1e-12)

        def rbf_kernel(A, B):
            return np.exp(-gamma * cdist(A, B, metric='sqeuclidean'))

        Kxx, Kyy, Kxy = rbf_kernel(P, P), rbf_kernel(Q, Q), rbf_kernel(P, Q)
        n, m = len(P), len(Q)
        mmd2 = (Kxx.sum() - np.trace(Kxx)) / (n * (n - 1)) \
             + (Kyy.sum() - np.trace(Kyy)) / (m * (m - 1)) \
             - 2.0 * Kxy.mean()
        return float(np.sqrt(max(mmd2, 0.0)))

    def frechet_distance(self, P, Q):
        """Fréchet distance между гауссианами, подогнанными к облакам (формула FID).

        d^2 = ||mu1 - mu2||^2 + tr(S1 + S2 - 2 (S1 S2)^(1/2)).
        Классический FID — та же формула на фичах InceptionV3. Ковариации
        регуляризуются eps*I — важно при d > n (эмбеддинги).
        """
        mu1, mu2 = P.mean(axis=0), Q.mean(axis=0)
        S1, S2 = np.cov(P, rowvar=False), np.cov(Q, rowvar=False)
        d = P.shape[1]
        S1 = S1 + (1e-8 * np.trace(S1) / d + 1e-12) * np.eye(d)
        S2 = S2 + (1e-8 * np.trace(S2) / d + 1e-12) * np.eye(d)
        cov_mean = sqrtm(S1 @ S2)
        if np.iscomplexobj(cov_mean):
            cov_mean = cov_mean.real
        d2 = np.sum((mu1 - mu2) ** 2) + np.trace(S1) + np.trace(S2) - 2.0 * np.trace(cov_mean)
        return float(np.sqrt(max(d2, 0.0)))

    def js_divergence(self, P, Q, k=5):
        """Дивергенция Дженсена–Шеннона (в битах, 0 <= JS <= 1) для наборов точек.

        JS(P, Q) = H(M) - (H(P) + H(Q)) / 2, где M — смесь 0.5P + 0.5Q; объединенная
        выборка — честная выборка из M. Энтропии — kNN-оценка Козаченко–Леоненко.
        """

        def knn_entropy(X):
            n, d = X.shape
            dist, _ = cKDTree(X).query(X, k=k + 1)
            rho = np.maximum(dist[:, -1], 1e-12)
            log_ball_volume = 0.5 * d * np.log(np.pi) - gammaln(0.5 * d + 1.0)
            return digamma(n) - digamma(k) + log_ball_volume + (d / n) * np.sum(np.log(rho))

        pooled = np.vstack([P, Q])
        js = knn_entropy(pooled) - 0.5 * (knn_entropy(P) + knn_entropy(Q))
        return float(max(js, 0.0) / np.log(2.0))

    # ---------- все сразу ----------

    def compute_all(self, P, Q, nhood_sizes=(1, 3, 10), js_k=5, rtd_trials=5, rtd_batch=500, skip=()):
        """Все метрики одним вызовом; плоский словарь -> строка pandas.DataFrame.

        rtd_trials=5 по умолчанию (10 усреднений, как в статье RTD, вдвое дороже
        по времени — основная стоимость compute_all как раз в RTD).
        ntd_PQ / ntd_QP — Normalized Topological Divergence в двух направлениях
        (n_repeats усреднений по подвыборкам; 'ntd' в skip выключает оба).
        skip — имена метрик, которые не считать ('rtd', 'ntd' полезны в циклах).
        """
        result = {
            'mtd_PQ': self.mtd(P, Q),
            'mtd_QP': self.mtd(Q, P),
            'mmd': self.mmd(P, Q),
            'frechet': self.frechet_distance(P, Q),
            'js': self.js_divergence(P, Q, k=js_k),
        }
        if 'rtd' not in skip:
            result['rtd'] = self.rtd(P, Q, trials=rtd_trials, batch=rtd_batch)
        if 'ntd' not in skip:
            result['ntd_PQ'] = self.ntd(P, Q, n_repeats=self.ntd_repeats)
            result['ntd_QP'] = self.ntd(Q, P, n_repeats=self.ntd_repeats)
        result.update(self.improved_precision_recall(P, Q, nhood_sizes=nhood_sizes))
        return result
