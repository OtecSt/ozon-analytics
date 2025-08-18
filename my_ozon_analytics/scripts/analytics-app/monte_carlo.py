from dataclasses import dataclass
from typing import Optional, Dict, Iterable, Literal, Mapping, Tuple

import numpy as np
import pandas as pd

# --- Совместимость с app.py: конфиг и фабрика симулятора ---
@dataclass
class MCConfig:
    n_sims: int = 20_000
    seed: Optional[int] = None

# ============================
# Пример CLI (smoke-test)
# ============================

# Удобная фабрика под вызовы из app.py
def build_sim(cfg: "MCConfig") -> "MonteCarloSimulator":
    return MonteCarloSimulator(n_sims=int(cfg.n_sims), random_state=cfg.seed)

import numpy as np
import pandas as pd


# ============================
# Вспомогательные преобразования
# ============================

def _clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def _to_array(x: float | Iterable[float]) -> np.ndarray:
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return np.asarray(x, dtype=float)
    return np.asarray([x], dtype=float)

def _quantiles(x: np.ndarray, qs: Iterable[float]) -> Dict[float, float]:
    return {q: float(np.quantile(x, q)) for q in qs}

def _lognormal_mu_sigma(mean: float, cv: float) -> Tuple[float, float]:
    """
    Для заданных mean и коэффициента вариации cv возвращает параметры mu, sigma
    для np.random.lognormal(mu, sigma).
    """
    mean = float(mean)
    cv = float(cv)
    if mean <= 0 or cv < 0:
        # вырожденный случай — вернём малую дисперсию вокруг mean
        return np.log(max(mean, 1e-12)), 1e-6
    sigma2 = np.log(1.0 + cv * cv)
    mu = np.log(mean) - 0.5 * sigma2
    sigma = np.sqrt(sigma2)
    return mu, sigma


# ============================
# Спеки распределений
# ============================

@dataclass
class NormalSpec:
    """Трунгированная Normal(μ, σ) с отсечкой (lower/upper) — через clip."""
    mean: float
    sd: float
    lower: Optional[float] = None
    upper: Optional[float] = None

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        x = rng.normal(loc=self.mean, scale=max(self.sd, 1e-12), size=n)
        if self.lower is not None:
            x = np.maximum(x, self.lower)
        if self.upper is not None:
            x = np.minimum(x, self.upper)
        return x


@dataclass
class LogNormalSpec:
    """Логнормаль с заданными mean и cv (σ/μ). Гарантированно > 0."""
    mean: float
    cv: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        mu, sigma = _lognormal_mu_sigma(self.mean, self.cv)
        return rng.lognormal(mean=mu, sigma=sigma, size=n)


@dataclass
class BetaSpec:
    """
    Бета-распределение для долей/процентных ставок (например, возвраты).
    Можно задать:
      * mean и kappa (псевдо-объём наблюдений): alpha=mean*kappa, beta=(1-mean)*kappa
      * либо напрямую alpha, beta
    """
    mean: Optional[float] = None
    kappa: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None

    def _params(self) -> Tuple[float, float]:
        if self.alpha is not None and self.beta is not None:
            a, b = float(self.alpha), float(self.beta)
            return max(a, 1e-6), max(b, 1e-6)
        if self.mean is None or self.kappa is None:
            # дефолт: почти детерминированная малая дисперсия около 0
            return 1.0, 1e6
        a = max(self.mean * self.kappa, 1e-6)
        b = max((1.0 - self.mean) * self.kappa, 1e-6)
        return a, b

    def sample01(self, n: int, rng: np.random.Generator) -> np.ndarray:
        a, b = self._params()
        return rng.beta(a, b, size=n)


@dataclass
class TriangularSpec:
    """Треугольное распределение (min, mode, max). Удобно для экспертных оценок."""
    left: float
    mode: float
    right: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return rng.triangular(self.left, self.mode, self.right, size=n)


@dataclass
class DeterministicSpec:
    """Детерминированное значение."""
    value: float

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        return np.full(n, float(self.value))


# ============================
# Результаты моделирования
# ============================

@dataclass
class MonteCarloSummary:
    mean: float
    std: float
    p05: float
    p50: float
    p95: float
    var_05: float   # Value-at-Risk (5% квантиль)
    cvar_05: float  # Conditional VaR (среднее в нижних 5%)

    @staticmethod
    def from_samples(x: np.ndarray) -> "MonteCarloSummary":
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return MonteCarloSummary(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        q = _quantiles(x, [0.05, 0.5, 0.95])
        lower_tail = x[x <= q[0.05]]
        cvar = float(lower_tail.mean()) if lower_tail.size else float(q[0.05])
        return MonteCarloSummary(
            mean=float(x.mean()),
            std=float(x.std(ddof=1)) if x.size > 1 else 0.0,
            p05=float(q[0.05]),
            p50=float(q[0.5]),
            p95=float(q[0.95]),
            var_05=float(q[0.05]),
            cvar_05=cvar,
        )


@dataclass
class MonteCarloResult:
    """
    Храним сырые выборки и агрегаты.
    - unit_margin: выборка маржи за единицу (без возвратов, так корректнее)
    - total_margin: выборка общей маржи по горизонту (учитывает возвраты и qty)
    """
    unit_margin_samples: np.ndarray
    total_margin_samples: np.ndarray
    unit_margin_summary: MonteCarloSummary
    total_margin_summary: MonteCarloSummary

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {
            "unit_margin": self.unit_margin_summary.__dict__,
            "total_margin": self.total_margin_summary.__dict__,
        }


# ============================
# Основной симулятор
# ============================

class MonteCarloSimulator:
    """
    Универсальный Монте-Карло симулятор для unit-economics.

    Что моделируем:
      * price — цена за ед.
      * production_cost — себестоимость
      * commission — комиссия маркетплейса за ед.
      * promo — промо/скидка за ед. (как издержка)
      * returns_rate — доля возвратов (0..1), влияет на net_qty

    Результаты:
      * Распределение маржи за единицу: price - production_cost - commission - promo
      * Распределение общей маржи на горизонте: sum_t (qty_t * (1 - returns_t)) * unit_margin_draw

    Корреляции:
      * Можно задать корреляцию между {price, production_cost, commission, promo}
        через матрицу 4x4 (returns моделируется отдельно — бета распределением).
        Внедряется через совместную генерацию нормалей (Гауссовская копула для
        нормальных/логнормальных/клипованных normal).
    """

    def __init__(self, n_sims: int = 20_000, random_state: Optional[int] = None) -> None:
        self.n_sims = int(n_sims)
        self.rng = np.random.default_rng(random_state)

    # ------ correlated normals helper ------

    def _correlated_normals(self, corr: np.ndarray) -> np.ndarray:
        """
        Возвращает матрицу Z shape=(n_sims, k) ~ N(0, corr).
        Если corr невалидна/не указана — вернём независимые стандарты.
        """
        k = corr.shape[0]
        try:
            L = np.linalg.cholesky(corr)
        except Exception:
            # деградация — независимые нормали
            L = np.eye(k)
        Z = self.rng.standard_normal(size=(self.n_sims, k))
        return Z @ L.T

    # ------ core sampling ------

    def simulate_sku(
        self,
        *,
        # базовые средние значения (fallback, если спецификация не задана)
        base_price: float,
        base_production_cost: float,
        base_commission_per_unit: float,
        base_promo_per_unit: float,
        base_returns_rate: float,  # доля 0..1
        # спецификации распределений (опционально — если None, используем детерминированную базу)
        price_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        production_cost_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        commission_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        promo_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        returns_spec: Optional[BetaSpec | DeterministicSpec] = None,
        # объёмы продаж (по умолчанию один период с qty)
        qty: float | Iterable[float] = 1.0,
        # опциональная корреляция между (price, production_cost, commission, promo)
        # shape=(4,4), симметричная, положительно определённая
        corr_matrix_4x4: Optional[np.ndarray] = None,
    ) -> MonteCarloResult:
        """
        Возвращает сырые выборки и сводки по марже (unit & total).
        Возвраты влияют только на TOTAL через множитель (1 - returns_rate).
        """
        n = self.n_sims
        qty_vec = _to_array(qty)  # shape=(T,)
        T = qty_vec.size

        # --- подготовим спецификации с дефолтами ---
        price_spec = price_spec or DeterministicSpec(base_price)
        pc_spec = production_cost_spec or DeterministicSpec(base_production_cost)
        comm_spec = commission_spec or DeterministicSpec(base_commission_per_unit)
        promo_spec = promo_spec or DeterministicSpec(base_promo_per_unit)
        returns_spec = returns_spec or BetaSpec(mean=float(base_returns_rate), kappa=1e6)  # практически точная доля

        # --- выборки по {price, pc, commission, promo} с корреляцией (если задана) ---
        # Сценарий А: без корреляции — просто сэмплим по каждому спеку
        if corr_matrix_4x4 is None:
            price_draw = price_spec.sample(n, self.rng)
            pc_draw = pc_spec.sample(n, self.rng)
            comm_draw = comm_spec.sample(n, self.rng)
            promo_draw = promo_spec.sample(n, self.rng)
        else:
            # Сценарий B: задаём мультивариативную нормаль, и затем:
            #  * NormalSpec — линейное преобразование от стандартной нормали
            #  * LogNormalSpec — exp(mu + sigma * z)
            #  * Triangular/Deterministic — без корреляции (подмешиваем отдельно)
            # Реалистично — чаще price/pc/commission/promo задаются нормалями/логнормалями.
            Z = self._correlated_normals(np.asarray(corr_matrix_4x4, dtype=float))
            cols = []
            for idx, spec in enumerate([price_spec, pc_spec, comm_spec, promo_spec]):
                z = Z[:, idx]
                if isinstance(spec, NormalSpec):
                    x = spec.mean + spec.sd * z
                    if spec.lower is not None: x = np.maximum(x, spec.lower)
                    if spec.upper is not None: x = np.minimum(x, spec.upper)
                elif isinstance(spec, LogNormalSpec):
                    mu, sigma = _lognormal_mu_sigma(spec.mean, spec.cv)
                    x = np.exp(mu + sigma * z)
                elif isinstance(spec, TriangularSpec):
                    # корреляция для triangular некорректна — возьмём независимые draw
                    x = spec.sample(n, self.rng)
                elif isinstance(spec, DeterministicSpec):
                    x = spec.sample(n, self.rng)
                else:
                    # неизвестный тип — фоллбэк к детерминированному mean
                    x = np.full(n, float(getattr(spec, "mean", 0.0)))
                cols.append(x)
            price_draw, pc_draw, comm_draw, promo_draw = cols

        # --- возвраты (бета/детермин.) ---
        if isinstance(returns_spec, BetaSpec):
            rr = _clip01(returns_spec.sample01(n, self.rng))
        else:
            rr = _clip01(returns_spec.sample(n, self.rng))

        # --- unit margin выборка ---
        unit_margin = price_draw - pc_draw - comm_draw - promo_draw

        # --- итоговая маржа: суммируем по периодам qty_t * (1 - rr_t) * unit_margin_draw
        # Возвраты моделируем по периодам как независимые (одно и то же rr для всех T — ок для коротких горизонтов,
        # но реалистичнее сэмплить rr_t отдельно). Ниже реализуем независимые rr_t.
        rr_mat = self.rng.beta(  # если returns_spec Beta — приближённо используй его среднюю дисперсию
            *(returns_spec._params() if isinstance(returns_spec, BetaSpec) else (1e6, 1.0)),
            size=(n, T)
        ) if isinstance(returns_spec, BetaSpec) else np.tile(rr.reshape(-1, 1), (1, T))

        rr_mat = _clip01(rr_mat)
        qty_row = qty_vec.reshape(1, T).astype(float)
        net_qty_mat = qty_row * (1.0 - rr_mat)  # shape=(n, T)
        total_margin = (net_qty_mat.sum(axis=1)) * unit_margin  # (n,)* (n,) => (n,)

        # --- сводки ---
        unit_summary = MonteCarloSummary.from_samples(unit_margin)
        total_summary = MonteCarloSummary.from_samples(total_margin)

        return MonteCarloResult(
            unit_margin_samples=unit_margin,
            total_margin_samples=total_margin,
            unit_margin_summary=unit_summary,
            total_margin_summary=total_summary,
        )

    # ------ удобные адаптеры под наши данные ------

    def simulate_from_analytics(
        self,
        analytics_row: Mapping[str, float],
        *,
        qty: float | Iterable[float] = 1.0,
        # задание неопределённостей по умолчанию (можно переопределить через *_spec)
        price_cv: float = 0.03,
        production_cost_cv: float = 0.05,
        commission_cv: float = 0.10,
        promo_cv: float = 0.15,
        returns_kappa: float = 400.0,  # чем выше, тем уже бета
        # возможная корреляция между (price, production_cost, commission, promo)
        corr_matrix_4x4: Optional[np.ndarray] = None,
        # явные спецификации (если заданы — переопределяют дефолтные cv/κ)
        price_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        production_cost_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        commission_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        promo_spec: Optional[NormalSpec | LogNormalSpec | TriangularSpec | DeterministicSpec] = None,
        returns_spec: Optional[BetaSpec | DeterministicSpec] = None,
    ) -> MonteCarloResult:
        """
        Упрощённый вызов для строки из analytics_df (из твоего ForecastPlanner):
        ожидаются ключи:
          - avg_net_price_per_unit
          - production_cost_per_unit
          - commission_per_unit
          - promo_per_unit
          - returns_pct  (В ПРОЦЕНТАХ!)

        qty — общий объём горизонта или массив по периодам (до возвратов).
        """
        base_price = float(analytics_row.get("avg_net_price_per_unit", 0.0))
        base_pc = float(analytics_row.get("production_cost_per_unit", 0.0))
        base_comm = float(analytics_row.get("commission_per_unit", 0.0))
        base_promo = float(analytics_row.get("promo_per_unit", 0.0))
        base_rr = float(analytics_row.get("returns_pct", 0.0)) / 100.0

        # если спецификации не заданы — строим дефолтные
        if price_spec is None:
            # цена чаще ближе к нормальной (с отсечкой снизу 0)
            price_spec = NormalSpec(mean=base_price, sd=abs(base_price * price_cv), lower=0.0)
        if production_cost_spec is None:
            # себестоимость >0 — логнормаль
            production_cost_spec = LogNormalSpec(mean=max(base_pc, 1e-9), cv=production_cost_cv)
        if commission_spec is None:
            commission_spec = LogNormalSpec(mean=max(base_comm, 1e-9), cv=commission_cv)
        if promo_spec is None:
            promo_spec = LogNormalSpec(mean=max(base_promo, 1e-9), cv=promo_cv)
        if returns_spec is None:
            returns_spec = BetaSpec(mean=_clip01(np.array([base_rr]))[0], kappa=returns_kappa)

        return self.simulate_sku(
            base_price=base_price,
            base_production_cost=base_pc,
            base_commission_per_unit=base_comm,
            base_promo_per_unit=base_promo,
            base_returns_rate=base_rr,
            price_spec=price_spec,
            production_cost_spec=production_cost_spec,
            commission_spec=commission_spec,
            promo_spec=promo_spec,
            returns_spec=returns_spec,
            qty=qty,
            corr_matrix_4x4=corr_matrix_4x4,
        )

    def simulate_for_forecast(
        self,
        forecast_df: pd.DataFrame,
        analytics_df: pd.DataFrame,
        *,
        periods: Optional[Iterable[pd.Period]] = None,
        n_sims_override: Optional[int] = None,
        # дефолтные неопределённости
        price_cv: float = 0.03,
        production_cost_cv: float = 0.05,
        commission_cv: float = 0.10,
        promo_cv: float = 0.15,
        returns_kappa: float = 400.0,
        corr_matrix_4x4: Optional[np.ndarray] = None,
    ) -> Dict[str, MonteCarloResult]:
        """
        Массовый прогон по всем SKU из прогноза.
        Ожидается forecast_df из Planner (листы "Forecast"): колонки sku, period, forecast_qty.
        Берём base-метрики из analytics_df (по sku).
        Возвращает словарь: sku -> MonteCarloResult.
        """
        if n_sims_override is not None:
            old = self.n_sims
            self.n_sims = int(n_sims_override)

        # фильтруем горизонты
        df = forecast_df.copy()
        if periods is not None:
            per_set = set(map(lambda p: str(p), periods))
            df = df[df["period"].astype(str).isin(per_set)]

        # соберём qty-вектора по sku
        qty_map: Dict[str, np.ndarray] = {}
        for sku, g in df.groupby("sku"):
            # используем GROSS прогноз до возвратов (forecast_qty)
            q = g.sort_values("period")["forecast_qty"].to_numpy(dtype=float)
            qty_map[str(sku)] = q

        # быстрый индекс для аналитики
        aidx = analytics_df.set_index("sku")

        results: Dict[str, MonteCarloResult] = {}
        for sku, qty_vec in qty_map.items():
            if sku not in aidx.index:
                continue
            row = aidx.loc[sku].to_dict()

            res = self.simulate_from_analytics(
                analytics_row=row,
                qty=qty_vec,
                price_cv=price_cv,
                production_cost_cv=production_cost_cv,
                commission_cv=commission_cv,
                promo_cv=promo_cv,
                returns_kappa=returns_kappa,
                corr_matrix_4x4=corr_matrix_4x4,
            )
            results[sku] = res

        if n_sims_override is not None:
            self.n_sims = old

        return results


# ============================
# Утиль: быстрые фабрики specs
# ============================

def make_default_specs_from_analytics(
    analytics_row: Mapping[str, float],
    *,
    price_cv: float = 0.03,
    production_cost_cv: float = 0.05,
    commission_cv: float = 0.10,
    promo_cv: float = 0.15,
    returns_kappa: float = 400.0,
) -> Dict[str, object]:
    """
    Быстро получить набор дефолтных спецификаций под одну строку analytics_df.
    Удобно, если хочешь передать в simulate_sku() свои параметры.
    """
    base_price = float(analytics_row.get("avg_net_price_per_unit", 0.0))
    base_pc = float(analytics_row.get("production_cost_per_unit", 0.0))
    base_comm = float(analytics_row.get("commission_per_unit", 0.0))
    base_promo = float(analytics_row.get("promo_per_unit", 0.0))
    base_rr = float(analytics_row.get("returns_pct", 0.0)) / 100.0

    return {
        "price_spec": NormalSpec(mean=base_price, sd=abs(base_price * price_cv), lower=0.0),
        "production_cost_spec": LogNormalSpec(mean=max(base_pc, 1e-9), cv=production_cost_cv),
        "commission_spec": LogNormalSpec(mean=max(base_comm, 1e-9), cv=commission_cv),
        "promo_spec": LogNormalSpec(mean=max(base_promo, 1e-9), cv=promo_cv),
        "returns_spec": BetaSpec(mean=_clip01(np.array([base_rr]))[0], kappa=returns_kappa),
    }


# ============================
# Пример CLI (smoke-test)
# ============================

if __name__ == "__main__":
    # Небольшая демонстрация без файлов.
    sim = MonteCarloSimulator(n_sims=10000, random_state=42)

    # Базовые метрики (условные)
    base = dict(
        base_price=1200.0,
        base_production_cost=650.0,
        base_commission_per_unit=120.0,
        base_promo_per_unit=60.0,
        base_returns_rate=0.08,  # 8%
    )

    # Горизонт: три месяца, прогноз до возвратов
    qty = [300, 320, 350]

    # Простые specs (либо можно не задавать — возьмутся дефолты)
    result = sim.simulate_sku(
        **base,
        qty=qty,
        price_spec=NormalSpec(mean=1200.0, sd=36.0, lower=0.0),           # ~3% sd
        production_cost_spec=LogNormalSpec(mean=650.0, cv=0.07),          # ~7% cv
        commission_spec=LogNormalSpec(mean=120.0, cv=0.10),               # ~10% cv
        promo_spec=LogNormalSpec(mean=60.0, cv=0.20),                      # ~20% cv
        returns_spec=BetaSpec(mean=0.08, kappa=400.0),                     # узкое вокруг 8%
        corr_matrix_4x4=np.array([                                          # разумные корреляции
            [1.0,  0.3,  0.2, -0.1],
            [0.3,  1.0,  0.2,  0.1],
            [0.2,  0.2,  1.0,  0.2],
            [-0.1, 0.1,  0.2,  1.0],
        ])
    )

    print("Unit margin summary:", result.unit_margin_summary)
    print("Total margin summary:", result.total_margin_summary)