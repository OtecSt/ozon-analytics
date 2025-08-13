# planner.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal

import pandas as pd

# Берём существующие классы из твоего модуля
from forecast_planning import ForecastPlanner, Assumptions


@dataclass
class PlanResult:
    """
    Возвращаемый набор таблиц после расчёта плана.
    Ничего не пишется на диск — все данные только в памяти.
    """
    forecast: pd.DataFrame                 # помесячный прогноз + юнит-экономика (аналог листа "Forecast")
    production_justification: pd.DataFrame # агрегированный justification по SKU
    backtest: pd.DataFrame                 # результаты бэктеста (если включён)
    assumptions: pd.DataFrame              # фактически использованные параметры
    monthly_sales: pd.DataFrame            # исторические помесячные отгрузки (wide-формат: index=period, cols=sku)
    analytics: pd.DataFrame                # рассчитанная аналитика из ForecastPlanner (unit-econ по SKU)

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        """Удобный доступ как к словарю — например, для Streamlit tabs."""
        return {
            "forecast": self.forecast,
            "production_justification": self.production_justification,
            "backtest": self.backtest,
            "assumptions": self.assumptions,
            "monthly_sales": self.monthly_sales,
            "analytics": self.analytics,
        }


def _safe_path(p: Optional[str | Path]) -> Optional[Path]:
    if p is None:
        return None
    pp = Path(p)
    return pp if pp.exists() else None


def run_planning(
    *,
    orders_path: str | Path,
    sales_path: str | Path,
    returns_path: str | Path,
    costs_path: str | Path,
    planned_inbound_path: Optional[str | Path] = None,
    horizon: int = 3,
    model: Literal["ets", "arima"] = "ets",
    backtest: int = 0,
    price_drift: float = 0.0,
    promo_delta_pp: float = 0.0,
    commission_delta_pp: float = 0.0,
    returns_delta_pp: float = 0.0,
    capacity_limit: Optional[float] = None,
    min_margin_pct: float = 0.05,
    min_batch: int = 1,
) -> PlanResult:
    """
    Тонкая обёртка над ForecastPlanner: читает исходники по путям,
    считает прогноз/юнит-экономику и возвращает DataFrame’ы (без записи на диск).

    Параметры:
    - *_path: пути к файлам исходных выгрузок (см. твой ForecastPlanner)
    - horizon: горизонт (в месяцах)
    - model: "ets" или "arima"
    - backtest: длина holdout’а (в месяцах). 0 — отключить.
    - *_delta_pp: поправки в ПРОЦЕНТНЫХ пунктах (0.03 == +3 п.п.)
    - price_drift: относительное изменение цены (0.05 == +5%)
    - capacity_limit: опциональный лимит мощностей (пробрасывается в assumptions)
    - min_margin_pct: минимальная маржинальность для флага "recommended"
    - min_batch: минимальный объём продаж для "recommended"
    """
    orders_path = Path(orders_path)
    sales_path = Path(sales_path)
    returns_path = Path(returns_path)
    costs_path = Path(costs_path)
    inbound_path = _safe_path(planned_inbound_path)

    # Собираем допущения
    ass = Assumptions(
        price_drift=price_drift,
        promo_delta_pp=promo_delta_pp,
        commission_delta_pp=commission_delta_pp,
        returns_delta_pp=returns_delta_pp,
        capacity_limit=capacity_limit,
    )

    # Инициализируем планировщик
    planner = ForecastPlanner(
        orders_path=orders_path,
        sales_path=sales_path,
        returns_path=returns_path,
        costs_path=costs_path,
        horizon=int(horizon),
        model=model,
        backtest_n=int(backtest),
        assumptions=ass,
        planned_inbound_path=inbound_path,
        min_margin_pct=float(min_margin_pct),
        min_batch=int(min_batch),
    )

    # Пайплайн без записи на диск
    planner.load_and_prepare()
    planner.forecast_sales()
    planner.compute_future_metrics()
    planner.run_backtest()

    # Собираем результат
    return PlanResult(
        forecast=planner.future_metrics.copy(),
        production_justification=planner.production_just_df.copy(),
        backtest=planner.backtest_df.copy(),
        assumptions=planner.assumptions_df.copy(),
        monthly_sales=planner.monthly_sales.copy(),
        analytics=planner.analytics_df.copy(),
    )


# --- Небольшой CLI для smoke-теста (ничего не пишет на диск) ---
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Run planning in-memory (no file outputs).")
    ap.add_argument("--orders", required=True)
    ap.add_argument("--sales", required=True)
    ap.add_argument("--returns", required=True)
    ap.add_argument("--costs", required=True)
    ap.add_argument("--planned-inbound", default=None)

    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--model", choices=["ets", "arima"], default="ets")
    ap.add_argument("--backtest", type=int, default=0)

    ap.add_argument("--price-drift", type=float, default=0.0)
    ap.add_argument("--promo-delta-pp", type=float, default=0.0)
    ap.add_argument("--commission-delta-pp", type=float, default=0.0)
    ap.add_argument("--returns-delta-pp", type=float, default=0.0)
    ap.add_argument("--capacity-limit", type=float, default=None)

    ap.add_argument("--min-margin-pct", type=float, default=0.05)
    ap.add_argument("--min-batch", type=int, default=1)

    args = ap.parse_args()

    res = run_planning(
        orders_path=args.orders,
        sales_path=args.sales,
        returns_path=args.returns,
        costs_path=args.costs,
        planned_inbound_path=args.planned_inbound,
        horizon=args.horizon,
        model=args.model,
        backtest=args.backtest,
        price_drift=args.price_drift,
        promo_delta_pp=args.promo_delta_pp,
        commission_delta_pp=args.commission_delta_pp,
        returns_delta_pp=args.returns_delta_pp,
        capacity_limit=args.capacity_limit,
        min_margin_pct=args.min_margin_pct,
        min_batch=args.min_batch,
    )

    # Простой отчёт в stdout
    print("forecast:", res.forecast.shape)
    print("production_justification:", res.production_justification.shape)
    print("backtest:", res.backtest.shape)
    print("assumptions:", res.assumptions.shape)
    print("monthly_sales:", res.monthly_sales.shape)
    print("analytics:", res.analytics.shape)