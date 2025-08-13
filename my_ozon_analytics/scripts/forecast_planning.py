
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sku_analytics import (
        load_orders,
        load_sales,
        load_returns,
        load_costs,
        compute_promos,
        compute_analytics,
    )
except Exception as e:
    raise ImportError("Не удалось импортировать sku_analytics. Убедитесь, что файл sku_analytics.py доступен.") from e


@dataclass
class Assumptions:
    price_drift: float = 0.0
    promo_delta_pp: float = 0.0
    commission_delta_pp: float = 0.0
    returns_delta_pp: float = 0.0
    capacity_limit: Optional[float] = None


class ForecastPlanner:
    def _score_and_recommend(self, r: pd.Series) -> tuple[float, str, str]:
        price = float(r.get("avg_net_price", 0.0)) or 1.0
        promo_share = float(r.get("unit_cost", 0.0) - r.get("commission_per_unit", 0.0,)) / price if False else 0.0
        sku = r["sku"]
        base = self.analytics_df.set_index("sku")
        comm_u = float(base.loc[sku, "commission_per_unit"]) if "commission_per_unit" in base.columns and sku in base.index else 0.0
        promo_u = float(base.loc[sku, "promo_per_unit"]) if "promo_per_unit" in base.columns and sku in base.index else 0.0
        promo_share = (promo_u / price) if price else 0.0
        commission_share = (comm_u / price) if price else 0.0

        margin_pct = float(r.get("forecast_margin_pct", 0.0))
        returns_pct = float(base.loc[sku, "returns_pct"]) if "returns_pct" in base.columns and sku in base.index else 0.0

        margin_component = max(0.0, min(40.0, margin_pct)) / 40.0 * 60.0

        penalty = 0.0
        if returns_pct > 18: penalty += 30.0
        elif returns_pct > 12: penalty += 15.0
        if promo_share > 0.12: penalty += 10.0
        if commission_share > 0.20: penalty += 10.0

        bonus = 0.0
        abc = base.loc[sku, "ABC_class"] if "ABC_class" in base.columns and sku in base.index else None
        xyz = base.loc[sku, "XYZ_class"] if "XYZ_class" in base.columns and sku in base.index else None
        if abc == "A": bonus += 8.0
        elif abc == "B": bonus += 4.0
        if xyz == "X": bonus += 6.0
        elif xyz == "Y": bonus += 3.0

        score = max(0.0, min(100.0, margin_component - penalty + bonus))

        action = "Не запускать / Под вопросом"
        reason = []
        if margin_pct < 5.0 and promo_share > 0.12:
            action = "Урезать промо (≤10% от цены)"
            reason.append("низкая маржа и высокая доля промо")
        if margin_pct < 5.0 and returns_pct > 12.0:
            action = "Снизить возвраты (качество/описание/упаковка/логистика)"
            reason.append("низкая маржа и высокий возврат")
        if margin_pct < 5.0 and commission_share > 0.20:
            action = "Поднять цену (комиссия >20% от цены)"
            reason.append("низкая маржа и высокая комиссия")
        if margin_pct >= 5.0 and score >= 60.0:
            action = "Масштабировать (включить в план)"
            if not reason:
                reason.append("устойчивая маржа и приемлемые риски")
        if margin_pct <= 0.0 and score < 40.0:
            action = "Не запускать (убыточно)"
            if not reason:
                reason.append("отрицательная маржа")

        return float(score), "; ".join(reason) if reason else "средний риск-профиль", action

    def add_recommendations(self) -> None:
        if self.future_metrics.empty:
            return
        rows = []
        for idx, r in self.future_metrics.iterrows():
            score, reason, action = self._score_and_recommend(r)
            rr = r.to_dict()
            rr["score"] = score
            rr["recommendation_reason"] = reason
            rr["recommended_action"] = action
            rows.append(rr)
        self.future_metrics = pd.DataFrame(rows)

        agg = (
            self.future_metrics.groupby("sku", as_index=False)
            .agg(
                total_net_qty=("net_qty","sum"),
                total_revenue=("forecast_revenue","sum"),
                total_margin=("forecast_margin","sum"),
                avg_margin_pct=("forecast_margin_pct","mean"),
                avg_score=("score","mean"),
                recommended_any=("recommended", lambda s: "Да" if (s=="Да").any() else "Нет"),
                main_action=("recommended_action", lambda s: s.value_counts().idxmax() if not s.empty else "N/A"),
            )
        )
        agg["recommended_for_plan"] = np.where((agg["avg_score"]>=60) & (agg["avg_margin_pct"]>=5.0), "Да", "Нет")
        self.production_just_df = agg[[
            "sku","total_net_qty","total_revenue","total_margin","avg_margin_pct","avg_score","main_action","recommended_for_plan"
        ]]

    def save_gold(self, gold_dir: Path) -> None:
        gold_dir = Path(gold_dir)
        gold_dir.mkdir(parents=True, exist_ok=True)
        (self.future_metrics.sort_values(["sku","period"]) if not self.future_metrics.empty else
         pd.DataFrame(columns=[
            "sku","period","forecast_qty","planned_inbound","net_qty","avg_net_price","unit_cost",
            "forecast_revenue","forecast_cost","forecast_margin","forecast_margin_pct",
            "recommended","score","recommendation_reason","recommended_action"
         ])
        ).to_csv(gold_dir / "forecast_sku_monthly.csv", index=False, encoding="utf-8-sig")
        (self.production_just_df.sort_values(["recommended_for_plan","avg_score","total_margin"], ascending=[False,False,False]) if not self.production_just_df.empty else
         pd.DataFrame(columns=["sku","total_net_qty","total_revenue","total_margin","avg_margin_pct","avg_score","main_action","recommended_for_plan"])
        ).to_csv(gold_dir / "production_justification.csv", index=False, encoding="utf-8-sig")
        self.assumptions_df.to_csv(gold_dir / "assumptions.csv", index=False, encoding="utf-8-sig")
    def __init__(
        self,
        orders_path: Path,
        sales_path: Path,
        returns_path: Path,
        costs_path: Path,
        horizon: int = 3,
        model: str = "ets",
        backtest_n: int = 0,
        assumptions: Optional[Assumptions] = None,
        planned_inbound_path: Optional[Path] = None,
        min_margin_pct: float = 0.05,
        min_batch: int = 1,
    ) -> None:
        self.orders_path = Path(orders_path)
        self.sales_path = Path(sales_path)
        self.returns_path = Path(returns_path)
        self.costs_path = Path(costs_path)
        self.horizon = int(horizon)
        self.model = model.lower()
        self.backtest_n = int(backtest_n)
        self.assumptions = assumptions or Assumptions()
        self.planned_inbound_path = planned_inbound_path
        self.min_margin_pct = float(min_margin_pct)
        self.min_batch = int(min_batch)

        self.orders_df: pd.DataFrame = pd.DataFrame()
        self.sales_df: pd.DataFrame = pd.DataFrame()
        self.returns_df: pd.DataFrame = pd.DataFrame()
        self.costs_df: pd.DataFrame = pd.DataFrame()
        self.analytics_df: pd.DataFrame = pd.DataFrame()

        self.monthly_sales: pd.DataFrame = pd.DataFrame()
        self.forecasts: Dict[str, np.ndarray] = {}

        self.future_metrics: pd.DataFrame = pd.DataFrame()
        self.backtest_df: pd.DataFrame = pd.DataFrame()
        self.assumptions_df: pd.DataFrame = pd.DataFrame()
        self.production_just_df: pd.DataFrame = pd.DataFrame()

        self.planned_inbound_df: pd.DataFrame = pd.DataFrame()

    def load_and_prepare(self) -> None:
        self.orders_df = load_orders(self.orders_path)
        self.sales_df = load_sales(self.sales_path)
        self.returns_df = load_returns(self.returns_path)
        self.costs_df = load_costs(self.costs_path)
        promos_df = compute_promos(self.orders_df)

        self.analytics_df = compute_analytics(
            sales_df=self.sales_df,
            returns_df=self.returns_df,
            promos_df=promos_df,
            costs_df=self.costs_df,
            orders_df=self.orders_df,
            return_window_days=90,
            promo_df=None,
            accrual_df=None,
            inventory_df=None,
        )

        od = self.orders_df.dropna(subset=["month_ship"]).copy()
        od["month_ship"] = od["month_ship"].astype("period[M]")
        piv = od.pivot_table(index="month_ship", columns="sku", values="qty_shipped", aggfunc="sum").sort_index()
        piv.columns = piv.columns.astype(str)
        self.monthly_sales = piv.fillna(0)

        if self.planned_inbound_path and Path(self.planned_inbound_path).exists():
            pi = pd.read_excel(self.planned_inbound_path)
            if "sku" in pi.columns and "period" in pi.columns and "qty" in pi.columns:
                pi["sku"] = pi["sku"].astype(str).str.strip()
                pi["period"] = pd.PeriodIndex(pd.to_datetime(pi["period"].astype(str), errors="coerce"), freq="M")
                pi["qty"] = pd.to_numeric(pi["qty"], errors="coerce").fillna(0)
                self.planned_inbound_df = pi.dropna(subset=["period"])
            else:
                self.planned_inbound_df = pd.DataFrame(columns=["sku","period","qty"])
        else:
            self.planned_inbound_df = pd.DataFrame(columns=["sku","period","qty"])

    def _make_model(self):
        if self.model == "arima":
            return "arima"
        return "ets"

    def _forecast_series(self, series: pd.Series) -> np.ndarray:
        s = series.astype(float).fillna(0)
        if s.sum() == 0 or len(s) < 2:
            return np.zeros(self.horizon)

        model_name = self._make_model()
        if model_name == "ets":
            try:
                from statsmodels.tsa.holtwinters import ExponentialSmoothing
                seasonal = 12 if len(s) >= 24 else None
                mdl = ExponentialSmoothing(s, trend="add", seasonal=seasonal, initialization_method="estimated")
                fit = mdl.fit()
                return fit.forecast(self.horizon).values
            except Exception:
                ma = s.rolling(3, min_periods=1).mean().iloc[-1]
                return np.full(self.horizon, float(ma))
        else:
            try:
                from statsmodels.tsa.statespace.sarimax import SARIMAX
                if len(s) >= 24:
                    fit = SARIMAX(s, order=(1,1,1), seasonal_order=(0,0,0,12),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                else:
                    fit = SARIMAX(s, order=(1,1,1),
                                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
                fc = fit.forecast(self.horizon).values
                return np.maximum(0, fc)
            except Exception:
                ma = s.rolling(3, min_periods=1).mean().iloc[-1]
                return np.full(self.horizon, float(ma))

    def forecast_sales(self) -> None:
        self.forecasts = {}
        if self.monthly_sales.empty:
            self.future_metrics = pd.DataFrame()
            return
        for sku in self.monthly_sales.columns:
            s = self.monthly_sales[sku]
            self.forecasts[sku] = np.maximum(0, self._forecast_series(s))

    def _apply_assumptions_to_units(self, price: float, promo_u: float, comm_u: float, ret_pct: float) -> Tuple[float, float, float, float]:
        price_adj = price * (1.0 + self.assumptions.price_drift)
        promo_adj = promo_u + self.assumptions.promo_delta_pp * price_adj
        comm_adj = comm_u + self.assumptions.commission_delta_pp * price_adj
        ret_adj = max(0.0, ret_pct + self.assumptions.returns_delta_pp)
        return price_adj, promo_adj, comm_adj, ret_adj

    def compute_future_metrics(self) -> None:
        if not self.forecasts:
            self.future_metrics = pd.DataFrame()
            return

        last_period = self.monthly_sales.index.max()
        future_periods = pd.period_range(last_period + 1, periods=self.horizon, freq="M")

        met = self.analytics_df.set_index("sku")
        for c in ["avg_net_price_per_unit","production_cost_per_unit","commission_per_unit","promo_per_unit","returns_pct","ending_stock"]:
            if c not in met.columns:
                met[c] = 0.0

        rows = []
        for sku, fc in self.forecasts.items():
            if sku not in met.index:
                continue
            avg_price = float(met.loc[sku, "avg_net_price_per_unit"])
            prod_cost = float(met.loc[sku, "production_cost_per_unit"])
            comm_cost = float(met.loc[sku, "commission_per_unit"])
            promo_cost = float(met.loc[sku, "promo_per_unit"])
            ret_pct = float(met.loc[sku, "returns_pct"]) / 100.0
            end_stock = float(met.loc[sku, "ending_stock"]) if not pd.isna(met.loc[sku, "ending_stock"]) else 0.0

            price_adj, promo_adj, comm_adj, ret_adj_pp = self._apply_assumptions_to_units(
                avg_price, promo_cost, comm_cost, ret_pct * 100.0
            )
            ret_adj = ret_adj_pp / 100.0
            unit_cost = prod_cost + promo_adj + comm_adj

            inbound_map = {}
            if not self.planned_inbound_df.empty:
                inbound_map = (
                    self.planned_inbound_df[self.planned_inbound_df["sku"] == sku]
                    .set_index("period")["qty"].to_dict()
                )

            available = end_stock
            for i, qty in enumerate(fc):
                per = future_periods[i]
                inbound = float(inbound_map.get(per, 0.0))
                available += inbound
                net_qty = qty * max(0.0, 1.0 - ret_adj)
                sell_qty = min(net_qty, available)
                available = max(0.0, available - sell_qty)

                revenue = sell_qty * price_adj
                cost = sell_qty * unit_cost
                margin = revenue - cost
                margin_pct = (margin / revenue) if revenue > 0 else 0.0
                recommended = bool((margin > 0) and (margin_pct >= self.min_margin_pct) and (sell_qty >= self.min_batch))

                rows.append({
                    "sku": sku,
                    "period": per,
                    "forecast_qty": float(qty),
                    "planned_inbound": float(inbound),
                    "net_qty": float(sell_qty),
                    "avg_net_price": price_adj,
                    "unit_cost": unit_cost,
                    "forecast_revenue": float(revenue),
                    "forecast_cost": float(cost),
                    "forecast_margin": float(margin),
                    "forecast_margin_pct": float(margin_pct * 100.0),
                    "recommended": "Да" if recommended else "Нет",
                })

        self.future_metrics = pd.DataFrame(rows)

        if not self.future_metrics.empty:
            pj = (
                self.future_metrics.groupby("sku", as_index=False)
                .agg(
                    total_net_qty=("net_qty", "sum"),
                    total_revenue=("forecast_revenue", "sum"),
                    total_margin=("forecast_margin", "sum"),
                    avg_margin_pct=("forecast_margin_pct", "mean"),
                    recommended_any=("recommended", lambda s: "Да" if (s=="Да").any() else "Нет"),
                )
            )
        else:
            pj = pd.DataFrame(columns=["sku","total_net_qty","total_revenue","total_margin","avg_margin_pct","recommended_any"])

        self.production_just_df = pj

        self.assumptions_df = pd.DataFrame({
            "parameter": ["model", "horizon", "backtest_n", "price_drift", "promo_delta_pp", "commission_delta_pp", "returns_delta_pp", "capacity_limit", "min_margin_pct", "min_batch"],
            "value": [
                self.model, self.horizon, self.backtest_n,
                self.assumptions.price_drift, self.assumptions.promo_delta_pp,
                self.assumptions.commission_delta_pp, self.assumptions.returns_delta_pp,
                self.assumptions.capacity_limit, self.min_margin_pct, self.min_batch
            ]
        })

    @staticmethod
    def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        denom = np.where(y_true == 0.0, 1.0, y_true)
        return float(np.mean(np.abs((y_true - y_pred) / denom))) * 100.0

    def run_backtest(self) -> None:
        if self.backtest_n <= 0 or self.monthly_sales.empty:
            self.backtest_df = pd.DataFrame()
            return

        bt_rows = []
        for sku in self.monthly_sales.columns:
            s = self.monthly_sales[sku].astype(float)
            if len(s) <= self.backtest_n + 1:
                continue
            train = s.iloc[:-self.backtest_n]
            test = s.iloc[-self.backtest_n:]
            fc = self._forecast_series(train)[:len(test)]
            mape = self._mape(test.values, fc)
            for per, y, yhat in zip(test.index.to_timestamp(), test.values, fc):
                bt_rows.append({
                    "sku": sku,
                    "period": per.to_period("M"),
                    "actual": float(y),
                    "forecast": float(yhat),
                    "abs_perc_error": float(abs((y - yhat) / (y if y != 0 else 1)) * 100.0),
                    "mape_sku": float(mape),
                })
        self.backtest_df = pd.DataFrame(bt_rows)

    def save_report(self, output_path: Path) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(output_path) as writer:
            (self.future_metrics.sort_values(["sku","period"]) if not self.future_metrics.empty else
             pd.DataFrame(columns=["sku","period","forecast_qty","planned_inbound","net_qty","avg_net_price","unit_cost","forecast_revenue","forecast_cost","forecast_margin","forecast_margin_pct","recommended"])
             ).to_excel(writer, sheet_name="Forecast", index=False)

            (self.production_just_df.sort_values("total_margin", ascending=False) if not self.production_just_df.empty else
             pd.DataFrame(columns=["sku","total_net_qty","total_revenue","total_margin","avg_margin_pct","recommended_any"])
             ).to_excel(writer, sheet_name="Production_Justification", index=False)

            if not self.backtest_df.empty:
                summary = self.backtest_df.groupby("sku", as_index=False)["mape_sku"].mean().rename(columns={"mape_sku":"MAPE"})
                summary.loc["ИТОГО","MAPE"] = summary["MAPE"].mean() if not summary.empty else np.nan
                summary.to_excel(writer, sheet_name="Backtest", index=False)
                self.backtest_df.sort_values(["sku","period"]).to_excel(writer, sheet_name="Backtest_Detail", index=False)
            else:
                pd.DataFrame(columns=["sku","period","actual","forecast","abs_perc_error","mape_sku"]).to_excel(writer, sheet_name="Backtest", index=False)

            self.assumptions_df.to_excel(writer, sheet_name="Assumptions", index=False)


def main():
    parser = argparse.ArgumentParser(description="Прогноз и план под Ozon с Assumptions, Backtest и Production Justification")
    parser.add_argument("--orders", type=str, required=True)
    parser.add_argument("--sales", type=str, required=True)
    parser.add_argument("--returns", type=str, required=True)
    parser.add_argument("--costs", type=str, required=True)
    parser.add_argument("--planned-inbound", type=str, default=None)

    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--model", type=str, default="ets", choices=["ets","arima"])
    parser.add_argument("--backtest", type=int, default=0)

    parser.add_argument("--price-drift", type=float, default=0.0)
    parser.add_argument("--promo-delta-pp", type=float, default=0.0)
    parser.add_argument("--commission-delta-pp", type=float, default=0.0)
    parser.add_argument("--returns-delta-pp", type=float, default=0.0)
    parser.add_argument("--capacity-limit", type=float, default=None)

    parser.add_argument("--min-margin-pct", type=float, default=0.05)
    parser.add_argument("--min-batch", type=int, default=1)

    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gold-dir", type=str, default="/Users/aleksandr/Desktop/озон рад/soft/готовое/my_ozon_analytics/gold")

    args = parser.parse_args()

    ass = Assumptions(
        price_drift=args.price_drift,
        promo_delta_pp=args.promo_delta_pp,
        commission_delta_pp=args.commission_delta_pp,
        returns_delta_pp=args.returns_delta_pp,
        capacity_limit=args.capacity_limit,
    )

    planner = ForecastPlanner(
        orders_path=Path(args.orders),
        sales_path=Path(args.sales),
        returns_path=Path(args.returns),
        costs_path=Path(args.costs),
        horizon=args.horizon,
        model=args.model,
        backtest_n=args.backtest,
        assumptions=ass,
        planned_inbound_path=Path(args.planned_inbound) if args.planned_inbound else None,
        min_margin_pct=args.min_margin_pct,
        min_batch=args.min_batch,
    )

    planner.load_and_prepare()
    planner.forecast_sales()
    planner.compute_future_metrics()
    planner.add_recommendations()
    planner.run_backtest()
    planner.save_report(Path(args.output))
    planner.save_gold(Path(args.gold_dir))


if __name__ == "__main__":
    main()
