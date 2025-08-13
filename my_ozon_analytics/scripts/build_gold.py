#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_gold.py — генерация GOLD-слоя для дашбордов из выгрузок Ozon.
Создаёт CSV:
- fact_sku_daily.csv
- fact_sku_monthly.csv
- mart_unit_econ.csv
- data_dictionary.csv

Зависит от sku_analytics.py (лежит рядом или в PYTHONPATH).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np

# --- Normalization helpers ---

def normalize_sku(s):
    import pandas as pd
    s = pd.Series(s, copy=False).astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    return s

# --- Universal file loader: CSV ("," or ";") and Excel ---
def load_any(path):
    path = Path(path)
    if path.suffix.lower() in [".xls", ".xlsx"]:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        # first try default comma; if single column and semicolons in header, retry with sep=';'
        try:
            df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="utf-8", low_memory=False)
        if df.shape[1] == 1:
            try:
                with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                    head = f.read(4096)
                if ";" in head:
                    try:
                        df = pd.read_csv(path, sep=";", encoding="utf-8-sig", low_memory=False)
                    except Exception:
                        df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
            except Exception:
                pass
        return df
    raise ValueError(f"Unsupported file format: {path}")

# Импорт из аналитического модуля
from sku_analytics import (
    load_orders,
    load_sales,
    load_returns,
    load_costs,
    load_promo_actions,
    load_accruals,
    load_inventory,
    compute_promos,
    compute_analytics,
)

def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

def build_daily_fact(orders_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    orders_df = orders_df.copy()
    returns_df = returns_df.copy()
    # Coerce dates if they are not datetime yet
    if not np.issubdtype(getattr(orders_df.get("date_ship"), "dtype", np.dtype("O")), np.datetime64):
        orders_df["date_ship"] = pd.to_datetime(orders_df.get("date_ship"), errors="coerce")
    if not np.issubdtype(getattr(returns_df.get("date_return"), "dtype", np.dtype("O")), np.datetime64):
        returns_df["date_return"] = pd.to_datetime(returns_df.get("date_return"), errors="coerce")
    # Normalize keys
    if "sku" in orders_df.columns:
        orders_df["sku"] = normalize_sku(orders_df["sku"])
    if "sku" in returns_df.columns:
        returns_df["sku"] = normalize_sku(returns_df["sku"])
    # Отгрузки (по дате отгрузки)
    ship = orders_df.dropna(subset=["date_ship"]).copy()
    ship["date"] = ship["date_ship"].dt.date
    ship_g = ship.groupby(["date", "sku"], as_index=False).agg(
        shipped_qty=("qty_shipped", "sum"),
        promo_rub=("discount_rub", "sum"),
        order_value_rub_sum=("order_value_rub", "sum"),
        shipments=("shipment_id", "nunique"),
    )
    # Возвраты (по дате возврата)
    ret = returns_df.dropna(subset=["date_return"]).copy()
    ret["date"] = ret["date_return"].dt.date
    ret_g = ret.groupby(["date", "sku"], as_index=False).agg(
        returns_qty=("returns_qty", "sum"),
        returns_rub=("returns_rub", "sum"),
    )
    # Джойн
    daily = pd.merge(ship_g, ret_g, on=["date", "sku"], how="outer").fillna(0)
    # Типы
    int_cols = ["shipped_qty","shipments","returns_qty"]
    for c in int_cols:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0).astype(int)
    float_cols = ["promo_rub","order_value_rub_sum","returns_rub"]
    for c in float_cols:
        daily[c] = pd.to_numeric(daily[c], errors="coerce").fillna(0.0)
    # Сортировка
    daily = daily.sort_values(["date","sku"]).reset_index(drop=True)
    return daily

def build_monthly_fact(orders_df: pd.DataFrame, returns_df: pd.DataFrame) -> pd.DataFrame:
    orders_df = orders_df.copy()
    returns_df = returns_df.copy()
    if not np.issubdtype(getattr(orders_df.get("date_ship"), "dtype", np.dtype("O")), np.datetime64):
        orders_df["date_ship"] = pd.to_datetime(orders_df.get("date_ship"), errors="coerce")
    if not np.issubdtype(getattr(returns_df.get("date_return"), "dtype", np.dtype("O")), np.datetime64):
        returns_df["date_return"] = pd.to_datetime(returns_df.get("date_return"), errors="coerce")
    if "sku" in orders_df.columns:
        orders_df["sku"] = normalize_sku(orders_df["sku"])
    if "sku" in returns_df.columns:
        returns_df["sku"] = normalize_sku(returns_df["sku"])
    ship = orders_df.dropna(subset=["date_ship"]).copy()
    ship["period"] = ship["date_ship"].dt.to_period("M")
    ship_g = ship.groupby(["period","sku"], as_index=False).agg(
        shipped_qty=("qty_shipped","sum"),
        promo_rub=("discount_rub","sum"),
        order_value_rub_sum=("order_value_rub","sum"),
    )
    ret = returns_df.dropna(subset=["date_return"]).copy()
    ret["period"] = ret["date_return"].dt.to_period("M")
    ret_g = ret.groupby(["period","sku"], as_index=False).agg(
        returns_qty=("returns_qty","sum"),
        returns_rub=("returns_rub","sum"),
    )
    monthly = pd.merge(ship_g, ret_g, on=["period","sku"], how="outer").fillna(0)
    # Приведём period к строке YYYY-MM для совместимости с BI
    monthly["period"] = monthly["period"].astype(str)
    int_cols = ["shipped_qty","returns_qty"]
    for c in int_cols:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").fillna(0).astype(int)
    float_cols = ["promo_rub","order_value_rub_sum","returns_rub"]
    for c in float_cols:
        monthly[c] = pd.to_numeric(monthly[c], errors="coerce").fillna(0.0)
    monthly = monthly.sort_values(["period","sku"]).reset_index(drop=True)
    monthly["period"] = monthly["period"].astype(str)
    return monthly

def build_mart_unit(analytics_df: pd.DataFrame, *, cogs_mode: str = "NET") -> pd.DataFrame:
    # Fallback COGS computation if not provided by analytics
    df = analytics_df.copy()
    if "cogs" not in df.columns:
        # Prefer explicit production_cost_total if present
        if {"production_cost_total"}.issubset(df.columns):
            df["cogs"] = pd.to_numeric(df["production_cost_total"], errors="coerce")
        else:
            # Compute from per-unit cost and quantity (NET or GROSS)
            qty_col = "net_qty" if cogs_mode.upper() == "NET" and "net_qty" in df.columns else (
                "total_qty" if "total_qty" in df.columns else None
            )
            if qty_col and "production_cost" in df.columns:
                df["cogs"] = pd.to_numeric(df["production_cost"], errors="coerce") * pd.to_numeric(df[qty_col], errors="coerce")
    analytics_df = df
    # Плоский экспорт основных полей для BI
    keep = [
        "sku","total_qty","total_rev","total_fee","total_payout",
        "returns_qty","returns_rub","promo_cost",
        "production_cost","production_cost_total","comb_cost",
        "net_qty","net_revenue","margin",
        "avg_price_per_unit","avg_net_price_per_unit",
        "commission_per_unit","promo_per_unit","margin_per_unit",
        "contribution_margin","break_even_price",
        "margin_pct","returns_pct","promo_intensity_pct",
        "avg_order_value","conversion_order_return","recommended_action",
        "ordered_qty_promo_sum","unique_visitors_total_sum","unique_visitors_card_sum",
        "ordered_amount_promo_sum","ordered_amount_sum","card_to_cart_conversion_mean",
        "accrual_amount_sum","accrual_qty_sum","accrual_cost_per_unit","margin_adj",
        "opening_stock","incoming","incoming_from_supplies","outgoing","outgoing_shipped",
        "ending_stock","valid_stock_end","invalid_stock_end","average_inventory",
        "cogs","inventory_turnover","ABC_class","XYZ_class"
    ]
    exist = [c for c in keep if c in analytics_df.columns]
    mart = analytics_df[exist].copy()
    return mart

def build_data_dictionary(fact_daily: pd.DataFrame, fact_monthly: pd.DataFrame, mart: pd.DataFrame) -> pd.DataFrame:
    desc_daily = {
        "date": "Дата (shipment/return)",
        "sku": "Идентификатор SKU",
        "shipped_qty": "Отгружено, шт.",
        "promo_rub": "Промо/скидки по заказам, ₽",
        "order_value_rub_sum": "Сумма заказов из csv, ₽ (если доступно)",
        "shipments": "Число отправлений",
        "returns_qty": "Возвраты, шт.",
        "returns_rub": "Возвраты, ₽"
    }
    desc_monthly = {
        "period": "Период YYYY-MM",
        "sku": "Идентификатор SKU",
        "shipped_qty": "Отгружено, шт.",
        "promo_rub": "Промо/скидки, ₽",
        "order_value_rub_sum": "Сумма заказов, ₽",
        "returns_qty": "Возвраты, шт.",
        "returns_rub": "Возвраты, ₽"
    }
    desc_mart = {
        "sku":"Идентификатор SKU",
        "total_qty":"Продано всего, шт.",
        "total_rev":"Валовая выручка, ₽",
        "total_fee":"Комиссия Ozon, ₽",
        "total_payout":"К выплате, ₽",
        "returns_qty":"Возвраты, шт.",
        "returns_rub":"Возвраты, ₽",
        "promo_cost":"Расходы на промо, ₽",
        "production_cost":"Себестоимость за ед., ₽",
        "production_cost_total":"Итого себестоимость, ₽",
        "comb_cost":"Совокупные затраты, ₽",
        "net_qty":"Чистое кол-во, шт.",
        "net_revenue":"Чистая выручка, ₽",
        "margin":"Маржа, ₽",
        "avg_price_per_unit":"Средняя цена, ₽/ед.",
        "avg_net_price_per_unit":"Средняя чистая цена, ₽/ед.",
        "commission_per_unit":"Комиссия за ед., ₽",
        "promo_per_unit":"Промо за ед., ₽",
        "margin_per_unit":"Маржа за ед., ₽",
        "contribution_margin":"Contribution margin (unit/price)",
        "break_even_price":"Точка безубыточности, ₽/ед.",
        "margin_pct":"Маржа, % от чистой выручки",
        "returns_pct":"Доля возвратов, %",
        "promo_intensity_pct":"Интенсивность промо, % от выручки",
        "avg_order_value":"Средний чек, ₽",
        "conversion_order_return":"Конверсия заказ→возврат, %",
        "recommended_action":"SCALE/FIX/STOP",
        "ordered_qty_promo_sum":"Заказано по акциям, шт.",
        "unique_visitors_total_sum":"Уникальные посетители, всего",
        "unique_visitors_card_sum":"Уникальные посетители карточки",
        "ordered_amount_promo_sum":"Сумма заказов по акциям, ₽",
        "ordered_amount_sum":"Сумма заказов, ₽",
        "card_to_cart_conversion_mean":"Конверсия карточка→корзина, %",
        "accrual_amount_sum":"Сумма начислений, ₽",
        "accrual_qty_sum":"Кол-во начислений",
        "accrual_cost_per_unit":"Ср. начисление за ед., ₽",
        "margin_adj":"Маржа с учётом начислений, ₽",
        "opening_stock":"Остаток на начало, шт.",
        "incoming":"Приход, шт.",
        "incoming_from_supplies":"Приход из поставок, шт.",
        "outgoing":"Расход, шт.",
        "outgoing_shipped":"Расход отгруженный, шт.",
        "ending_stock":"Остаток на конец, шт.",
        "valid_stock_end":"Валидный остаток, шт.",
        "invalid_stock_end":"Невалидный остаток, шт.",
        "average_inventory":"Средний запас, шт.",
        "cogs":"Себестоимость проданных, ₽",
        "inventory_turnover":"Оборачиваемость запасов",
        "ABC_class":"ABC‑класс по выручке",
        "XYZ_class":"XYZ‑класс по стабильности"
    }
    def build_dd(table_name, df, desc):
        rows = []
        for c in df.columns:
            # получаем dtype безопасно
            col_dtype = df[c].dtypes
            if hasattr(col_dtype, "__iter__") and not isinstance(col_dtype, str):
                # несколько колонок с одинаковым именем — берём первый тип
                col_dtype = col_dtype[0]
            rows.append({
                "table": table_name,
                "column": c,
                "dtype": str(col_dtype),
                "description": desc.get(c, "")
            })
        return rows
    rows = []
    rows += build_dd("fact_sku_daily", fact_daily, desc_daily)
    rows += build_dd("fact_sku_monthly", fact_monthly, desc_monthly)
    rows += build_dd("mart_unit_econ", mart, desc_mart)
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Сборка GOLD-слоя CSV для дашбордов из выгрузок Ozon")
    ap.add_argument("--orders", required=True, help="CSV '25 заказы ...csv'")
    ap.add_argument("--sales", required=True, help="XLSX 'Отчет о реализации ...xlsx'")
    ap.add_argument("--returns", required=True, help="XLSX '... возвраты ...xlsx'")
    ap.add_argument("--costs", required=True, help="XLSX 'production_costs.xlsx'")
    ap.add_argument("--promo-actions", default=None, help="XLSX '2 аналитика акции.xlsx' (опц.)")
    ap.add_argument("--accruals", default=None, help="XLSX отчёт по начислениям (опц.)")
    ap.add_argument("--inventory", default=None, help="XLSX продажи/остатки по складам (опц.)")
    ap.add_argument("--output-dir", required=True, help="Папка для CSV")
    ap.add_argument("--return-window-days", type=int, default=90, help="Окно учёта возвратов при аналитике")
    ap.add_argument("--cogs-mode", choices=["NET", "GROSS"], default="NET", help="Способ расчёта COGS, если не пришёл из analytics")
    args = ap.parse_args()


    # Robust loading with fallback to universal loader
    try:
        orders_df = load_orders(Path(args.orders))
    except Exception:
        orders_df = load_any(Path(args.orders))

    try:
        sales_df = load_sales(Path(args.sales))
    except Exception:
        sales_df = load_any(Path(args.sales))

    try:
        returns_df = load_returns(Path(args.returns))
    except Exception:
        returns_df = load_any(Path(args.returns))

    try:
        costs_df = load_costs(Path(args.costs))
    except Exception:
        costs_df = load_any(Path(args.costs))

    promos_df = compute_promos(orders_df)

    try:
        promo_df = load_promo_actions(Path(args.promo_actions)) if args.promo_actions else pd.DataFrame()
    except Exception:
        promo_df = load_any(Path(args.promo_actions)) if args.promo_actions else pd.DataFrame()

    try:
        accrual_df = load_accruals(Path(args.accruals)) if args.accruals else pd.DataFrame()
    except Exception:
        accrual_df = load_any(Path(args.accruals)) if args.accruals else pd.DataFrame()

    try:
        inventory_df = load_inventory(Path(args.inventory)) if args.inventory else pd.DataFrame()
    except Exception:
        inventory_df = load_any(Path(args.inventory)) if args.inventory else pd.DataFrame()

    analytics_df = compute_analytics(
        sales_df=sales_df,
        returns_df=returns_df,
        promos_df=promos_df,
        costs_df=costs_df,
        orders_df=orders_df,
        return_window_days=int(args.return_window_days),
        promo_df=promo_df,
        accrual_df=accrual_df,
        inventory_df=inventory_df,
    )

    fact_daily = build_daily_fact(orders_df, returns_df)
    fact_monthly = build_monthly_fact(orders_df, returns_df)
    mart = build_mart_unit(analytics_df, cogs_mode=args.cogs_mode)
    dd = build_data_dictionary(fact_daily, fact_monthly, mart)

    out = Path(args.output_dir)
    _safe_to_csv(fact_daily, out / "fact_sku_daily.csv")
    _safe_to_csv(fact_monthly, out / "fact_sku_monthly.csv")
    _safe_to_csv(mart, out / "mart_unit_econ.csv")
    _safe_to_csv(dd, out / "data_dictionary.csv")

    print("✔ GOLD построен в:", out.resolve())
    for f in ["fact_sku_daily.csv","fact_sku_monthly.csv","mart_unit_econ.csv","data_dictionary.csv"]:
        print(" -", (out / f).resolve())

if __name__ == "__main__":
    main()
