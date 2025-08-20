import os, sys as _sys_boot

# применяем быстрые пресеты до инициализации виджетов Streamlit
import streamlit as st
_pending = st.session_state.pop("date_range_pending", None)
if _pending:
    st.session_state["date_from"], st.session_state["date_to"] = _pending
_APP_DIR_BOOT = os.path.dirname(__file__)
if _APP_DIR_BOOT not in _sys_boot.path:
    _sys_boot.path.insert(0, _APP_DIR_BOOT)
import sys
from pathlib import Path

# Абсолютные пути
APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR.parent
ROOT_DIR = SCRIPTS_DIR.parent

for p in (APP_DIR, SCRIPTS_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


import io
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# Унифицированный рендер Plotly с русской локалью
def st_plot(fig):
    try:
        import streamlit as _st_local
        _st_local.plotly_chart(
            fig,
            use_container_width=True,
            config={"locale": "ru", "displayModeBar": False},
        )
    except Exception:
        # Фоллбек без конфига
        st.plotly_chart(fig, use_container_width=True)

# ---- Monte Carlo module (safe, cached import) ----
import importlib

@st.cache_resource
def _try_import_mc():
    try:
        return importlib.import_module("monte_carlo")
    except Exception as e:
        st.session_state["mc_import_error"] = f"{e}"
        return None



mc = _try_import_mc()

# ---- MC shims ----
from dataclasses import dataclass

@dataclass
class Assumptions:
    price_drift_pp: float = 0.0          # +доля к цене (доля, не %)
    promo_delta_pp: float = 0.0          # +п.п. от цены -> доля
    commission_delta_pp: float = 0.0     # +п.п. от цены -> доля
    returns_delta_pp: float = 0.0        # +п.п. к доле возвратов (в П.П., не в долях)

def _apply_assumptions(base_price, base_prod, base_comm, base_promo, base_rr, ass: Assumptions | None):
    """Скорректировать базовые параметры с учётом допущений."""
    if ass is None:
        return base_price, base_prod, base_comm, base_promo, base_rr
    price = float(base_price) * (1.0 + float(ass.price_drift_pp))
    comm  = float(base_comm)  + price * float(ass.commission_delta_pp)
    promo = float(base_promo) + price * float(ass.promo_delta_pp)
    rr = float(base_rr) + float(ass.returns_delta_pp) / 100.0
    rr = max(0.0, min(1.0, rr))
    return price, float(base_prod), comm, promo, rr

# ---- MC adapters (use new MonteCarloSimulator API) ----
def _mc_unit_margin(analytics_df: pd.DataFrame, sku: str, cfg, assumptions) -> dict:
    if mc is None or not hasattr(mc, "MonteCarloSimulator"):
        raise RuntimeError("MonteCarloSimulator не найден в модуле monte_carlo")

    row = analytics_df.loc[analytics_df["sku"].astype(str) == str(sku)]
    if row.empty:
        raise ValueError(f"SKU '{sku}' не найден в analytics")
    r0 = row.iloc[0]

    base_price = float(r0.get("avg_net_price_per_unit", r0.get("avg_price_per_unit", 0.0)))
    base_prod  = float(r0.get("production_cost_per_unit", 0.0))
    base_comm  = float(r0.get("commission_per_unit", 0.0))
    base_promo = float(r0.get("promo_per_unit", 0.0))
    base_rr    = float(r0.get("returns_pct", 0.0)) / 100.0
    qty        = [float(r0.get("total_qty", r0.get("shipped_qty", 0.0)) or 0.0)]

    # применяем допущения (шим)
    adj_price, adj_prod, adj_comm, adj_promo, adj_rr = _apply_assumptions(
        base_price, base_prod, base_comm, base_promo, base_rr, assumptions
    )

    sim = mc.MonteCarloSimulator(n_sims=int(cfg.n_sims), random_state=int(cfg.seed))
    res = sim.simulate_sku(
        base_price=adj_price,
        base_production_cost=adj_prod,
        base_commission_per_unit=adj_comm,
        base_promo_per_unit=adj_promo,
        base_returns_rate=adj_rr,
        qty=qty,
    )

    samples = getattr(res, "samples_unit_margin", None)
    if samples is None:
        samples = np.asarray(getattr(res, "unit_margin_samples", []), dtype=float)
    samples = np.asarray(samples, dtype=float)
    p05 = float(np.quantile(samples, 0.05)) if samples.size else 0.0
    p50 = float(np.quantile(samples, 0.50)) if samples.size else 0.0
    p95 = float(np.quantile(samples, 0.95)) if samples.size else 0.0
    prob_negative = float((samples < 0).mean()) if samples.size else 0.0
    return {"samples": samples, "p05": p05, "p50": p50, "p95": p95, "prob_negative": prob_negative}

def _mc_portfolio_margin(analytics_df: pd.DataFrame, qty_map: Dict[str, float], cfg) -> dict:
    """
    Адаптер под старый интерфейс mc.simulate_portfolio_margin(...):
    Складывает выборки total_margin по SKU (поэлементно) и возвращает dict со сэмплами и квантилями.
    """
    if mc is None or not hasattr(mc, "MonteCarloSimulator"):
        raise RuntimeError("MonteCarloSimulator не найден в модуле monte_carlo")
    n_sims = int(cfg.n_sims)
    rng_seed = int(cfg.seed)
    total_samples = None

    for i, (sku, qty) in enumerate(qty_map.items()):
        row = analytics_df.loc[analytics_df["sku"].astype(str) == str(sku)]
        if row.empty:
            continue
        r0 = row.iloc[0]
        base_price = float(r0.get("avg_net_price_per_unit", r0.get("avg_price_per_unit", 0.0)))
        base_prod  = float(r0.get("production_cost_per_unit", 0.0))
        base_comm  = float(r0.get("commission_per_unit", 0.0))
        base_promo = float(r0.get("promo_per_unit", 0.0))
        base_rr    = float(r0.get("returns_pct", 0.0)) / 100.0

        sim = mc.MonteCarloSimulator(n_sims=n_sims, random_state=rng_seed + i)
        res = sim.simulate_sku(
            base_price=base_price,
            base_production_cost=base_prod,
            base_commission_per_unit=base_comm,
            base_promo_per_unit=base_promo,
            base_returns_rate=base_rr,
            qty=[float(qty)],
        )
        samples = np.asarray(getattr(res, "samples_total_margin", []), dtype=float)
        if samples.size == 0:
            # fallback: перемножить unit_margin_samples * qty
            um = np.asarray(getattr(res, "samples_unit_margin", []), dtype=float)
            samples = um * float(qty) if um.size else np.zeros(n_sims, dtype=float)
        total_samples = samples if total_samples is None else (total_samples + samples)

    if total_samples is None:
        total_samples = np.zeros(n_sims, dtype=float)
    p05 = float(np.quantile(total_samples, 0.05))
    mean = float(np.mean(total_samples))
    p95 = float(np.quantile(total_samples, 0.95))
    return {"samples": total_samples, "p05": p05, "mean": mean, "p95": p95}

# ---- Unified config access (secrets/env) ----
def _cfg(key: str, default=None):
    """
    Возвращает значение конфига по ключу.
    Приоритет: Streamlit secrets -> переменные окружения -> default.
    """
    try:
        # st уже импортирован выше
        val = st.secrets.get(key, None)  # type: ignore[attr-defined]
        if val is not None:
            return val
    except Exception:
        pass
    import os
    return os.environ.get(key, default)

COGS_MODE = (_cfg("COGS_MODE", "NET") or "NET").upper()
RETURNS_ALERT_PCT = float(_cfg("RETURNS_ALERT_PCT", "5"))

# --- Granularity helpers & aggregation ---
def _has(df, cols):
    return (df is not None) and (not df.empty) and set(cols).issubset(df.columns)

@st.cache_data(ttl=600, show_spinner=False)
def aggregate_from_daily(daily: pd.DataFrame) -> dict:
    if daily is None or daily.empty or "date" not in daily.columns:
        return {"daily": pd.DataFrame(), "weekly": pd.DataFrame(), "monthly": pd.DataFrame()}

    d = daily.copy()
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    num_cols = [c for c in ["order_value_rub_sum","returns_rub","promo_rub",
                            "shipped_qty","returns_qty","shipments","cogs","margin"]
                if c in d.columns]
    for c in num_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(0)

    # DAY
    daily_agg = (d.groupby("date", as_index=False)
                   .agg({c: "sum" for c in num_cols})
                   .sort_values("date"))

    # WEEK: ISO-с понедельника
    d["week_start"] = d["date"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    weekly_agg = (d.groupby("week_start", as_index=False)
                    .agg({c: "sum" for c in num_cols})
                    .rename(columns={"week_start": "period"})
                    .sort_values("period"))

    # MONTH: строго из daily, чтобы не копить погрешность
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    monthly_agg = (d.groupby("month", as_index=False)
                     .agg({c: "sum" for c in num_cols})
                     .rename(columns={"month": "period"})
                     .sort_values("period"))

    return {"daily": daily_agg, "weekly": weekly_agg, "monthly": monthly_agg}

# --- Caption helpers ---

def render_caption(title: str, bullets: list[str], note: str | None = None):
    """Единый шаблон подписи под графиком."""
    lines = [f"**{title}**", ""]
    if bullets:
        for b in bullets:
            lines.append(f"- {b}")
    if note:
        lines += ["", note]
    import streamlit as st  # локальный импорт на всякий случай
    st.markdown("\n".join(lines))

def trend_summary(ts: pd.DataFrame, date_col: str, value_col: str, sma_window: int = 7) -> str:
    """Короткая авто-интерпретация тренда: рост/падение/плато, провалы, затухание до нуля."""
    if ts is None or ts.empty or not {date_col, value_col}.issubset(ts.columns):
        return "Данные для интерпретации отсутствуют."
    s = ts.sort_values(date_col)[value_col].astype(float).fillna(0.0)
    if len(s) < max(8, sma_window + 1):
        return "Недостаточно точек для устойчивого вывода."
    # базовая динамика: сравнение среднего последних k против предыдущих k
    k = min(14, max(7, len(s)//6))
    tail_mean = s.tail(k).mean()
    prev_mean = s.iloc[-2*k:-k].mean() if len(s) >= 2*k else s.head(max(3, len(s)//3)).mean()
    delta = tail_mean - prev_mean
    pct = (delta / prev_mean * 100) if prev_mean else 0.0

    # детекция «затухания до нуля»
    zero_streak = int((s.tail(min(60, len(s))) == 0).astype(int).groupby((s != 0).astype(int).cumsum()).cumcount().max() or 0)

    if zero_streak >= 7:
        return f"Наблюдается длительная серия нулевых продаж (≈{zero_streak} дней). Требуется проверка остатков/статуса карточек."
    if pct > 10:
        return f"Тренд положительный: средний уровень последних недель выше на {pct:.1f}%."
    if pct < -10:
        return f"Тренд отрицательный: средний уровень последних недель ниже на {abs(pct):.1f}%."
    return "Существенных изменений тренда не выявлено (колебания в пределах нормы)."


def _summarize_series(df: pd.DataFrame) -> dict:
    """Краткая сводка по серии: период и сумма выручки."""
    if df is None or df.empty or "period" not in df.columns or "order_value_rub_sum" not in df.columns:
        return {"period": None, "order_value_rub_sum": 0.0}
    period = df["period"].min(), df["period"].max()
    val = float(df["order_value_rub_sum"].sum())
    return {"period": period, "order_value_rub_sum": val}

# --- KPI/Finance/Fan helpers ---
def _kpis_finance_blocks(ana: pd.DataFrame, daily_f: pd.DataFrame) -> dict:
    """Вычисляет фин. KPI: gross/net/margin, AOV, ROMI/ROI (если есть ad_spend)."""
    out = {}
    # Gross/Net/Margin
    gross = float(daily_f.get("order_value_rub_sum", pd.Series(dtype=float)).sum()) if (daily_f is not None and not daily_f.empty) else float(ana.get("total_rev", pd.Series(dtype=float)).sum())
    net   = float(ana.get("net_revenue", pd.Series(dtype=float)).sum())
    margin = float(ana.get("margin", pd.Series(dtype=float)).sum())
    out["gross"] = gross
    out["net"] = net
    out["margin"] = margin
    out["margin_pct"] = (margin / net * 100.0) if net else 0.0

    # AOV = Net / число заказов, если есть orders_cnt, иначе Net / отгружено шт. как приближение
    orders_cnt = None
    for cand in ("orders_cnt", "orders", "orders_n"):
        if cand in ana.columns:
            try:
                orders_cnt = float(pd.to_numeric(ana[cand], errors="coerce").fillna(0).sum())
                break
            except Exception:
                pass
    if orders_cnt is None:
        # пробуем qty как прокси
        qty = None
        for cand in ("total_qty", "shipped_qty", "qty"):
            if cand in ana.columns:
                qty = float(pd.to_numeric(ana[cand], errors="coerce").fillna(0).sum())
                break
        orders_cnt = qty if qty and qty > 0 else None
    out["aov"] = (net / orders_cnt) if orders_cnt and orders_cnt > 0 else 0.0

    # ROMI/ROI, если есть рекламные траты
    ad_spend = 0.0
    for cand in ("ad_spend", "ads_spend", "marketing_spend", "advertising_cost"):
        if cand in ana.columns:
            ad_spend = float(pd.to_numeric(ana[cand], errors="coerce").fillna(0).sum()); break
    if ad_spend > 0:
        # ROMI = (incremental_margin / ad_spend). Здесь используем общий margin как приближение.
        out["romi"] = (margin / ad_spend)
        # ROI (руб./руб.) = (net - ad_spend) / ad_spend
        out["roi"] = ((net - ad_spend) / ad_spend)
    else:
        out["romi"] = None
        out["roi"] = None

    return out

def _fan_forecast_net(daily_df: pd.DataFrame, weeks_ahead: int = 8) -> pd.DataFrame:
    """
    Простой fan-chart прогноза NET по неделям: p10/p50/p90 на горизонте 4–8 недель.
    Метод: сумма NET по ISO-неделям; среднее/стд последних 12 недель; прогноз = N(μ, σ) i.i.d.
    """
    if daily_df is None or daily_df.empty:
        return pd.DataFrame(columns=["week", "p10", "p50", "p90"])
    d = daily_df.copy()
    if "date" not in d.columns:
        return pd.DataFrame(columns=["week", "p10", "p50", "p90"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    # Оценка нетто: order_value_rub_sum - returns_rub - promo_rub (если есть)
    rev = pd.to_numeric(d.get("order_value_rub_sum", 0), errors="coerce").fillna(0)
    ret = pd.to_numeric(d.get("returns_rub", 0), errors="coerce").fillna(0)
    promo = pd.to_numeric(d.get("promo_rub", 0), errors="coerce").fillna(0)
    d["net"] = rev - ret - promo
    d["week"] = d["date"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    w = d.groupby("week", as_index=False)["net"].sum().sort_values("week")
    if len(w) < 4:
        return pd.DataFrame(columns=["week", "p10", "p50", "p90"])

    tail = w.tail(min(12, len(w)))["net"].astype(float)
    mu = float(tail.mean())
    sigma = float(tail.std(ddof=0))
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = abs(mu) * 0.15  # мягкое допущение

    # Строим горизонты
    last_week = w["week"].max()
    weeks = [last_week + pd.Timedelta(weeks=i) for i in range(1, weeks_ahead + 1)]
    # Квантили нормального приближения
    from math import erf, sqrt
    # вспомогательная функция квантилей нормального: инверсию не пишем, используем коэффициенты p10/p50/p90
    z = {"p10": -1.28155, "p50": 0.0, "p90": 1.28155}
    rows = []
    for wk in weeks:
        rows.append({
            "week": wk,
            "p10": mu + z["p10"] * sigma,
            "p50": mu + z["p50"] * sigma,
            "p90": mu + z["p90"] * sigma,
        })
    return pd.DataFrame(rows)

# --- RU headers helper ---
RENAME_MAP_RU = {
    "sku": "SKU",
    "category": "Категория",
    "total_rev": "Выручка, ₽",
    "net_revenue": "Чистая выручка, ₽",
    "margin": "Маржа, ₽",
    "returns_pct": "Возвраты, %",
    "returns_qty": "Возвраты, шт.",
    "returns_rub": "Возвраты, ₽",
    "promo_intensity_pct": "Промо, %",
    "promo_cost": "Промо, ₽",
    "avg_price_per_unit": "Цена (вал.), ₽/ед.",
    "avg_net_price_per_unit": "Цена (нетто), ₽/ед.",
    "production_cost_per_unit": "Себестоимость, ₽/ед.",
    "commission_per_unit": "Комиссия, ₽/ед.",
    "promo_per_unit": "Промо, ₽/ед.",
    "margin_per_unit": "Маржа/ед., ₽",
    "break_even_price": "Точка безуб., ₽/ед.",
    "contribution_margin": "Вклад маржи",
    "margin_pct": "Маржа, %",
    "shipped_qty": "Отгрузки, шт.",
    "shipments": "Доставки, шт.",
    "period": "Период",
    "date": "Дата",
    "cogs": "COGS, ₽",
    "commission_total": "Комиссия, ₽",
    "forecast_qty": "Прогноз, шт.",
    "ending_stock": "Остаток на конец, шт.",
    "average_inventory": "Средний запас, шт.",
    "inventory_turnover": "Оборачиваемость",
    "opening_stock": "Остаток на начало, шт.",
    "incoming": "Поступления, шт.",
    "outgoing": "Списания/продажи, шт.",
}

def df_ru(df: pd.DataFrame) -> pd.DataFrame:
    """Переименовывает знакомые тех. колонки в русские заголовки."""
    try:
        return df.rename(columns={k: v for k, v in RENAME_MAP_RU.items() if k in df.columns})
    except Exception:
        return df

def show_table_ru(df: pd.DataFrame, title: str | None = None, use_container_width: bool = True):
    """Отображает таблицу с русскими заголовками и форматированием ₽ / % / шт."""
    if df is None or df.empty:
        if title:
            st.markdown(f"#### {title}")
        st.info("Нет данных для отображения.")
        return

    df2 = df.copy()

    # Определяем типовые колонки по названию
    money_like = [c for c in df2.columns if any(k in c for k in
                   ["rev", "revenue", "margin", "cogs", "promo", "returns_rub", "commission", "price", "cost"])]
    pct_like   = [c for c in df2.columns if c.endswith("_pct") or c in
                   {"margin_pct", "returns_pct", "promo_intensity_pct"}]
    qty_like   = [c for c in df2.columns if any(k in c for k in
                   ["qty", "shipments", "stock", "inventory"])]

    # Форматирование
    for c in money_like:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)\
                        .map(lambda x: f"{x:,.0f} ₽".replace(",", " "))
    for c in pct_like:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0)\
                        .map(lambda x: f"{x:.1f}%")
    for c in qty_like:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce").fillna(0).astype(int)\
                        .map(lambda x: f"{x:,}".replace(",", " "))

    # Русские заголовки
    df2 = df_ru(df2)

    if title:
        st.markdown(f"#### {title}")
    st.dataframe(df2, use_container_width=use_container_width)


# --- Pricing & Promo helper: ensure columns ---
def _ensure_pricing_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Гарантируем минимальный набор колонок для Pricing & Promo Lab."""
    missing: list[str] = []
    a = df.copy()

    # qty
    if "total_qty" not in a.columns:
        for alias in ["net_qty", "shipped_qty", "qty"]:
            if alias in a.columns:
                a["total_qty"] = pd.to_numeric(a[alias], errors="coerce")
                break
    if "total_qty" not in a.columns:
        missing.append("total_qty")

    # avg_net_price_per_unit
    if "avg_net_price_per_unit" not in a.columns:
        if {"net_revenue","total_qty"}.issubset(a.columns):
            a["avg_net_price_per_unit"] = (
                pd.to_numeric(a["net_revenue"], errors="coerce").fillna(0) /
                pd.to_numeric(a["total_qty"], errors="coerce").replace(0, np.nan)
            ).fillna(0)
        elif {"total_rev","total_qty"}.issubset(a.columns):
            a["avg_net_price_per_unit"] = (
                pd.to_numeric(a["total_rev"], errors="coerce").fillna(0) /
                pd.to_numeric(a["total_qty"], errors="coerce").replace(0, np.nan)
            ).fillna(0)
        else:
            missing.append("avg_net_price_per_unit")

    # production_cost_per_unit
    if "production_cost_per_unit" not in a.columns:
        if {"cogs","total_qty"}.issubset(a.columns):
            a["production_cost_per_unit"] = (
                pd.to_numeric(a["cogs"], errors="coerce").fillna(0) /
                pd.to_numeric(a["total_qty"], errors="coerce").replace(0, np.nan)
            ).fillna(0)
        elif "production_cost" in a.columns:
            a["production_cost_per_unit"] = pd.to_numeric(a["production_cost"], errors="coerce").fillna(0)
        else:
            missing.append("production_cost_per_unit")

    # commission_per_unit
    if "commission_per_unit" not in a.columns:
        if {"total_fee","total_qty"}.issubset(a.columns):
            a["commission_per_unit"] = (
                pd.to_numeric(a["total_fee"], errors="coerce").fillna(0) /
                pd.to_numeric(a["total_qty"], errors="coerce").replace(0, np.nan)
            ).fillna(0)
        else:
            missing.append("commission_per_unit")

    # promo_per_unit
    if "promo_per_unit" not in a.columns:
        if {"promo_cost","total_qty"}.issubset(a.columns):
            a["promo_per_unit"] = (
                pd.to_numeric(a["promo_cost"], errors="coerce").fillna(0) /
                pd.to_numeric(a["total_qty"], errors="coerce").replace(0, np.nan)
            ).fillna(0)
        else:
            missing.append("promo_per_unit")

    return a, missing


# Локальные компоненты (простые KPI и графики на Plotly)
try:
    from kpis import kpi_row
except Exception:
    # Fallback: простая реализация KPI-строки, если нет модуля kpis.py
    def kpi_row(items):
        items = list(items)
        cols = st.columns(len(items))
        for col, item in zip(cols, items):
            if isinstance(item, dict):
                title = item.get("title", "")
                value = item.get("value", "")
            else:
                title, value = item
            with col:
                st.metric(title, value)

try:
    import charts
except Exception:
    import plotly.express as px
    import plotly.graph_objects as go
    class charts:  # fallback-обёртка
        @staticmethod
        def line(df, x, y, title=None, **kwargs):
            fig = px.line(df, x=x, y=y, title=title)
            if kwargs.get("y_is_currency"):
                fig.update_traces(hovertemplate="%{y:.0f} ₽")
            elif kwargs.get("y_is_percent"):
                fig.update_traces(hovertemplate="%{y:.1f} %")
            return fig
        @staticmethod
        def bar(df, x, y, title=None, **kwargs):
            fig = px.bar(df, x=x, y=y, title=title)
            if kwargs.get("y_is_currency"):
                fig.update_traces(hovertemplate="%{y:.0f} ₽")
            elif kwargs.get("y_is_percent"):
                fig.update_traces(hovertemplate="%{y:.1f} %")
            return fig
        @staticmethod
        def scatter(df, x, y, color=None, hover_data=None, title=None, **kwargs):
            fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover_data, title=title)
            if kwargs.get("y_is_currency"):
                fig.update_traces(hovertemplate="%{y:.0f} ₽")
            elif kwargs.get("y_is_percent"):
                fig.update_traces(hovertemplate="%{y:.1f} %")
            return fig
        @staticmethod
        def heatmap_pivot(pivot, title=None):
            fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index)))
            fig.update_layout(title=title)
            return fig
        @staticmethod
        def waterfall(labels, values, title=None):
            fig = go.Figure(go.Waterfall(
                orientation="v",
                measure=["relative"] * (len(values) - 1) + ["total"],
                x=labels,
                y=values,
                connector={"line": {"color": "#888", "width": 1}},
            ))
            fig.update_layout(title=title, template="plotly_white", margin=dict(l=8, r=8, t=48, b=8))
            return fig


# Пытаться использовать ваш data_loader, но иметь фоллбек
try:
    # ожидаемый интерфейс: load_gold(dir_path) -> dict с ключами: daily, monthly, analytics
    from data_loader import load_gold  # type: ignore
except Exception:
    load_gold = None  # fallback ниже

# Планировщик (прогноз/план и допущения)
try:
    from planner import ForecastPlanner, Assumptions as PlannerAssumptions  # type: ignore
except Exception:
    ForecastPlanner = None
    PlannerAssumptions = None


# ---------- Общие настройки ----------




st.set_page_config(
    page_title="Аналитика и планирование Ozon",
    page_icon="📦",
    layout="wide",
)

# === Premium Dark: Nardo Grey → Chocolate ===
import plotly.io as pio
import plotly.graph_objects as go

st.markdown("""
<style>
:root{
  /* базовая палитра */
  --bg-0:#1e2124;                 /* самый тёмный */
  --bg-1:#25282c;                 /* canvas */
  --bg-2:#2c3035;                 /* карточки/контейнеры */
  --bg-3:#343941;                 /* ховеры, выделение */

  --ink:#e7eaf0;                  /* основной текст */
  --muted:#a8b0bd;                /* вторичный текст */

  /* nardo→chocolate */
  --nardo:#6e7072;
  --graphite:#2f3237;
  --choco:#3e2f28;

  /* акценты */
  --copper:#d4a373;               /* главный акцент (медь) */
  --copper-700:#b9855b;
  --good:#22c55e;
  --warn:#f59e0b;
  --bad:#ef4444;

  --radius:18px;
  --shadow-sm:0 8px 24px rgba(0,0,0,.25);
  --shadow-md:0 16px 38px rgba(0,0,0,.35);
}

/* фон — глубокий градиент nardo→chocolate */
html, body{
  background: radial-gradient(1200px 800px at 20% -10%, #2b2f34 0%, transparent 50%),
              radial-gradient(1200px 900px at 110% 10%, #3a2c25 0%, transparent 55%),
              linear-gradient(135deg, var(--graphite), var(--choco));
  color:var(--ink);
}

/* сайдбар — графит с тонкой рамкой */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #25282c, #202326);
  border-right:1px solid rgba(255,255,255,.06);
  box-shadow: inset -1px 0 0 rgba(0,0,0,.35);
}

/* контейнер шире */
.block-container{ max-width:1480px; padding-left:14px; padding-right:14px; }

/* типографика */
h1,h2,h3,h4{ color:var(--ink); letter-spacing:.2px }
h1{ font-weight:800 } h2{ font-weight:700 } h3,h4{ font-weight:600 }
p, label, span, div{ color:var(--ink) }

/* карточки/метрики — стеклянные с лёгким глянцем */
[data-testid="stMetric"], .stAlert, .stDataFrame, .stTable, .element-container [class*="card"]{
  background: linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.02));
  backdrop-filter: blur(6px);
  border:1px solid rgba(255,255,255,.08);
  border-radius:var(--radius);
  box-shadow:var(--shadow-sm);
}
[data-testid="stMetric"]{ padding:16px 18px; }
[data-testid="stMetric"] div{ color:var(--muted) }
[data-testid="stMetric"] [data-testid="stMetricValue"]{
  color:#fff; font-weight:800; text-shadow:0 1px 0 rgba(0,0,0,.35);
}

/* линии-групп и разделители */
hr, .st-emotion-cache-hr{ border-color: rgba(255,255,255,.08) !important; }

/* табы */
.stTabs [role="tablist"]{ border-bottom:1px solid rgba(255,255,255,.08); }
.stTabs [role="tab"]{
  color:var(--muted) !important;
}
.stTabs [role="tab"][aria-selected="true"]{
  color:var(--copper) !important;
  border-bottom:2px solid var(--copper) !important;
}

/* таблицы/гриды */
.stDataFrame thead tr th{ color:#fff; background:rgba(255,255,255,.03); }
.stDataFrame tbody tr{ background:rgba(255,255,255,.01); }
.stDataFrame tbody tr:hover{ background:rgba(255,255,255,.05); }

/* поля ввода/селекты/радио/слайдеры/дейтпикеры */
input, textarea, .stSelectbox, .stTextInput, .stDateInput, .stNumberInput{
  background:rgba(0,0,0,.2) !important;
  color:#fff !important;
  border:1px solid rgba(255,255,255,.1) !important;
  border-radius:12px !important;
}
.stSlider [role="slider"]{ border:2px solid var(--copper) !important; }
.stSlider .st-og{ background: var(--copper) !important; }
[data-baseweb="radio"] div[role="radio"][aria-checked="true"]{
  outline-color:var(--copper) !important;
  border-color:var(--copper) !important;
}

/* кнопки общего вида (графитовый градиент) */
.stButton>button{
  background: linear-gradient(180deg, #43474d, #2e3237);
  color:#fff; border:none; border-radius:14px;
  padding:.6rem 1.1rem; font-weight:700;
  box-shadow:var(--shadow-sm);
}
.stButton>button:hover{ filter:brightness(1.04); transform:translateY(-1px); }

/* акцентные кнопки (медь) — добавь class="btn-accent" при необходимости */
button[kind="primary"], .btn-accent{
  background: linear-gradient(180deg, var(--copper), var(--copper-700)) !important;
  color:#1a1a1a !important; border:none !important; border-radius:14px !important;
  box-shadow:var(--shadow-md) !important;
}
button[kind="primary"]:hover, .btn-accent:hover{ filter:brightness(1.06); }

/* чипы-пресеты периода — премиальные «таблетки» */
.periods-row .stButton>button{
  background: linear-gradient(180deg, rgba(255,255,255,.10), rgba(255,255,255,.04));
  color:#fff;
  border:1px solid rgba(255,255,255,.14);
  border-radius:999px; padding:.44rem 1.1rem;
}
.periods-row .stButton>button:hover{
  border-color:var(--copper); color:var(--copper);
  box-shadow:0 0 0 4px rgba(212,163,115,.18);
}

/* бейджи для KPI */
.badge{display:inline-block;padding:4px 10px;border-radius:999px;color:#111;font-size:12px;font-weight:800}
.badge.good{background:var(--good)} .badge.warn{background:var(--warn)} .badge.bad{background:var(--bad)}
.badge.neutral{background:var(--copper); color:#111}
</style>
""", unsafe_allow_html=True)

# Plotly – тёмный премиум-шаблон в одной палитре (медь/нейтраль/статусы)
pio.templates["nardo_choco_dark"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#262a2f",
        font=dict(
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto",
            size=13,
            color="#e7eaf0"
        ),
        margin=dict(l=8, r=8, t=56, b=8),

        # БОЛЕЕ КОНТРАСТНАЯ ПАЛИТРА ДЛЯ ТЁМНОЙ ТЕМЫ
        colorway=[
            "#d4a373",  # copper (accent)
            "#93c5fd",  # blue
            "#22c55e",  # green
            "#f59e0b",  # amber
            "#ef4444",  # red
            "#a78bfa",  # purple
            "#eab308"   # gold
        ],

        # ТЁМНЫЕ ТУЛТИПЫ
        hoverlabel=dict(
            bgcolor="rgba(34,36,40,0.9)",
            bordercolor="#d4a373",
            font=dict(color="#ffffff")
        ),

        # МЯГКАЯ СЕТКА ДЛЯ ЧИТАЕМОСТИ
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(color="#cfd5df"),
            title=dict(font=dict(color="#e7eaf0"))
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            zerolinecolor="rgba(255,255,255,0.10)",
            linecolor="rgba(255,255,255,0.15)",
            tickfont=dict(color="#cfd5df"),
            title=dict(font=dict(color="#e7eaf0"))
        ),

        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0.10)",
            borderwidth=0
        )
    )
)
pio.templates.default = "nardo_choco_dark"



# ---------- Кеши и загрузка данных ----------

@st.cache_data(show_spinner=True)
def _fallback_load_gold(dir_path: str) -> dict:
    """Фоллбек-лоадер GOLD-слоя, если нет подходящей функции в data_loader."""
    base = Path(dir_path)
    daily = pd.read_csv(base / "fact_sku_daily.csv", encoding="utf-8-sig", low_memory=False)
    monthly = pd.read_csv(base / "fact_sku_monthly.csv", encoding="utf-8-sig", low_memory=False)
    mart = pd.read_csv(base / "mart_unit_econ.csv", encoding="utf-8-sig", low_memory=False)

    # Приведение типов
    if "date" in daily.columns:
        daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    if "period" in monthly.columns:
        # период в YYYY-MM -> период pandas
        monthly["period"] = pd.PeriodIndex(pd.to_datetime(monthly["period"], errors="coerce"), freq="M")

    # Нормализация SKU к строке
    for df in (daily, monthly, mart):
        if "sku" in df.columns:
            df["sku"] = df["sku"].astype(str).str.strip()

    return {"daily": daily, "monthly": monthly, "analytics": mart}


@st.cache_data(show_spinner=True)
def load_bundle(dir_path: str | Path) -> dict:
    if load_gold is not None:
        try:
            bundle = load_gold(dir_path)  # ожидаем словарь или dataclass c .daily, .monthly, .analytics
            # Убедимся в единообразии интерфейса
            if isinstance(bundle, dict):
                daily = bundle.get("daily")
                monthly = bundle.get("monthly")
                analytics = bundle.get("analytics")
            else:
                daily = getattr(bundle, "daily", None)
                monthly = getattr(bundle, "monthly", None)
                analytics = getattr(bundle, "analytics", None)
            assert isinstance(daily, pd.DataFrame) and isinstance(monthly, pd.DataFrame) and isinstance(analytics, pd.DataFrame)
            return {"daily": daily, "monthly": monthly, "analytics": analytics}
        except Exception:
            # тихо уходим в фоллбек
            pass
    return _fallback_load_gold(dir_path)


def _format_money(x: float) -> str:
    try:
        return f"{x:,.0f} ₽".replace(",", " ")
    except Exception:
        return str(x)


def _format_pct(x: float) -> str:
    try:
        return f"{x:.1f}%"
    except Exception:
        return str(x)

# --- Badge helper ---
def _badge(text: str, kind: str = "neutral"):
    st.markdown(f'<span class="badge {kind}">{text}</span>', unsafe_allow_html=True)



# === Executive helpers: Waterfall (gross → net → margin) ===
from typing import Mapping, Optional

def _coalesce_map(d: Mapping, *keys, default=0.0) -> float:
    for k in keys:
        if k in d and d[k] is not None:
            try:
                return float(d[k])
            except Exception:
                pass
    return float(default)

def build_waterfall(analytics_like: Mapping[str, float]) -> "go.Figure":
    """Водопад: Валовая выручка → −Возвраты → −Комиссия → −Промо → −COGS → Маржа."""
    rev_gross = _coalesce_map(analytics_like, "total_rev", "gross_rev", "revenue_gross")
    returns_rub = _coalesce_map(analytics_like, "returns_rub", default=0.0)
    promo_rub   = _coalesce_map(analytics_like, "promo_cost", "promo_rub", default=0.0)
    cogs_rub    = _coalesce_map(analytics_like, "cogs", "cogs_rub", default=0.0)
    comm_rub    = _coalesce_map(analytics_like, "commission_total", "commission_rub", default=0.0)

    net = rev_gross - returns_rub - promo_rub
    margin = net - cogs_rub - comm_rub

    labels = ["Валовая выручка", "- Возвраты", "- Комиссия", "- Промо", "- COGS", "Маржа (итог)"]
    measures = ["relative", "relative", "relative", "relative", "relative", "total"]
    y = [rev_gross, -returns_rub, -comm_rub, -promo_rub, -cogs_rub, margin]

    fig = go.Figure(go.Waterfall(
        x=labels,
        measure=measures,
        y=y,
        text=[_format_money(v) for v in y],
        connector={"line": {"width": 1}}
    ))
    fig.update_traces(hovertemplate="%{y:.0f} ₽")
    fig.update_layout(title="Денежный водопад", showlegend=False, template="plotly_white", margin=dict(l=8, r=8, t=48, b=8))
    return fig
# === end executive helpers ===


# ---------- Sidebar: источники и навигация ----------

with st.sidebar:
    st.markdown("## ⚙️ Данные")
    gold_dir = st.text_input(
        "Папка GOLD (CSV)",
        value=str(ROOT_DIR / "gold"),  # исправили ROOT → ROOT_DIR
        help="Папка с fact_sku_daily.csv, fact_sku_monthly.csv, mart_unit_econ.csv",
    )
    reload_btn = st.button("🔄 Перезагрузить")

    st.markdown("---")
    page = st.radio(
        "Навигация",
        ["Обзор", "Ассортимент", "SKU детально", "Unit Economics", "ABC/XYZ", "Остатки", "Returns Lab", "Pricing & Promo", "Forecast vs Actual", "Risk (Monte Carlo)", "What-if", "About & Diagnostics"],
        index=0,
    )
    top_n = st.number_input("TOP N (для рейтингов)", min_value=5, max_value=50, value=10, step=5)

if reload_btn:
    try:
        load_bundle.clear()
    except Exception:
        pass
    try:
        st.cache_data.clear()
    except Exception:
        pass
    st.rerun()

# Проверка существования каталога GOLD до загрузки
_gdir = Path(gold_dir)
if not _gdir.exists():
    st.error(f"Папка GOLD не найдена: {_gdir}")
    st.stop()


#
# ---------- Загрузка наборов ----------
#
try:
    data = load_bundle(gold_dir)
    fact_daily: pd.DataFrame = data["daily"].copy()
    fact_monthly: pd.DataFrame = data["monthly"].copy()
    analytics: pd.DataFrame = data["analytics"].copy()
except Exception as e:
    st.error(f"Не удалось загрузить GOLD из «{gold_dir}». {e}")
    st.stop()

if analytics.empty:
    st.warning("Таблица analytics (mart_unit_econ.csv) пустая. Сформируйте GOLD через build_gold.py.")
    st.stop()

# Удобные списки/карты
sku_list = sorted(analytics["sku"].astype(str).unique().tolist())
rev_sum = float(analytics.get("total_rev", pd.Series(dtype=float)).sum())
net_rev_sum = float(analytics.get("net_revenue", pd.Series(dtype=float)).sum())
margin_sum = float(analytics.get("margin", pd.Series(dtype=float)).sum())
returns_qty_sum = float(analytics.get("returns_qty", pd.Series(dtype=float)).sum())
promo_sum = float(analytics.get("promo_cost", pd.Series(dtype=float)).sum())

# === Period presets (ЕДИНЫЙ МЕХАНИЗМ) — поставить ВЫШЕ сайдбара ===
from datetime import date

def _set_range(days_back: int | None = None, quarter: bool = False):
    today = pd.Timestamp(date.today())
    if quarter:
        m = ((today.month-1)//3)*3 + 1
        start = pd.Timestamp(year=today.year, month=m, day=1)
    elif days_back is None:  # MTD
        start = pd.Timestamp(date.today().replace(day=1))
    else:
        start = today - pd.Timedelta(days=days_back)
    st.session_state["date_range_pending"] = (start, today)
    st.rerun()

with st.container():
    st.markdown('<div class="periods-row">', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        if st.button("MTD"):
            _set_range(None)
    with c2:
        if st.button("Last 7d"):
            _set_range(6)
    with c3:
        if st.button("Last 30d"):
            _set_range(29)
    with c4:
        if st.button("Квартал"):
            _set_range(quarter=True)
    st.markdown('</div>', unsafe_allow_html=True)
# --- Sidebar filters (depend on loaded data) ---
with st.sidebar:
    st.markdown("---")
    st.markdown("## 📅 Фильтры")
    granularity = st.radio("Гранулярность", ["День","Неделя","Месяц"], index=0, horizontal=True)

    _def_from = st.session_state.get("date_from", pd.to_datetime("2025-01-01"))
    _def_to   = st.session_state.get("date_to",   pd.to_datetime("today"))

    date_from = st.date_input("С даты", value=_def_from, key="date_from")
    date_to   = st.date_input("По дату", value=_def_to,   key="date_to")

    cogs_mode = st.selectbox("COGS режим", ["NET", "GROSS"], index=(0 if COGS_MODE == "NET" else 1))
    selected_sku = st.multiselect("SKU", sku_list[:50], max_selections=50)

_daily = fact_daily.copy()
if "date" in _daily.columns:
    _daily = _daily[(pd.to_datetime(_daily["date"]) >= pd.to_datetime(date_from)) & (pd.to_datetime(_daily["date"]) <= pd.to_datetime(date_to))]

_monthly = fact_monthly.copy()
if "period" in _monthly.columns:
    # Приводим к Timestamp для сравнения
    _monthly_period_ts = pd.to_datetime(_monthly["period"].astype(str), errors="coerce")
    mask = (_monthly_period_ts >= pd.to_datetime(date_from).to_period("M").to_timestamp()) & (_monthly_period_ts <= pd.to_datetime(date_to).to_period("M").to_timestamp())
    _monthly = _monthly.loc[mask]

if selected_sku:
    _daily = _daily[_daily["sku"].astype(str).isin(selected_sku)] if not _daily.empty else _daily
    _monthly = _monthly[_monthly["sku"].astype(str).isin(selected_sku)] if not _monthly.empty else _monthly

# --- Granularity series for trends ---
try:
    aggs = aggregate_from_daily(fact_daily)
except Exception:
    aggs = {"daily": pd.DataFrame(), "weekly": pd.DataFrame(), "monthly": pd.DataFrame()}

if granularity == "День":
    series_df = aggs["daily"].rename(columns={"date": "period"})
    sma_window = 7
elif granularity == "Неделя":
    series_df = aggs["weekly"]
    sma_window = 4
else:
    series_df = aggs["monthly"]
    sma_window = 3


# ---------- Страницы ----------

def page_overview():
    st.markdown("### 📊 Обзор")
    # --- KPI (в 4 колонки в ряд) ---
    # KPI на основе фильтрованных данных, с fallback на analytics
    if not _daily.empty and {"order_value_rub_sum"}.issubset(_daily.columns):
        _rev = float(_daily["order_value_rub_sum"].sum())
    else:
        _rev = rev_sum
    _net = net_rev_sum  # при отсутствии нетто в daily оставляем из analytics
    _margin = margin_sum
    # Новые KPI-блоки (финансы)
    ana = analytics
    daily_f = _daily
    fin = _kpis_finance_blocks(ana, daily_f)
    kpi_row([
        {"title": "Валовая выручка", "value": _format_money(fin["gross"])},
        {"title": "Чистая выручка",  "value": _format_money(fin["net"])},
        {"title": "Маржа (ИТОГО)",   "value": _format_money(fin["margin"])},
        {"title": "Маржа, %",        "value": _format_pct(fin["margin_pct"])},
    ])
    # Бейдж «светофор» для маржи, %
    try:
        _kind_m = "good" if float(fin["margin_pct"]) > 15 else ("warn" if float(fin["margin_pct"]) >= 5 else "bad")
        _cols_badge_m = st.columns(4)
        with _cols_badge_m[3]:
            _badge(f"{float(fin['margin_pct']):.1f} %", _kind_m)
    except Exception:
        pass
    kpi_row([
        {"title": "AOV", "value": _format_money(fin["aov"])},
        {"title": "ROMI", "value": (f"{fin['romi']:.2f}x" if fin.get("romi") is not None else "н/д")},
        {"title": "ROI", "value": (f"{fin['roi']:.2f}x" if fin.get("roi") is not None else "н/д")},
        {"title": "SKU в риске", "value": f"{int(((ana.get('returns_pct', pd.Series(dtype=float)) > RETURNS_ALERT_PCT).sum() if 'returns_pct' in ana.columns else 0) + ((ana.get('margin', pd.Series(dtype=float)) < 0).sum() if 'margin' in ana.columns else 0))}"},
    ])

    # --- KPI (риски) ---
    # Возвраты ₽: приоритет готовой суммы; иначе оценка avg_net_price_per_unit * returns_qty
    if "returns_rub" in analytics.columns:
        _returns_rub = float(analytics["returns_rub"].sum())
    elif {"avg_net_price_per_unit", "returns_qty"}.issubset(analytics.columns):
        _returns_rub = float((analytics["avg_net_price_per_unit"] * analytics["returns_qty"]).sum())
    else:
        _returns_rub = 0.0
    _promo_rub = float(_daily.get("promo_rub", pd.Series(dtype=float)).sum()) if not _daily.empty else float(promo_sum)
    # Фактическая доля возвратов
    fact_ret_pct = (_returns_rub / _rev * 100) if _rev else 0
    # Оценка возвратов P50/P95 (процентильная оценка по выборке)
    try:
        ser = pd.to_numeric(analytics.get("returns_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        if len(ser) >= 5:
            p50_ret = float(np.percentile(ser, 50))
            p95_ret = float(np.percentile(ser, 95))
        else:
            p50_ret, p95_ret = None, None
    except Exception:
        p50_ret, p95_ret = None, None
    # Prob(GM<0) по дневной серии маржи (нормальное приближение)
    prob_neg_gm = None
    try:
        dm = daily_f.copy()
        if not dm.empty:
            # маржа по дням: net - cogs - commission
            rev_d = pd.to_numeric(dm.get("order_value_rub_sum", 0), errors="coerce").fillna(0)
            ret_d = pd.to_numeric(dm.get("returns_rub", 0), errors="coerce").fillna(0)
            promo_d = pd.to_numeric(dm.get("promo_rub", 0), errors="coerce").fillna(0)
            net_d = rev_d - ret_d - promo_d
            cogs_d = pd.to_numeric(dm.get("cogs", 0), errors="coerce").fillna(0) if "cogs" in dm.columns else 0
            comm_d = pd.to_numeric(dm.get("commission", 0), errors="coerce").fillna(0) if "commission" in dm.columns else 0
            margin_d = (net_d - cogs_d - comm_d).astype(float)
            if (margin_d > 0).any() or (margin_d < 0).any():
                mu = float(margin_d.mean()); sd = float(margin_d.std(ddof=0))
                if not np.isfinite(sd) or sd == 0:
                    prob_neg_gm = 0.0 if mu >= 0 else 1.0
                else:
                    # P(X<0) для N(mu, sd)
                    from math import erf, sqrt
                    z = (0 - mu) / sd
                    prob_neg_gm = 0.5 * (1 + erf(z / sqrt(2)))
    except Exception:
        prob_neg_gm = None
    kpi_row([
        {"title": "Возвраты, ₽", "value": _format_money(_returns_rub)},
        {"title": "Возвраты, % (факт)", "value": _format_pct(fact_ret_pct)},
        {"title": "Возвраты P50 (оценка)", "value": _format_pct(p50_ret) if p50_ret is not None else "н/д"},
        {"title": "Возвраты P95 (оценка)", "value": _format_pct(p95_ret) if p95_ret is not None else "н/д"},
    ])
    # Бейдж «светофор» для возвратов, % (факт)
    try:
        _kind_r = "good" if float(fact_ret_pct) < 5 else ("warn" if float(fact_ret_pct) < 10 else "bad")
        _cols_badge_r = st.columns(4)
        with _cols_badge_r[1]:
            _badge(f"{float(fact_ret_pct):.1f} %", _kind_r)
    except Exception:
        pass
    kpi_row([
        {"title": "Prob(GM<0)", "value": (_format_pct(100*prob_neg_gm) if prob_neg_gm is not None else "н/д")},
        {"title": "Промо, %", "value": _format_pct((_promo_rub / _rev * 100) if _rev else 0)},
        {"title": "—", "value": " "},
        {"title": "—", "value": " "},
    ])

    # --- Денежный водопад по портфелю (обзор) ---
    # Подбор доступных компонент из analytics (суммы за период/фильтр)
    gross_rev = float(analytics.get("total_rev", pd.Series(dtype=float)).sum())

    # Возвраты ₽ — готовая колонка; иначе оценка avg_net_price_per_unit * returns_qty
    if "returns_rub" in analytics.columns:
        returns_rub = float(analytics["returns_rub"].sum())
    elif {"avg_net_price_per_unit", "returns_qty"}.issubset(analytics.columns):
        returns_rub = float((analytics["avg_net_price_per_unit"] * analytics["returns_qty"]).sum())
    else:
        returns_rub = 0.0

    # Комиссия ₽ — приоритет готовой суммы; иначе комиссия/ед * qty
    if "commission_total" in analytics.columns:
        commission_rub = float(analytics["commission_total"].sum())
    elif {"commission_per_unit", "total_qty"}.issubset(analytics.columns):
        commission_rub = float((analytics["commission_per_unit"] * analytics["total_qty"]).sum())
    else:
        commission_rub = 0.0

    # Промо ₽ — приоритет promo_cost; иначе промо/ед * qty
    if "promo_cost" in analytics.columns:
        promo_rub = float(analytics["promo_cost"].sum())
    elif {"promo_per_unit", "total_qty"}.issubset(analytics.columns):
        promo_rub = float((analytics["promo_per_unit"] * analytics["total_qty"]).sum())
    else:
        promo_rub = 0.0

    # Себестоимость ₽ — приоритет готового COGS; иначе production_cost_per_unit * qty
    if "cogs" in analytics.columns:
        cogs_rub = float(analytics["cogs"].sum())
    elif {"production_cost_per_unit", "total_qty"}.issubset(analytics.columns):
        cogs_rub = float((analytics["production_cost_per_unit"] * analytics["total_qty"]).sum())
    else:
        cogs_rub = 0.0

    _totals_map = {
        "total_rev": gross_rev,
        "returns_rub": returns_rub,
        "commission_total": commission_rub,
        "promo_cost": promo_rub,
        "cogs": cogs_rub,
    }
    st_plot(build_waterfall(_totals_map))

    # --- график ниже на всю ширину ---
    show_scatter = not analytics.get("total_rev", pd.Series([])).empty and not analytics.get("margin", pd.Series([])).empty
    if show_scatter:
        fig = charts.scatter(
            analytics.rename(columns={"total_rev": "revenue", "margin": "margin"}),
            x="revenue", y="margin", color=("ABC_class" if "ABC_class" in analytics.columns else None),
            hover_data=["sku"], title="Маржа vs Выручка по SKU"
        )
        st_plot(fig)
        render_caption(
            title="Маржа vs Выручка по SKU (ABC-анализ)",
            bullets=[
                "Ось X — выручка по SKU",
                "Ось Y — маржа по SKU",
                "Цвет — ABC-класс: A — лидеры оборота, B — средние, C — хвост",
            ],
            note="Как читать: точки ниже оси X — убыточные SKU; правее — больший оборот. Приоритет на разбор — левый нижний квадрант.",
        )

    # Линия выручки + SMA (адаптивно: день/неделя/месяц)
    if _has(series_df, ["period", "order_value_rub_sum"]):
        ts = series_df[["period", "order_value_rub_sum"]].sort_values("period").copy()
        if len(ts) >= 2:
            ts["SMA"] = ts["order_value_rub_sum"].rolling(sma_window, min_periods=1).mean()
        st_plot(charts.line(ts, x="period", y=[c for c in ["order_value_rub_sum", "SMA"] if c in ts.columns],
                            title=f"Динамика выручки · {granularity}"))
        render_caption(
            title=f"Динамика выручки · {granularity}",
            bullets=[
                "Ось X — период (день/неделя/месяц)",
                "Ось Y — выручка, ₽",
                f"Синяя линия — сглаживание SMA{str(sma_window)}",
            ],
            note=trend_summary(ts.rename(columns={"period": "date"}), "date", "order_value_rub_sum", sma_window=sma_window),
        )
        # Fan chart прогноза Net (4–8 недель)
        fc = _fan_forecast_net(_daily, weeks_ahead=8)
        if not fc.empty:
            fig_fc = go.Figure()
            fig_fc.add_trace(go.Scatter(x=fc["week"], y=fc["p50"], mode="lines+markers", name="p50"))
            # заполняем веер между p10 и p90
            fig_fc.add_trace(go.Scatter(x=pd.concat([fc["week"], fc["week"][::-1]]),
                                        y=pd.concat([fc["p90"], fc["p10"][::-1]]),
                                        fill='toself', line=dict(width=0), name="p10–p90", hoverinfo="skip"))
            fig_fc.update_traces(hovertemplate="%{y:.0f} ₽")
            fig_fc.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=48, b=8), title="Прогноз Net: p50 и веер p10–p90 (недели)")
            st_plot(fig_fc)
    else:
        st.info("Нет данных для построения динамики.")
# --- Diagnostics page ---
import sys as _sys_diag, platform as _platform_diag, io as _io_diag, json as _json_diag

def page_about_diag(fact_daily, fact_monthly, analytics, forecast):
    st.subheader("About & Diagnostics")
    st.write({
        "python": _sys_diag.version.split()[0],
        "platform": _platform_diag.platform(),
        "streamlit": st.__version__,
        "pandas": pd.__version__,
        "data_source": "GOLD_BASE_URL" if st.secrets.get("GOLD_BASE_URL") else "repo gold/"
    })
    def shape(df):
        return {"rows": int(len(df)),
                "cols": int(df.shape[1]) if not df.empty else 0,
                "columns": list(df.columns[:10]) if not df.empty else []}
    st.write({
        "fact_sku_daily": shape(fact_daily),
        "fact_sku_monthly": shape(fact_monthly),
        "mart_unit_econ": shape(analytics),
        "forecast_sku_monthly": shape(forecast),
    })
    def ssum(df, col):
        return float(pd.to_numeric(df.get(col, pd.Series(dtype=float)),
                                   errors="coerce").fillna(0).sum()) if not df.empty else 0.0
    snapshot = {
        "kpi_sums": {
            "order_value_rub_sum_daily": ssum(fact_daily, "order_value_rub_sum"),
            "shipped_qty_daily": ssum(fact_daily, "shipped_qty"),
            "returns_rub_daily": ssum(fact_daily, "returns_rub"),
            "returns_qty_daily": ssum(fact_daily, "returns_qty"),
            "cogs_mart": ssum(analytics, "cogs"),
            "margin_mart": ssum(analytics, "margin"),
        }
    }
    buf = _io_diag.StringIO(); _json_diag.dump(snapshot, buf, ensure_ascii=False, indent=2)
    st.download_button("Скачать snapshot.json", buf.getvalue(),
                       file_name="snapshot.json", mime="application/json")

    # Топ-лист
    st.markdown(f"#### ТОП-{int(top_n)} прибыльных / убыточных SKU")
    if "margin" in analytics.columns:
        base_cols = ["sku", "total_rev", "margin", "returns_pct", "promo_intensity_pct"]
        cols_present = [c for c in base_cols if c in analytics.columns]
        top = analytics.sort_values("margin", ascending=False).head(int(top_n))[cols_present]
        flop = analytics.sort_values("margin", ascending=True).head(int(top_n))[cols_present]
        rename_map = {
            "sku": "SKU",
            "total_rev": "Выручка, ₽",
            "margin": "Маржа, ₽",
            "returns_pct": "Возвраты, %",
            "promo_intensity_pct": "Промо, %",
        }
        c1, c2 = st.columns(2)
        with c1:
            show_table_ru(
                top.rename(columns={k: v for k, v in rename_map.items() if k in cols_present}),
                title=f"ТОП-{int(top_n)} прибыльных SKU"
            )
        with c2:
            show_table_ru(
                flop.rename(columns={k: v for k, v in rename_map.items() if k in cols_present}),
                title=f"ТОП-{int(top_n)} убыточных SKU"
            )
    else:
        st.info("В analytics нет столбца 'margin' — сформируйте GOLD через build_gold.py.")

    # Мостик Unit Economics (водопад)
    st.markdown("#### Мостик: от валовой выручки к марже")
    # Подбор доступных компонент из analytics
    gross_rev = float(analytics.get("total_rev", pd.Series(dtype=float)).sum())
    # Возвраты ₽ — берём 'returns_rub' если есть; иначе оцениваем из avg_net_price_per_unit * returns_qty
    if "returns_rub" in analytics.columns:
        returns_rub = float(analytics["returns_rub"].sum())
    elif {"avg_net_price_per_unit", "returns_qty"}.issubset(analytics.columns):
        returns_rub = float((analytics["avg_net_price_per_unit"] * analytics["returns_qty"]).sum())
    else:
        returns_rub = 0.0
    # Комиссия ₽ — приоритет готовой суммы; иначе комиссия/ед * qty
    if "commission_total" in analytics.columns:
        commission_rub = float(analytics["commission_total"].sum())
    elif {"commission_per_unit", "total_qty"}.issubset(analytics.columns):
        commission_rub = float((analytics["commission_per_unit"] * analytics["total_qty"]).sum())
    else:
        commission_rub = 0.0
    # Промо ₽ — приоритет promo_cost; иначе промо/ед * qty
    if "promo_cost" in analytics.columns:
        promo_rub = float(analytics["promo_cost"].sum())
    elif {"promo_per_unit", "total_qty"}.issubset(analytics.columns):
        promo_rub = float((analytics["promo_per_unit"] * analytics["total_qty"]).sum())
    else:
        promo_rub = 0.0
    # Себестоимость ₽ — приоритет готового COGS; иначе production_cost_per_unit * qty
    if "cogs" in analytics.columns:
        cogs_rub = float(analytics["cogs"].sum())
    elif {"production_cost_per_unit", "total_qty"}.issubset(analytics.columns):
        cogs_rub = float((analytics["production_cost_per_unit"] * analytics["total_qty"]).sum())
    else:
        cogs_rub = 0.0

    margin_total_calc = gross_rev - returns_rub - commission_rub - promo_rub - cogs_rub
    labels = [
        "Валовая выручка",
        "- Возвраты",
        "- Комиссия",
        "- Промо",
        "- COGS",
        "Маржа (итог)",
    ]
    values = [
        gross_rev,
        -returns_rub,
        -commission_rub,
        -promo_rub,
        -cogs_rub,
        margin_total_calc,
    ]
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * (len(values) - 1) + ["total"],
        x=labels,
        y=values,
        connector={"line": {"color": "#888", "width": 1}},
    ))
    fig_wf.update_layout(
        template="plotly_white",
        margin=dict(l=8, r=8, t=48, b=8),
        title="Мостик Unit Economics",
    )
    st_plot(fig_wf)
    render_caption(
        title="Unit economics: мостик выручка → маржа",
        bullets=[
            "Столбцы — последовательные эффекты: возвраты, комиссия, промо, COGS",
            "Последний столбец — итоговая маржа",
        ],
        note="Как читать: какой фактор «съедает» больше всего маржи, туда и направляем первую оптимизацию.",
    )

# --- Returns Lab page ---
def page_returns_lab():
    st.markdown("### ♻️ Returns Lab")
    # Scatter: маржа vs возвраты
    if {"returns_pct", "margin"}.issubset(analytics.columns):
        fig_sc = px.scatter(analytics, x="returns_pct", y="margin", color=("category" if "category" in analytics.columns else None),
                            hover_data=[c for c in ["sku", "total_rev", "net_revenue"] if c in analytics.columns], title="Маржа vs Возвраты, %")
        fig_sc.update_layout(template="plotly_white")
        st_plot(fig_sc)
        render_caption(
            title="Маржа vs Возвраты",
            bullets=[
                "Ось X — возвраты, %",
                "Ось Y — маржа, ₽",
                "Цвет — категория (если присутствует)",
            ],
            note="Зона риска — высокая доля возвратов и низкая маржа; начните разбор с этих точек."
        )
    else:
        st.info("Нет необходимых колонок 'returns_pct' и 'margin' в analytics.")

    # Heatmap: возвраты по дням и SKU
    if not _daily.empty and {"date", "sku"}.issubset(_daily.columns) and "returns_qty" in _daily.columns:
        pv = (_daily.pivot_table(index="sku", columns="date", values="returns_qty", aggfunc="sum").fillna(0))
        st_plot(charts.heatmap_pivot(pv, title="Возвраты по дням и SKU"))
        render_caption(
            title="Тепловая карта возвратов",
            bullets=[
                "Ось X — даты, ось Y — SKU",
                "Оттенок — количество возвратов",
            ],
            note="Тёмные вертикальные полосы — проблемные даты/партии; сплошные тёмные строки — проблемные SKU."
        )
    else:
        st.info("Недостаточно данных для тепловой карты (нужны 'date', 'sku', 'returns_qty' в daily).")

# --- Pricing & Promo Lab page ---
def page_pricing_promo():
    st.markdown("### 💸 Pricing & Promo Lab")
    a0 = analytics.copy()
    a, miss = _ensure_pricing_cols(a0)

    # обязательные для интерфейса поля
    if "sku" not in a.columns:
        st.info("Недостаточно колонок для расчёта: sku")
        return
    # если нет интенсивности промо — считаем её нулём
    if "promo_intensity_pct" not in a.columns:
        a["promo_intensity_pct"] = 0.0

    if miss:
        st.info("Недостаточно колонок для расчёта: " + ", ".join(miss))
        return

    price_delta = st.slider("Δ Цена, %", -20, 20, 0)
    promo_delta = st.slider("Δ Промо, п.п.", -20, 20, 0)
    commission_delta = st.slider("Δ Комиссия, п.п.", -10, 10, 0)

    df = a.copy()
    df["avg_net_price_per_unit_adj"] = df["avg_net_price_per_unit"] * (1 + price_delta/100)
    df["promo_intensity_pct_adj"] = (df["promo_intensity_pct"] + promo_delta).clip(0, 100)
    df["commission_per_unit_adj"] = df["commission_per_unit"] * (1 + commission_delta/100)

    df["margin_per_unit_adj"] = (
        df["avg_net_price_per_unit_adj"]
        - df["production_cost_per_unit"]
        - df["commission_per_unit_adj"]
        - (df["avg_net_price_per_unit_adj"] * df["promo_intensity_pct_adj"]/100)
    )
    df["margin_adj"] = df["margin_per_unit_adj"] * df["total_qty"]

    st_plot(charts.bar(df.nlargest(int(top_n), "margin_adj"), x="sku", y="margin_adj", title="Маржа после изменений", orientation="v", y_is_currency=True))
    render_caption(
        title="Сценарий после изменений цены/промо/комиссии",
        bullets=[
            "Столбики — маржа по SKU после применённых дельт",
            "Изменения применяются к цене (нетто), промо и комиссии",
        ],
        note="Проверьте ТОП убыточных после изменений — возможно, их лучше исключить из промо."
    )

# --- Forecast vs Actual page ---
# --- Forecast vs Actual page ---
def page_fva():
    st.markdown("### 📈 Forecast vs Actual")
    # Пытаемся загрузить forecast_sku_monthly.csv из той же папки GOLD (не кэшируем, чтобы не ломать кэш основного лоадера)
    try:
        forecast = pd.read_csv(Path(gold_dir) / "forecast_sku_monthly.csv", encoding="utf-8-sig", low_memory=False)
    except Exception:
        forecast = pd.DataFrame()

    # --- Приводим факт к помесячному формату YYYY-MM ---
    fact = _monthly.copy()
    if fact.empty:
        st.info("Нет факта по месяцам для отображения.")
        return

    # допуски по именованию: period/date, shipped_qty/qty
    if "period" not in fact.columns and "date" in fact.columns:
        fact = fact.rename(columns={"date": "period"})
    fact["period"] = pd.to_datetime(fact["period"], errors="coerce").dt.to_period("M").astype(str)

    if "shipped_qty" not in fact.columns and "qty" in fact.columns:
        fact = fact.rename(columns={"qty": "shipped_qty"})

    fact = fact.groupby("period", as_index=False)["shipped_qty"].sum()

    # --- Приводим прогноз к формату YYYY-MM и стандартным именам ---
    fc = forecast.copy()
    if not fc.empty:
        # aliases for period
        if "period" not in fc.columns:
            for c in ["month", "date", "period_str"]:
                if c in fc.columns:
                    fc = fc.rename(columns={c: "period"})
                    break
        # aliases for forecast qty
        if "forecast_qty" not in fc.columns:
            for c in ["qty", "forecast", "forecast_sku_qty", "plan_qty"]:
                if c in fc.columns:
                    fc = fc.rename(columns={c: "forecast_qty"})
                    break

        if "period" in fc.columns:
            fc["period"] = pd.to_datetime(fc["period"], errors="coerce").dt.to_period("M").astype(str)
        if "forecast_qty" in fc.columns:
            fc = fc.groupby("period", as_index=False)["forecast_qty"].sum()
        else:
            # если нет колонки прогнозного объёма — считаем прогноз недоступным
            fc = pd.DataFrame(columns=["period", "forecast_qty"])
    else:
        fc = pd.DataFrame(columns=["period", "forecast_qty"])

    # --- Merge и графики ---
    m = fact.merge(fc, on="period", how="outer").fillna(0.0).sort_values("period")

    y_cols = [c for c in ["shipped_qty", "forecast_qty"] if c in m.columns]
    if not y_cols:
        st.info("Нет колонок для сравнения Forecast vs Actual.")
        return

    st_plot(charts.line(m, x="period", y=y_cols, title="Forecast vs Actual"))
    if len(y_cols) == 2:
        render_caption(
            title="Forecast vs Actual",
            bullets=[
                "Ось X — период (месяц)",
                "Синяя линия — факт отгрузок",
                "Оранжевая линия — прогноз/план",
            ],
            note="Отклонения помогают скорректировать план производства и промо-активности."
        )
    else:
        render_caption(
            title="Факт отгрузок",
            bullets=[
                "Ось X — период (месяц)",
                "Ось Y — отгружено, шт.",
            ],
            note="Файл с прогнозом не найден: добавьте forecast_sku_monthly.csv в GOLD, чтобы видеть сравнение."
        )
# ---------- Новая страница "Ассортимент" ----------

def page_assortment():
    st.markdown("### 🧩 Ассортимент: вклад SKU и категорий")
    # Если есть колонка категории — используем её; иначе только SKU
    cat_col = "category" if "category" in analytics.columns else None

    # Treemap: размер = выручка, цвет = маржа
    base_cols = [c for c in ["sku", "total_rev", "margin", cat_col] if c is not None and c in analytics.columns]
    if {"sku", "total_rev"}.issubset(set(base_cols)):
        df_tm = analytics[base_cols].copy()
        path_cols = [cat_col, "sku"] if cat_col else ["sku"]
        fig_tm = px.treemap(df_tm, path=path_cols, values="total_rev", color=("margin" if "margin" in df_tm.columns else None),
                            color_continuous_scale="RdYlGn", title="Treemap: вклад в выручку")
        fig_tm.update_traces(hovertemplate="%{value:.0f} ₽")
        fig_tm.update_layout(margin=dict(l=8, r=8, t=48, b=8), template="plotly_white")
        st_plot(fig_tm)
        render_caption(
            title="Treemap ассортимента",
            bullets=[
                "Размер плитки — вклад SKU/категории в выручку",
                "Цвет — маржа (краснее — ниже, зеленее — выше)",
            ],
            note="Как читать: крупные красные плитки — кандидаты на пересмотр цены, COGS или промо."
        )
    else:
        st.info("Для treemap нужны колонки 'sku' и 'total_rev'.")

    # Pareto 80/20 по выручке
    st.markdown("#### Pareto 80/20 по выручке")
    if {"sku", "total_rev"}.issubset(analytics.columns):
        d = analytics.groupby("sku", as_index=False)["total_rev"].sum().sort_values("total_rev", ascending=False)
        d["cum_pct"] = d["total_rev"].cumsum() / d["total_rev"].sum() * 100
        fig_p = go.Figure()
        fig_p.add_bar(x=d["sku"], y=d["total_rev"], name="Выручка")
        fig_p.add_trace(go.Scatter(x=d["sku"], y=d["cum_pct"], yaxis="y2", mode="lines+markers", name="Накопительный %"))
        try:
            fig_p.data[0].update(hovertemplate="%{y:.0f} ₽")   # bar: revenue
            fig_p.data[1].update(hovertemplate="%{y:.1f} %")   # line: cumulative %
        except Exception:
            pass
        fig_p.update_layout(
            template="plotly_white",
            margin=dict(l=8, r=8, t=48, b=8),
            yaxis=dict(title="Выручка, ₽"),
            yaxis2=dict(title="%", overlaying='y', side='right', range=[0, 100])
        )
        st_plot(fig_p)
        render_caption(
            title="Pareto 80/20 по выручке",
            bullets=[
                "Столбики — выручка по SKU",
                "Линия — накопительная доля (шкала справа)",
            ],
            note="Обычно 20% SKU дают ~80% выручки — на них фокусируемся в первую очередь."
        )
    else:
        st.info("Нет необходимых колонок для Pareto (нужны 'sku' и 'total_rev').")

    # Тренды помесячно (отгрузки/возвраты)
    st.markdown("#### Тренды (помесячно)")
    if not fact_monthly.empty and set(["period", "shipped_qty"]).issubset(fact_monthly.columns):
        agg = (
            fact_monthly.groupby("period", as_index=False)[["shipped_qty", "returns_qty"]]
            .sum()
            .sort_values("period")
        )
        agg["period_str"] = agg["period"].astype(str)
        if "agg" in locals():
            fig_line = charts.line(agg, x="period_str", y="shipped_qty", title="Отгружено, шт.")
            st_plot(fig_line)
            if "returns_qty" in agg.columns:
                fig_line2 = charts.line(agg, x="period_str", y="returns_qty", title="Возвраты, шт.")
                st_plot(fig_line2)
        else:
            st.info("Нет данных для построения графика (agg не определён).")


def page_sku_detail():
    st.markdown("### 🔍 SKU детально")
    sku = st.selectbox("Выберите SKU", options=sku_list)

    # Проверка наличия таблицы analytics
    if "analytics" not in globals():
        st.warning("Таблица analytics не загружена.")
        return

    row = analytics.loc[analytics["sku"].astype(str) == str(sku)]
    if row.empty:
        st.info("Нет строки в analytics для этого SKU.")
        return

    r = row.iloc[0]

    # --- KPI-блок (основные метрики) ---
    gross = float(r.get("total_rev", 0.0))
    net = float(r.get("net_revenue", gross - float(r.get("returns_rub", 0.0)) - float(r.get("promo_cost", 0.0))))
    cogs = float(r.get("cogs", 0.0))
    commission_total = float(r.get("commission_total", 0.0))
    if not commission_total:
        # оценка по ед. * qty
        commission_total = float(r.get("commission_per_unit", 0.0)) * float(r.get("total_qty", 0.0))
    margin = float(r.get("margin", net - cogs - commission_total))
    margin_pct = (margin / net * 100.0) if net else 0.0

    kpi_row([
        {"title": "Валовая выручка", "value": _format_money(gross)},
        {"title": "Чистая выручка",  "value": _format_money(net)},
        {"title": "Маржа (ИТОГО)",   "value": _format_money(margin)},
        {"title": "Маржа, %",        "value": _format_pct(margin_pct)},
    ])

    # --- Таблица с ключевыми полями по SKU ---
    cols_pref = [
        "sku", "total_qty", "avg_net_price_per_unit",
        "production_cost_per_unit", "commission_per_unit", "promo_per_unit",
        "returns_qty", "returns_pct", "total_rev", "net_revenue", "margin", "margin_pct"
    ]
    cols_present = [c for c in cols_pref if c in analytics.columns]
    df_one = pd.DataFrame([{c: r.get(c, None) for c in cols_present}])
    show_table_ru(df_one, title="Метрики по SKU")

    # --- Денежный водопад по SKU ---
    try:
        st_plot(build_waterfall(r.to_dict()))
    except Exception:
        pass

    # --- Дневной тренд по SKU ---
    try:
        if not fact_daily.empty and {"date", "sku"}.issubset(fact_daily.columns):
            d = fact_daily[fact_daily["sku"].astype(str) == str(sku)].copy()
            if not d.empty and "order_value_rub_sum" in d.columns:
                d["date"] = pd.to_datetime(d["date"], errors="coerce")
                d = d.groupby("date", as_index=False)["order_value_rub_sum"].sum().sort_values("date")
                st_plot(charts.line(d, x="date", y="order_value_rub_sum", title=f"Выручка по дням · SKU {sku}"))
    except Exception:
        pass


def page_unit_econ():
    st.markdown("### 🧮 Unit Economics")
    sku = st.selectbox("SKU", options=sku_list, index=0, key="ue_sku")

    row = analytics.loc[analytics["sku"] == sku]
    if row.empty:
        st.info("Нет метрик по выбранному SKU.")
        return
    r = row.iloc[0]

    # Водопад по составу цены/ед.
    price = float(r.get("avg_net_price_per_unit", 0.0))
    prod = float(r.get("production_cost_per_unit", 0.0))
    comm = float(r.get("commission_per_unit", 0.0))
    promo = float(r.get("promo_per_unit", 0.0))
    margin_u = float(r.get("margin_per_unit", price - prod - comm - promo))

    df = pd.DataFrame({
        "component": ["Цена (нетто)", "Себестоимость", "Комиссия", "Промо", "Маржа/ед."],
        "value": [price, -prod, -comm, -promo, margin_u]
    })
    fig_bar = charts.bar(df, x="component", y="value", title="Разложение единичной экономики")
    try:
        fig_bar.update_traces(hovertemplate="%{y:.0f} ₽")
    except Exception:
        pass
    st_plot(fig_bar)
    render_caption(
        title="Разложение единичной экономики",
        bullets=[
            "Цена (нетто) — исходная выручка на единицу",
            "Минусы — себестоимость, комиссия и промо",
            "Итог — маржа на единицу",
        ],
        note="Если маржа/ед. близка к нулю или отрицательна — меняем цену, COGS или условия комиссии."
    )

    # Порог безубыточности
    be = float(r.get("break_even_price", prod + comm + promo))
    st.info(f"Точка безубыточности: **{_format_money(be)}** на единицу.")


def page_abc_xyz():
    st.markdown("### 🧭 ABC / XYZ")
    # Табличные разрезы
    cols = st.columns(2)
    with cols[0]:
        st.markdown("#### ABC по выручке")
        if "ABC_class" in analytics.columns:
            st.dataframe(analytics["ABC_class"].value_counts().rename_axis("class").to_frame("SKUs"))
        else:
            st.info("Нет столбца ABC_class")

    with cols[1]:
        st.markdown("#### XYZ по стабильности спроса")
        if "XYZ_class" in analytics.columns:
            st.dataframe(analytics["XYZ_class"].value_counts().rename_axis("class").to_frame("SKUs"))
        else:
            st.info("Нет столбца XYZ_class")

    # Комбо-матрица
    if {"ABC_class", "XYZ_class"}.issubset(analytics.columns):
        st.markdown("#### Матрица ABC×XYZ (число SKU)")
        mat = (
            analytics.groupby(["ABC_class", "XYZ_class"])["sku"].count()
            .rename("count").reset_index().pivot(index="ABC_class", columns="XYZ_class", values="count").fillna(0).astype(int)
        )

        # Таблица метрик по SKU (если есть нужные колонки)
        cols_full = [c for c in ["sku", "total_rev", "margin", "ABC_class", "XYZ_class"] if c in analytics.columns]
        if set(["sku", "ABC_class"]).issubset(analytics.columns) and cols_full:
            st.markdown("#### Полный список SKU с ABC/XYZ")
            st.dataframe(analytics[cols_full].sort_values("total_rev", ascending=False), use_container_width=True)

        # Выручка по классам ABC и XYZ
        if {"ABC_class", "total_rev"}.issubset(analytics.columns):
            st_plot(charts.bar(
                analytics.groupby("ABC_class", as_index=False)["total_rev"].sum(),
                x="ABC_class", y="total_rev", title="Выручка по ABC"
            ))
        if {"XYZ_class", "total_rev"}.issubset(analytics.columns):
            st_plot(charts.bar(
                analytics.groupby("XYZ_class", as_index=False)["total_rev"].sum(),
                x="XYZ_class", y="total_rev", title="Выручка по XYZ"
            ))

        st.dataframe(mat)


def page_inventory():
    st.markdown("### 🏭 Остатки и оборачиваемость")
    have_inv_cols = [c for c in ["ending_stock", "average_inventory", "inventory_turnover", "opening_stock", "incoming", "outgoing"] if c in analytics.columns]
    if not have_inv_cols:
        st.info("В analytics нет складских колонок. Добавьте inventory в GOLD (см. build_gold.py).")
        return

    # KPI по всем SKU
    end_sum = float(analytics.get("ending_stock", pd.Series(dtype=float)).sum()) if "ending_stock" in analytics.columns else 0.0
    avg_inv_sum = float(analytics.get("average_inventory", pd.Series(dtype=float)).sum()) if "average_inventory" in analytics.columns else 0.0
    inv_turn = float(analytics.get("inventory_turnover", pd.Series(dtype=float)).mean()) if "inventory_turnover" in analytics.columns else 0.0
    kpi_row([
        {"title": "Остаток на конец (суммарно)", "value": f"{int(end_sum):,}".replace(",", " ")},
        {"title": "Средний запас (суммарно)", "value": f"{int(avg_inv_sum):,}".replace(",", " ")},
        {"title": "Оборачиваемость (ср.)", "value": f"{inv_turn:.2f}"},
    ])

    # ТОП по остаткам
    if "ending_stock" in analytics.columns:
        top_end = analytics.sort_values("ending_stock", ascending=False).head(20)[["sku", "ending_stock", "average_inventory", "inventory_turnover"]]
        show_table_ru(top_end, title="Топ-20 по остаткам")

    # Фильтр по SKU и отображение ряда
    sku = st.selectbox("SKU", options=sku_list, index=0, key="inv_sku")
    sub = analytics.loc[analytics["sku"] == sku]
    show_table_ru(sub[[c for c in have_inv_cols + ["sku"] if c in sub.columns]], title="Профиль SKU")


def page_what_if():
    st.markdown("### 🧪 What-if")
    if mc is None:
        st.info("Модуль monte_carlo.py не загружен: " + st.session_state.get("mc_import_error", ""))
        return

    tab_mc, tab_plan = st.tabs(["Монте-Карло (риск-маржа)", "Прогноз/План"])

    # -------- Монте-Карло --------
    with tab_mc:
        st.markdown("#### Распределение маржи с учётом неопределённости")
        st.caption("Сэмплируем цену/комиссию/промо/возвраты вокруг исторических средних; возвращаем распределение маржи на единицу и по портфелю.")

        # Параметры симуляции
        n_sims = st.slider("Число симуляций", min_value=1_000, max_value=100_000, value=20_000, step=1_000)
        seed = st.number_input("Seed (для воспроизводимости)", value=42)

        st.markdown("##### Допущения (дельты, в **процентах**)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            price_drift_pp = st.number_input("Сдвиг цены, %", value=0.0, step=0.5, help="+3 = +3 п.п. к цене")
        with c2:
            promo_delta_pp = st.number_input("Промо + п.п. от цены", value=0.0, step=0.2)
        with c3:
            comm_delta_pp = st.number_input("Комиссия + п.п. от цены", value=0.0, step=0.2)
        with c4:
            returns_delta_pp = st.number_input("Возвраты + п.п.", value=0.0, step=0.2)

        cfg = mc.MCConfig(n_sims=int(n_sims), seed=int(seed))
        ass = Assumptions(
            price_drift_pp=float(price_drift_pp) / 100.0,
            promo_delta_pp=float(promo_delta_pp) / 100.0,
            commission_delta_pp=float(comm_delta_pp) / 100.0,
            returns_delta_pp=float(returns_delta_pp),
        )

        st.markdown("##### Симуляция по SKU")
        sku = st.selectbox("SKU", options=sku_list, index=0, key="mc_sku")
        if st.button("▶︎ Запустить симуляцию по SKU"):
            try:
                res = _mc_unit_margin(analytics, sku, cfg=cfg, assumptions=ass)
                samples = res["samples"]
                q05, q50, q95 = res["p05"], res["p50"], res["p95"]
                prob_neg = res["prob_negative"]

                kpi_row([
                    {"title": "P05 (ед.)", "value": _format_money(q05)},
                    {"title": "P50 (ед.)", "value": _format_money(q50)},
                    {"title": "P95 (ед.)", "value": _format_money(q95)},
                ])
                kpi_row([{"title": "Вероятность отрицательной маржи", "value": _format_pct(100 * prob_neg)}])

                # Дополнительно: показать сводку по возвратам
                try:
                    if hasattr(mc, "MonteCarloSimulator"):
                        _row = analytics.loc[analytics["sku"] == sku]
                        if not _row.empty:
                            _r0 = _row.iloc[0]
                            _base_price = float(_r0.get("avg_net_price_per_unit", _r0.get("avg_price_per_unit", 0.0)))
                            _base_prod  = float(_r0.get("production_cost_per_unit", 0.0))
                            _base_comm  = float(_r0.get("commission_per_unit", 0.0))
                            _base_promo = float(_r0.get("promo_per_unit", 0.0))
                            _base_rr    = float(_r0.get("returns_pct", 0.0)) / 100.0
                            _qty        = [float(_r0.get("total_qty", _r0.get("shipped_qty", 0.0)) or 0.0)]
                            _sim = mc.MonteCarloSimulator(n_sims=int(n_sims), random_state=int(seed))
                            _adj_price, _adj_prod, _adj_comm, _adj_promo, _adj_rr = _apply_assumptions(
                                _base_price, _base_prod, _base_comm, _base_promo, _base_rr, ass
                            )
                            _res = _sim.simulate_sku(
                                base_price=_adj_price,
                                base_production_cost=_adj_prod,
                                base_commission_per_unit=_adj_comm,
                                base_promo_per_unit=_adj_promo,
                                base_returns_rate=_adj_rr,
                                qty=_qty,
                            )
                            _rs = getattr(_res, "returns_summary", None)
                            if _rs is not None:
                                kpi_row([
                                    {"title": "Возвраты P50", "value": _format_pct(_rs.p50)},
                                    {"title": "Возвраты Mean", "value": _format_pct(_rs.mean)},
                                ])
                except Exception:
                    pass

                # Гистограмма (теперь внутри try)
                hist = np.histogram(samples, bins=50)
                hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
                st_plot(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма маржи/ед."))

                # Скачать сэмплы
                csv = pd.Series(samples, name="unit_margin").to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Скачать распределение (CSV)", data=csv, file_name=f"mc_{sku}.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Ошибка симуляции: {e}")

        st.markdown("##### Портфельная симуляция")
        st.caption("Укажите объёмы по нескольким SKU — посчитаем распределение портфельной маржи.")

        # Простой ввод: выбор нескольких SKU и количества
        selected = st.multiselect("SKU в портфеле", options=sku_list, default=sku_list[:5])
        qty_map: Dict[str, float] = {}
        if selected:
            cols = st.columns(min(4, len(selected)))
            for i, s in enumerate(selected):
                with cols[i % len(cols)]:
                    qty_map[s] = float(st.number_input(f"{s} — qty", min_value=0.0, value=100.0, step=10.0))

        if st.button("▶︎ Запустить симуляцию портфеля"):
            try:
                res_p = _mc_portfolio_margin(analytics, qty_map, cfg=cfg)
                samples = res_p["samples"]
                kpi_row([
                    {"title": "P05 (портфель)", "value": _format_money(float(np.quantile(samples, 0.05)))},
                    {"title": "Mean (портфель)", "value": _format_money(float(np.mean(samples)))},
                    {"title": "P95 (портфель)", "value": _format_money(float(np.quantile(samples, 0.95)))},
                ])
                st.info(f"Вероятность отрицательной портфельной маржи: **{_format_pct(100 * float((samples < 0).mean()))}**")
                hist = np.histogram(samples, bins=60)
                hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
                st_plot(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма портфельной маржи"))
            except Exception as e:
                st.error(f"Ошибка симуляции портфеля: {e}")

    # -------- Планировщик / Прогноз --------
    with tab_plan:
        st.markdown("#### Прогноз и план (ForecastPlanner)")
        if ForecastPlanner is None or PlannerAssumptions is None:
            st.info("Модуль planner.py не найден или без нужных классов. Проверьте импорт `ForecastPlanner` и `Assumptions`.")
            return

        st.caption("Для прогноза нужны исходные выгрузки (orders/sales/returns/costs). Укажите пути к файлам.")
        c1, c2 = st.columns(2)
        with c1:
            orders_path = st.text_input("Файл заказов (CSV)", value="")
            sales_path = st.text_input("Отчёт о реализации (XLSX)", value="")
            returns_path = st.text_input("Возвраты (XLSX)", value="")
            costs_path = st.text_input("Себестоимость (XLSX)", value="")
        with c2:
            planned_inbound = st.text_input("План поставок (XLSX, опционально)", value="")
            horizon = st.number_input("Горизонт (мес.)", min_value=1, max_value=12, value=3, step=1)
            model = st.selectbox("Модель", options=["ets", "arima"], index=0)
            backtest = st.number_input("Backtest (последн. мес.)", min_value=0, max_value=12, value=0, step=1)

        st.markdown("##### Допущения (в %)")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            p_drift = st.number_input("Сдвиг цены, %", value=0.0, step=0.5)
        with c2:
            p_promo = st.number_input("Промо + п.п. к цене", value=0.0, step=0.2)
        with c3:
            p_comm = st.number_input("Комиссия + п.п. к цене", value=0.0, step=0.2)
        with c4:
            p_ret = st.number_input("Возвраты + п.п.", value=0.0, step=0.2)

        min_margin_pct = st.number_input("Порог маржи (мин), %", value=5.0, step=0.5)
        min_batch = st.number_input("Мин. партия, шт.", value=1, step=1)

        if st.button("▶︎ Посчитать прогноз/план"):
            try:
                ass = PlannerAssumptions(
                    price_drift=float(p_drift) / 100.0,
                    promo_delta_pp=float(p_promo) / 100.0,
                    commission_delta_pp=float(p_comm) / 100.0,
                    returns_delta_pp=float(p_ret),
                    capacity_limit=None,
                )
                planner = ForecastPlanner(
                    orders_path=Path(orders_path),
                    sales_path=Path(sales_path),
                    returns_path=Path(returns_path),
                    costs_path=Path(costs_path),
                    horizon=int(horizon),
                    model=model,
                    backtest_n=int(backtest),
                    assumptions=ass,
                    planned_inbound_path=Path(planned_inbound) if planned_inbound else None,
                    min_margin_pct=float(min_margin_pct) / 100.0,
                    min_batch=int(min_batch),
                )
                # pipeline
                planner.load_and_prepare()
                planner.forecast_sales()
                planner.compute_future_metrics()
                planner.run_backtest()

                st.success("Готово ✅")

                # Вывод
                if not planner.production_just_df.empty:
                    st.markdown("##### Production Justification (итоги по SKU)")
                    st.dataframe(planner.production_just_df.sort_values("total_margin", ascending=False))

                if not planner.future_metrics.empty:
                    st.markdown("##### Forecast (помесячно)")
                    st.dataframe(planner.future_metrics.sort_values(["sku", "period"]))

                    # Скачать как CSV
                    buff = io.BytesIO()
                    planner.future_metrics.to_csv(buff, index=False, encoding="utf-8-sig")
                    st.download_button("⬇️ Скачать Forecast (CSV)", data=buff.getvalue(), file_name="forecast.csv", mime="text/csv")

                if not planner.backtest_df.empty:
                    st.markdown("##### Backtest (MAPE по SKU, среднее)")
                    bt_summary = planner.backtest_df.groupby("sku", as_index=False)["mape_sku"].mean().rename(columns={"mape_sku": "MAPE"})
                    st.dataframe(bt_summary.sort_values("MAPE"))

            except Exception as e:
                st.error(f"Ошибка планировщика: {e}")


# ---------- Рендер выбранной страницы ----------

# --- Risk (Monte Carlo) standalone page ---

def page_risk():
    st.markdown("### 🎲 Risk (Monte Carlo)")
    if mc is None:
        st.info("Модуль monte_carlo.py не загружен: " + st.session_state.get("mc_import_error", ""))
        return

    st.markdown("#### Распределение маржи с учётом неопределённости")
    st.caption("Сэмплируем цену/комиссию/промо/возвраты вокруг исторических средних; возвращаем распределение маржи на единицу и по портфелю.")

    # Параметры симуляции
    n_sims = st.slider("Число симуляций", min_value=1_000, max_value=100_000, value=20_000, step=1_000)
    seed = st.number_input("Seed (для воспроизводимости)", value=42)

    st.markdown("##### Допущения (дельты, в **процентах**)")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        price_drift_pp = st.number_input("Сдвиг цены, %", value=0.0, step=0.5, help="+3 = +3 п.п. к цене")
    with c2:
        promo_delta_pp = st.number_input("Промо + п.п. от цены", value=0.0, step=0.2)
    with c3:
        comm_delta_pp = st.number_input("Комиссия + п.п. от цены", value=0.0, step=0.2)
    with c4:
        returns_delta_pp = st.number_input("Возвраты + п.п.", value=0.0, step=0.2)

    cfg = mc.MCConfig(n_sims=int(n_sims), seed=int(seed))
    ass = mc.Assumptions(
        price_drift_pp=float(price_drift_pp) / 100.0,
        promo_delta_pp=float(promo_delta_pp) / 100.0,
        commission_delta_pp=float(comm_delta_pp) / 100.0,
        returns_delta_pp=float(returns_delta_pp),
    )

    st.markdown("##### Симуляция по SKU")
    sku = st.selectbox("SKU", options=sku_list, index=0, key="mc_sku_standalone")
    if st.button("▶︎ Запустить симуляцию по SKU", key="mc_run_sku"):
        try:
            res = _mc_unit_margin(analytics, sku, cfg=cfg, assumptions=ass)
            samples = res["samples"]
            q05, q50, q95 = res["p05"], res["p50"], res["p95"]
            prob_neg = res["prob_negative"]

            kpi_row([
                {"title": "P05 (ед.)", "value": _format_money(q05)},
                {"title": "P50 (ед.)", "value": _format_money(q50)},
                {"title": "P95 (ед.)", "value": _format_money(q95)},
            ])
            kpi_row([{"title": "Вероятность отрицательной маржи", "value": _format_pct(100 * prob_neg)}])

            # Дополнительно: показать сводку по возвратам (если доступно через классический симулятор)
            try:
                if hasattr(mc, "MonteCarloSimulator"):
                    # Подготовим базовые параметры из analytics по выбранному SKU
                    _row = analytics.loc[analytics["sku"] == sku]
                    if not _row.empty:
                        _r0 = _row.iloc[0]
                        _base_price = float(_r0.get("avg_net_price_per_unit", _r0.get("avg_price_per_unit", 0.0)))
                        _base_prod  = float(_r0.get("production_cost_per_unit", 0.0))
                        _base_comm  = float(_r0.get("commission_per_unit", 0.0))
                        _base_promo = float(_r0.get("promo_per_unit", 0.0))
                        _base_rr    = float(_r0.get("returns_pct", 0.0)) / 100.0
                        _qty        = [float(_r0.get("total_qty", _r0.get("shipped_qty", 0.0)) or 0.0)]
                        _sim = mc.MonteCarloSimulator(n_sims=int(n_sims), random_state=int(seed))
                        _res = _sim.simulate_sku(
                            base_price=_base_price,
                            base_production_cost=_base_prod,
                            base_commission_per_unit=_base_comm,
                            base_promo_per_unit=_base_promo,
                            base_returns_rate=_base_rr,
                            qty=_qty,
                        )
                        _rs = getattr(_res, "returns_summary", None)
                        if _rs is not None:
                            kpi_row([
                                {"title": "Возвраты P50", "value": _format_pct(_rs.p50)},
                                {"title": "Возвраты Mean", "value": _format_pct(_rs.mean)},
                            ])
            except Exception:
                pass

            hist = np.histogram(samples, bins=50)
            hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
            st_plot(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма маржи/ед."))

            csv = pd.Series(samples, name="unit_margin").to_csv(index=False, encoding="utf-8-sig").encode("utf-8")
            st.download_button("⬇️ Скачать распределение (CSV)", data=csv, file_name=f"mc_{sku}.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Ошибка симуляции: {e}")

    st.markdown("##### Портфельная симуляция")
    st.caption("Укажите объёмы по нескольким SKU — посчитаем распределение портфельной маржи.")

    selected = st.multiselect("SKU в портфеле", options=sku_list, default=sku_list[:5], key="mc_portfolio_skus")
    qty_map: Dict[str, float] = {}
    if selected:
        cols = st.columns(min(4, len(selected)))
        for i, s in enumerate(selected):
            with cols[i % len(cols)]:
                qty_map[s] = float(st.number_input(f"{s} — qty", min_value=0.0, value=100.0, step=10.0, key=f"qty_{i}"))

    if st.button("▶︎ Запустить симуляцию портфеля", key="mc_run_portfolio"):
        try:
            res_p = _mc_portfolio_margin(analytics, qty_map, cfg=cfg)
            samples = res_p["samples"]
            kpi_row([
                {"title": "P05 (портфель)", "value": _format_money(float(np.quantile(samples, 0.05)))},
                {"title": "Mean (портфель)", "value": _format_money(float(np.mean(samples)))},
                {"title": "P95 (портфель)", "value": _format_money(float(np.quantile(samples, 0.95)))},
            ])
            st.info(f"Вероятность отрицательной портфельной маржи: **{_format_pct(100 * float((samples < 0).mean()))}**")
            hist = np.histogram(samples, bins=60)
            hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
            st_plot(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма портфельной маржи"))
        except Exception as e:
            st.error(f"Ошибка симуляции портфеля: {e}")

if page == "Обзор":
    page_overview()
elif page == "Ассортимент":
    page_assortment()
elif page == "SKU детально":
    page_sku_detail()
elif page == "Unit Economics":
    page_unit_econ()
elif page == "ABC/XYZ":
    page_abc_xyz()
elif page == "Остатки":
    page_inventory()
elif page == "Returns Lab":
    page_returns_lab()
elif page == "Pricing & Promo":
    page_pricing_promo()
elif page == "Forecast vs Actual":
    page_fva()
elif page == "Risk (Monte Carlo)":
    page_risk()
elif page == "What-if":
    page_what_if()
elif page == "About & Diagnostics":
    # если нет глобального forecast, передадим пустой DataFrame
    page_about_diag(fact_daily, fact_monthly, analytics, pd.DataFrame())