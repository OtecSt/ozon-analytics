
from __future__ import annotations
import sys
from pathlib import Path

# Абсолютные пути
APP_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = APP_DIR.parent
ROOT_DIR = SCRIPTS_DIR.parent

for p in (APP_DIR, SCRIPTS_DIR, ROOT_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Проверка на месте ли модуль
mc_path = APP_DIR / "monte_carlo.py"
print(f"[DEBUG] monte_carlo exists: {mc_path.exists()} at {mc_path}")

try:
    import monte_carlo as mc
    print(f"[DEBUG] monte_carlo imported OK from {getattr(mc, '__file__', '?')}")
except Exception as e:
    print(f"[ERROR] Failed to import monte_carlo: {e}")
    mc = None


import io
from typing import Dict

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

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
    """Отображает таблицу с русскими заголовками (если известны)."""
    if title:
        st.markdown(f"#### {title}")
    st.dataframe(df_ru(df), use_container_width=use_container_width)


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
            return fig
        @staticmethod
        def bar(df, x, y, title=None, **kwargs):
            fig = px.bar(df, x=x, y=y, title=title)
            return fig
        @staticmethod
        def scatter(df, x, y, color=None, hover_data=None, title=None, **kwargs):
            fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover_data, title=title)
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
    page_title="Ozon Analytics & Planning",
    page_icon="📦",
    layout="wide",
)


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
    load_bundle.clear()

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

# --- Sidebar filters (depend on loaded data) ---
with st.sidebar:
    st.markdown("---")
    st.markdown("## 📅 Фильтры")
    granularity = st.radio("Гранулярность", ["День","Неделя","Месяц"], index=0, horizontal=True)
    date_from = st.date_input("С даты", value=pd.to_datetime("2025-01-01"))
    date_to   = st.date_input("По дату", value=pd.to_datetime("today"))
    cogs_mode = st.selectbox("COGS режим", ["NET", "GROSS"], index=(0 if COGS_MODE == "NET" else 1))
    # динамический список SKU
    _sku_list = sorted(analytics["sku"].astype(str).unique().tolist())
    selected_sku = st.multiselect("SKU", _sku_list[:50], max_selections=50)

# --- Apply filters ---
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
    col1, col2 = st.columns([2, 3])

    with col1:
        # KPI на основе фильтрованных данных, с fallback на analytics
        if not _daily.empty and {"order_value_rub_sum"}.issubset(_daily.columns):
            _rev = float(_daily["order_value_rub_sum"].sum())
        else:
            _rev = rev_sum
        _net = net_rev_sum  # при отсутствии нетто в daily оставляем из analytics
        _margin = margin_sum
        kpi_row([
            {"title": "Валовая выручка", "value": _format_money(_rev)},
            {"title": "Чистая выручка", "value": _format_money(_net)},
            {"title": "Маржа (ИТОГО)", "value": _format_money(_margin)},
        ])
        # KPI (доля возвратов, доля промо, рискованные SKU)
        # Возвраты ₽: приоритет готовой суммы; иначе оценка avg_net_price_per_unit * returns_qty
        if "returns_rub" in analytics.columns:
            _returns_rub = float(analytics["returns_rub"].sum())
        elif {"avg_net_price_per_unit", "returns_qty"}.issubset(analytics.columns):
            _returns_rub = float((analytics["avg_net_price_per_unit"] * analytics["returns_qty"]).sum())
        else:
            _returns_rub = 0.0
        _promo_rub = float(_daily.get("promo_rub", pd.Series(dtype=float)).sum()) if not _daily.empty else float(promo_sum)
        # Рискованные SKU: возвраты% > RETURNS_ALERT_PCT п.п. или маржа < 0 (если столбцы есть)
        thr = float(RETURNS_ALERT_PCT)
        if {"returns_pct", "margin"}.issubset(analytics.columns):
            risk_cnt = int(((analytics["returns_pct"] > thr) | (analytics["margin"] < 0)).sum())
        elif "returns_pct" in analytics.columns:
            risk_cnt = int((analytics["returns_pct"] > thr).sum())
        elif "margin" in analytics.columns:
            risk_cnt = int((analytics["margin"] < 0).sum())
        else:
            risk_cnt = 0
        kpi_row([
            {"title": "Возвраты, %", "value": _format_pct((_returns_rub / _rev * 100) if _rev else 0)},
            {"title": "Промо, %", "value": _format_pct((_promo_rub / _rev * 100) if _rev else 0)},
            {"title": "SKU в риске", "value": f"{risk_cnt}"},
        ])

    with col2:
        show_scatter = not analytics.get("total_rev", pd.Series([])).empty and not analytics.get("margin", pd.Series([])).empty
        if show_scatter:
            fig = charts.scatter(
                analytics.rename(columns={"total_rev": "revenue", "margin": "margin"}),
                x="revenue", y="margin", color="ABC_class" if "ABC_class" in analytics.columns else None,
                hover_data=["sku"], title="Маржа vs Выручка по SKU"
            )
            st.plotly_chart(fig, use_container_width=True)
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
        st.plotly_chart(
            charts.line(ts, x="period", y=[c for c in ["order_value_rub_sum", "SMA"] if c in ts.columns],
                        title=f"Динамика выручки · {granularity}"),
            use_container_width=True,
        )
        render_caption(
            title=f"Динамика выручки · {granularity}",
            bullets=[
                "Ось X — период (день/неделя/месяц)",
                "Ось Y — выручка, ₽",
                f"Синяя линия — сглаживание SMA{str(sma_window)}",
            ],
            note=trend_summary(ts.rename(columns={"period": "date"}), "date", "order_value_rub_sum", sma_window=sma_window),
        )
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
    st.plotly_chart(fig_wf, use_container_width=True)
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
        st.plotly_chart(fig_sc, use_container_width=True)
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
        st.plotly_chart(charts.heatmap_pivot(pv, title="Возвраты по дням и SKU"), use_container_width=True)
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

    st.plotly_chart(charts.bar(df.nlargest(int(top_n), "margin_adj"), x="sku", y="margin_adj", title="Маржа после изменений", orientation="v", y_is_currency=True), use_container_width=True)
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

    st.plotly_chart(
        charts.line(m, x="period", y=y_cols, title="Forecast vs Actual"),
        use_container_width=True
    )
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
        fig_tm.update_layout(margin=dict(l=8, r=8, t=48, b=8), template="plotly_white")
        st.plotly_chart(fig_tm, use_container_width=True)
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
        fig_p.update_layout(
            template="plotly_white",
            margin=dict(l=8, r=8, t=48, b=8),
            yaxis=dict(title="Выручка, ₽"),
            yaxis2=dict(title="%", overlaying='y', side='right', range=[0, 100])
        )
        st.plotly_chart(fig_p, use_container_width=True)
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
        fig_line = charts.line(agg, x="period_str", y="shipped_qty", title="Отгружено, шт.")
        st.plotly_chart(fig_line, use_container_width=True)
        if "returns_qty" in agg.columns:
            fig_line2 = charts.line(agg, x="period_str", y="returns_qty", title="Возвраты, шт.")
            st.plotly_chart(fig_line2, use_container_width=True)


def page_sku_detail():
    st.markdown("### 🔎 SKU детально")
    sku = st.selectbox("Выберите SKU", options=sku_list, index=0)
    row = analytics.loc[analytics["sku"] == sku]
    if row.empty:
        st.info("Нет строки в analytics для этого SKU.")
        return
    r = row.iloc[0].to_dict()

    # KPI
    kpi_row([
        {"title": "Выручка (вал.)", "value": _format_money(float(r.get("total_rev", 0)))},
        {"title": "Чистая выручка", "value": _format_money(float(r.get("net_revenue", 0)))},
        {"title": "Маржа", "value": _format_money(float(r.get("margin", 0)))},
    ])
    kpi_row([
        {"title": "Доля возвратов", "value": _format_pct(float(r.get("returns_pct", 0)))},
        {"title": "Интенсивность промо", "value": _format_pct(float(r.get("promo_intensity_pct", 0)))},
        {"title": "Рекомендация", "value": r.get("recommended_action", "—")},
    ])

    # Таймсерии по месяцу
    st.markdown("#### Динамика (месяц)")
    sub = fact_monthly.loc[fact_monthly["sku"] == sku].copy()
    if not sub.empty and "period" in sub.columns:
        sub["period_str"] = sub["period"].astype(str)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(charts.line(sub, x="period_str", y="shipped_qty", title="Отгружено, шт."), use_container_width=True)
        with c2:
            if "returns_qty" in sub.columns:
                st.plotly_chart(charts.line(sub, x="period_str", y="returns_qty", title="Возвраты, шт."), use_container_width=True)

    # Табличка unit-econ
    keep_cols = [
        "avg_price_per_unit","avg_net_price_per_unit","production_cost_per_unit",
        "commission_per_unit","promo_per_unit","margin_per_unit",
        "break_even_price","contribution_margin","margin_pct"
    ]
    st.markdown("#### Unit-economics (единица)")
    st.dataframe(row[["sku"] + [c for c in keep_cols if c in row.columns]].reset_index(drop=True))
    render_caption(
        title="SKU детально",
        bullets=[
            "Помесячная динамика показывает сезонность и эффект акций",
            "Сравнение фактической маржи со средними значениями по портфелю",
        ],
        note="Если маржа ниже портфельной и возвраты выше среднего — проверьте цену, COGS и качество логистики.",
    )

    # Водопад Unit Economics по выбранному SKU
    r = row.iloc[0]
    rev = float(r.get("total_rev", 0))
    # Возвраты ₽ — готовая колонка или оценка
    if "returns_rub" in row.columns:
        returns_rub = float(r.get("returns_rub", 0))
    elif {"avg_net_price_per_unit", "returns_qty"}.issubset(row.columns):
        returns_rub = float(r.get("avg_net_price_per_unit", 0) * r.get("returns_qty", 0))
    else:
        returns_rub = 0.0
    commission_rub = float(r.get("commission_per_unit", 0) * r.get("total_qty", 0))
    promo_rub = float(r.get("promo_per_unit", 0) * r.get("total_qty", 0))
    cogs_rub = float(r.get("cogs", 0))

    labels = ["Валовая выручка", "- Возвраты", "- Комиссия", "- Промо", "- COGS", "Маржа (итог)"]
    values = [rev, -returns_rub, -commission_rub, -promo_rub, -cogs_rub, rev - returns_rub - commission_rub - promo_rub - cogs_rub]
    st.plotly_chart(charts.waterfall(labels, values, title="Unit Econ: мостик по SKU"), use_container_width=True)
    render_caption(
        title="Unit economics: мостик выручка → маржа",
        bullets=[
            "Столбцы — последовательные эффекты: возвраты, комиссия, промо, COGS",
            "Последний столбец — итоговая маржа",
        ],
        note="Как читать: какой фактор «съедает» больше всего маржи, туда и направляем первую оптимизацию.",
    )


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
    st.plotly_chart(fig_bar, use_container_width=True)
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
            st.plotly_chart(charts.bar(
                analytics.groupby("ABC_class", as_index=False)["total_rev"].sum(),
                x="ABC_class", y="total_rev", title="Выручка по ABC"
            ), use_container_width=True)
        if {"XYZ_class", "total_rev"}.issubset(analytics.columns):
            st.plotly_chart(charts.bar(
                analytics.groupby("XYZ_class", as_index=False)["total_rev"].sum(),
                x="XYZ_class", y="total_rev", title="Выручка по XYZ"
            ), use_container_width=True)

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
        st.info("Модуль monte_carlo.py не найден; раздел What‑if (Монте‑Карло) временно недоступен.")
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
        ass = mc.Assumptions(
            price_drift_pp=float(price_drift_pp) / 100.0,
            promo_delta_pp=float(promo_delta_pp) / 100.0,
            commission_delta_pp=float(comm_delta_pp) / 100.0,
            returns_delta_pp=float(returns_delta_pp),
        )

        st.markdown("##### Симуляция по SKU")
        sku = st.selectbox("SKU", options=sku_list, index=0, key="mc_sku")
        if st.button("▶︎ Запустить симуляцию по SKU"):
            try:
                res = mc.simulate_unit_margin(analytics, sku, cfg=cfg, assumptions=ass)
                samples = res["samples"]
                q05, q50, q95 = res["p05"], res["p50"], res["p95"]
                prob_neg = res["prob_negative"]

                kpi_row([
                    {"title": "P05 (ед.)", "value": _format_money(q05)},
                    {"title": "P50 (ед.)", "value": _format_money(q50)},
                    {"title": "P95 (ед.)", "value": _format_money(q95)},
                ])
                kpi_row([{"title": "Вероятность отрицательной маржи", "value": _format_pct(100 * prob_neg)}])

                # Гистограмма
                hist = np.histogram(samples, bins=50)
                hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
                st.plotly_chart(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма маржи/ед."), use_container_width=True)

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
                res_p = mc.simulate_portfolio_margin(analytics, qty_map, cfg=cfg)
                samples = res_p["samples"]
                kpi_row([
                    {"title": "P05 (портфель)", "value": _format_money(float(np.quantile(samples, 0.05)))},
                    {"title": "Mean (портфель)", "value": _format_money(float(np.mean(samples)))},
                    {"title": "P95 (портфель)", "value": _format_money(float(np.quantile(samples, 0.95)))},
                ])
                st.info(f"Вероятность отрицательной портфельной маржи: **{_format_pct(100 * float((samples < 0).mean()))}**")
                hist = np.histogram(samples, bins=60)
                hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
                st.plotly_chart(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма портфельной маржи"), use_container_width=True)
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
        st.info("Модуль monte_carlo.py не найден; раздел Monte‑Carlo временно недоступен.")
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
            res = mc.simulate_unit_margin(analytics, sku, cfg=cfg, assumptions=ass)
            samples = res["samples"]
            q05, q50, q95 = res["p05"], res["p50"], res["p95"]
            prob_neg = res["prob_negative"]

            kpi_row([
                {"title": "P05 (ед.)", "value": _format_money(q05)},
                {"title": "P50 (ед.)", "value": _format_money(q50)},
                {"title": "P95 (ед.)", "value": _format_money(q95)},
            ])
            kpi_row([{"title": "Вероятность отрицательной маржи", "value": _format_pct(100 * prob_neg)}])

            hist = np.histogram(samples, bins=50)
            hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
            st.plotly_chart(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма маржи/ед."), use_container_width=True)

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
            res_p = mc.simulate_portfolio_margin(analytics, qty_map, cfg=cfg)
            samples = res_p["samples"]
            kpi_row([
                {"title": "P05 (портфель)", "value": _format_money(float(np.quantile(samples, 0.05)))},
                {"title": "Mean (портфель)", "value": _format_money(float(np.mean(samples)))},
                {"title": "P95 (портфель)", "value": _format_money(float(np.quantile(samples, 0.95)))},
            ])
            st.info(f"Вероятность отрицательной портфельной маржи: **{_format_pct(100 * float((samples < 0).mean()))}**")
            hist = np.histogram(samples, bins=60)
            hist_df = pd.DataFrame({"bin_left": hist[1][:-1], "count": hist[0]})
            st.plotly_chart(charts.bar(hist_df, x="bin_left", y="count", title="Гистограмма портфельной маржи"), use_container_width=True)
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