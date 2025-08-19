# components/kpis.py
# Унифицированные KPI-блоки для Streamlit.
from __future__ import annotations
from typing import Any, Iterable, Optional
# Нормализация входного элемента KPI: поддержка dict и (title, value[, delta])
from collections.abc import Iterable as _Iterable

def _normalize_kpi_item(it: Any) -> dict:
    """Приводит вход к единому словарю для kpi_row/kpi_grid.
    Допускаются формы:
      - dict: {title, value, delta?, help?, kind?, delta_kind?, currency?, decimals?, delta_decimals?, invert?}
      - tuple/list: (title, value) или (title, value, delta)
      - иные типы: трактуются как value с заголовком 'KPI'
    """
    if isinstance(it, dict):
        return {
            "title": it.get("title", it.get("label", "KPI")),
            "value": it.get("value"),
            "delta": it.get("delta"),
            "help": it.get("help") or it.get("help_text"),
            "kind": it.get("kind"),
            "delta_kind": it.get("delta_kind"),
            "currency": it.get("currency", "₽"),
            "decimals": it.get("decimals"),
            "delta_decimals": it.get("delta_decimals"),
            "invert": bool(it.get("invert") or it.get("invert_good_bad", False)),
        }
    if isinstance(it, (list, tuple)) and len(it) in (2, 3):
        title, value = it[0], it[1]
        delta = it[2] if len(it) == 3 else None
        return {"title": str(title), "value": value, "delta": delta}
    # fallback: любое другое значение — как value без delta
    return {"title": "KPI", "value": it}


import math
import streamlit as st


# ========= helpers =========

def _fmt_number(
    value: Any,
    *,
    kind: str | None = None,          # "currency" | "percent" | "int" | "float" | None -> авто
    currency: str = "₽",
    decimals: int | None = None,      # None => авто
) -> str:
    """Человеко-читабельное форматирование чисел для метрик."""
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "—"

    # авто-тип, если не задан
    if kind is None:
        if isinstance(value, (int,)) and not isinstance(value, bool):
            kind = "int"
        elif isinstance(value, (float,)):
            kind = "float"
        else:
            # строка/прочее — просто вернём
            return str(value)

    # проценты
    if kind == "percent":
        v = float(value)
        d = 1 if decimals is None else decimals
        return f"{v:.{d}f}%"

    # валюта
    if kind == "currency":
        v = float(value)
        d = 0 if decimals is None else decimals
        # разделитель тысяч — с запятыми (Plotly/Streamlit рендерят ок)
        return f"{currency}{v:,.{d}f}"

    # целое
    if kind == "int":
        try:
            v = int(value)
        except Exception:
            v = round(float(value))
        return f"{v:,}"

    # float по умолчанию
    v = float(value)
    d = 2 if decimals is None else decimals
    return f"{v:,.{d}f}"


def _fmt_delta(
    delta: Any,
    *,
    kind: str | None = None,      # "percent" | "currency" | "int" | "float"
    currency: str = "₽",
    decimals: int | None = None,
) -> str:
    if delta is None:
        return ""
    # если строка — доверяем пользователю
    if isinstance(delta, str):
        return delta

    sign = "+" if float(delta) >= 0 else ""
    body = _fmt_number(delta, kind=kind, currency=currency, decimals=decimals)
    return f"{sign}{body}"


# ========= public API =========

def kpi(
    title: str,
    value: Any,
    *,
    delta: Any | None = None,
    help_text: str | None = None,
    kind: str | None = None,          # формат основного значения
    delta_kind: str | None = None,    # формат дельты
    currency: str = "₽",
    decimals: int | None = None,
    delta_decimals: int | None = None,
    invert_good_bad: bool = False,    # True — «меньше лучше» (например, расходы)
) -> None:
    """Единый KPI-блок (обёртка над st.metric) с нормальным форматированием."""
    if help_text:
        st.caption(help_text)

    value_str = _fmt_number(value, kind=kind, currency=currency, decimals=decimals)
    delta_str = _fmt_delta(delta, kind=delta_kind, currency=currency, decimals=delta_decimals) if delta is not None else None

    st.metric(
        label=title,
        value=value_str,
        delta=delta_str,
        delta_color=("inverse" if invert_good_bad else "normal"),
    )


def kpi_row(items: _Iterable[Any]) -> None:
    """
    Рендерит метрики в одну строку.
    Поддерживает элементы как словари, так и кортежи/списки:
      - dict: {title, value, delta?, help?, kind?, delta_kind?, currency?, decimals?, delta_decimals?, invert?}
      - tuple/list: (title, value) или (title, value, delta)
    """
    items = list(items)
    if not items:
        return
    norm = [_normalize_kpi_item(it) for it in items]
    cols = st.columns(len(norm))
    for c, it in zip(cols, norm):
        with c:
            kpi(
                it.get("title", "KPI"),
                it.get("value"),
                delta=it.get("delta"),
                help_text=it.get("help"),
                kind=it.get("kind"),
                delta_kind=it.get("delta_kind"),
                currency=it.get("currency", "₽"),
                decimals=it.get("decimals"),
                delta_decimals=it.get("delta_decimals"),
                invert_good_bad=bool(it.get("invert", False)),
            )

# Совместимость: карточка KPI как алиас к kpi

def kpi_card(title: str, value: Any, **kwargs) -> None:
    kpi(title, value, **kwargs)


def kpi_grid(items: list[dict], *, cols: int = 3) -> None:
    """
    Рендерит метрики сеткой по 'cols' в строке.
    """
    if not items:
        return
    row: list[dict] = []
    for i, it in enumerate(items, 1):
        row.append(it)
        if i % cols == 0:
            kpi_row(row)
            row = []
    if row:
        kpi_row(row)