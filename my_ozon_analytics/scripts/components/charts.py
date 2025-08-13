# components/charts.py
# Единый слой для графиков Plotly Express с аккуратной вёрсткой.
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


_DEF_MARGIN = dict(l=8, r=8, t=48, b=8)


def _apply_layout(fig, title: Optional[str], *, show_legend: bool = True) -> None:
    fig.update_layout(
        title=title or None,
        margin=_DEF_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
        showlegend=show_legend,
    )


def _apply_y_format(
    fig,
    *,
    y_is_currency: bool = False,
    y_is_percent: bool = False,
    currency: str = "₽",
    decimals: int = 0,
) -> None:
    if y_is_percent:
        fig.update_yaxes(ticksuffix="%", tickformat=f".{max(decimals,0)}f")
    elif y_is_currency:
        # d3-format: , = разделитель тысяч
        fig.update_yaxes(tickprefix=currency, tickformat=f",.{max(decimals,0)}f")
    else:
        fig.update_yaxes(tickformat=f",.{max(decimals,0)}f")


def line(
    df: pd.DataFrame,
    x: str,
    y: str | list[str],
    *,
    color: Optional[str] = None,
    title: Optional[str] = None,
    markers: bool = True,
    y_is_currency: bool = False,
    y_is_percent: bool = False,
    currency: str = "₽",
    decimals: int = 0,
    show_legend: bool = True,
):
    """Линейный график (поддерживает множественные y)."""
    fig = px.line(df, x=x, y=y, color=color, markers=markers, title=title)
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(fig, y_is_currency=y_is_currency, y_is_percent=y_is_percent, currency=currency, decimals=decimals)
    return fig


def area(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: Optional[str] = None,
    title: Optional[str] = None,
    stackgroup: Optional[str] = "one",
    y_is_currency: bool = False,
    y_is_percent: bool = False,
    currency: str = "₽",
    decimals: int = 0,
    show_legend: bool = True,
):
    """Площадная диаграмма (stacked area)."""
    fig = px.area(df, x=x, y=y, color=color, title=title)
    if stackgroup:
        # Plotly Express сам создаёт stack по цвету; параметр оставлен для совместимости
        pass
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(fig, y_is_currency=y_is_currency, y_is_percent=y_is_percent, currency=currency, decimals=decimals)
    return fig


def scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: Optional[str] = None,
    size: Optional[str] = None,
    hover_data: Optional[list[str]] = None,
    title: Optional[str] = None,
    trendline: Optional[str] = None,   # "ols"
    show_legend: bool = True,
):
    """Точечный график (подходит для Маржа vs Выручка и т.п.)."""
    fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=hover_data, trendline=trendline, title=title)
    fig.update_traces(mode="markers")
    _apply_layout(fig, title, show_legend=show_legend)
    return fig


def bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: Optional[str] = None,
    title: Optional[str] = None,
    orientation: str = "v",            # "v" | "h"
    barmode: str = "group",            # "group" | "stack" | "relative"
    y_is_currency: bool = False,
    y_is_percent: bool = False,
    currency: str = "₽",
    decimals: int = 0,
    show_legend: bool = True,
):
    """Столбчатая диаграмма (в том числе stacked)."""
    fig = px.bar(
        df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        color=color,
        title=title,
    )
    fig.update_layout(barmode=barmode)
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(fig, y_is_currency=y_is_currency, y_is_percent=y_is_percent, currency=currency, decimals=decimals)
    return fig


def hist(
    df: pd.DataFrame,
    x: str,
    *,
    color: Optional[str] = None,
    nbins: Optional[int] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
):
    """Гистограмма распределения (например, доли возвратов)."""
    fig = px.histogram(df, x=x, color=color, nbins=nbins, title=title)
    _apply_layout(fig, title, show_legend=show_legend)
    return fig


def box(
    df: pd.DataFrame,
    y: str,
    *,
    x: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
):
    """Box-plot (удобен для сравнения маржи по классам ABC/XYZ)."""
    fig = px.box(df, x=x, y=y, color=color, title=title, points="outliers")
    _apply_layout(fig, title, show_legend=show_legend)
    return fig


def heatmap_pivot(
    pivot: pd.DataFrame,
    *,
    title: Optional[str] = None,
    colorscale: str = "Blues",
    show_legend: bool = True,
):
    """Тепловая карта для уже сформированного pivot-DataFrame.
    Ожидается, что индекс = строки (например, SKU), колонки = периоды/категории.
    """
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=colorscale,
            colorbar=dict(title=""),
        )
    )
    _apply_layout(fig, title, show_legend=show_legend)
    return fig