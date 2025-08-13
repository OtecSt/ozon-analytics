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
        font=dict(family="Arial", size=12),
        title_font=dict(family="Arial", size=16),
        plot_bgcolor="#fafafa",
        paper_bgcolor="#ffffff",
        xaxis=dict(showgrid=True, gridcolor="#e0e0e0", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="#e0e0e0", zeroline=False),
        template="plotly_white",
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
    show_values: bool = False,
):
    """Линейный график (поддерживает множественные y)."""
    fig = px.line(df, x=x, y=y, color=color, markers=markers, title=title)
    fig.update_traces(hovertemplate="%{y}")
    if show_values:
        fig.update_traces(text=df[y] if isinstance(y, str) else None, textposition="top center")
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
    show_values: bool = False,
):
    """Площадная диаграмма (stacked area)."""
    fig = px.area(df, x=x, y=y, color=color, title=title)
    if stackgroup:
        # Plotly Express сам создаёт stack по цвету; параметр оставлен для совместимости
        pass
    fig.update_traces(hovertemplate="%{y}")
    if show_values:
        fig.update_traces(text=df[y], textposition="top center")
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
    fig.update_traces(hovertemplate="%{y}")
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
    show_values: bool = False,
):
    """Столбчатая диаграмма (в том числе stacked)."""
    fig = px.bar(
        df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        color=color,
        title=title,
    )
    fig.update_traces(hovertemplate="%{y}")
    if show_values:
        fig.update_traces(text=df[y] if isinstance(y, str) else None, textposition="auto")
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
    fig.update_traces(hovertemplate="%{x}: %{y}")
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
    fig.update_traces(hovertemplate="%{y}")
    _apply_layout(fig, title, show_legend=show_legend)
    return fig


def heatmap_pivot(
    pivot: pd.DataFrame,
    *,
    title: Optional[str] = None,
    colorscale: str = "Blues",
    show_legend: bool = True,
    show_values: bool = True,
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
            text=pivot.values if show_values else None,
            texttemplate="%{z}" if show_values else None,
            textfont=dict(color="black"),
        )
    )
    _apply_layout(fig, title, show_legend=show_legend)
    return fig


# --- Новые визуальные примитивы ---

import plotly.express as px
import plotly.graph_objects as go

def waterfall(labels, values, title=None):
    measures = ['relative'] * len(values)
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line":{"width":1}}
    ))
    fig.update_layout(title=title, template="plotly_white", margin=dict(l=8, r=8, t=48, b=8))
    return fig

def treemap_revenue(df, path_cols, value_col, color_col=None, title=None):
    fig = px.treemap(df, path=path_cols, values=value_col, color=color_col, title=title, color_continuous_scale="RdYlGn")
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8))
    return fig

def pareto(df, x_col, value_col, title=None):
    d = df.groupby(x_col, as_index=False)[value_col].sum().sort_values(value_col, ascending=False)
    d['cum_pct'] = d[value_col].cumsum() / d[value_col].sum() * 100
    fig = go.Figure()
    fig.add_bar(x=d[x_col], y=d[value_col], name=value_col)
    fig.add_trace(go.Scatter(x=d[x_col], y=d['cum_pct'], yaxis="y2", mode="lines+markers", name="Накопительный %"))
    fig.update_layout(
        title=title, template="plotly_white",
        yaxis=dict(title=value_col),
        yaxis2=dict(title="%", overlaying='y', side='right', rangemode='tozero', range=[0, 100]),
        margin=dict(l=8, r=8, t=48, b=8)
    )
    return fig

def heatmap_calendar(pivot, title=None):
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), coloraxis="coloraxis"))
    fig.update_layout(title=title, coloraxis=dict(colorscale="Blues"), template="plotly_white", margin=dict(l=8, r=8, t=48, b=8))
    return fig

def scatter_margin_returns(df, x='returns_pct', y='margin', color='category', hover=None, title=None):
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover, title=title)
    fig.update_layout(template="plotly_white", margin=dict(l=8, r=8, t=48, b=8))
    return fig

def target_line(fig, x0, x1, y, name="План"):
    fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(dash="dash", width=2), name=name)
    return fig


# Алиасы для совместимости с предыдущими версиями
def line_chart(df, x, y, **kwargs):
    """Алиас для line() для совместимости с предыдущими версиями."""
    return line(df, x, y, **kwargs)

def bar_chart(df, x, y, **kwargs):
    """Алиас для bar() для совместимости с предыдущими версиями."""
    return bar(df, x, y, **kwargs)