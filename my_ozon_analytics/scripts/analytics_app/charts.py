# components/charts.py
# Единый слой для графиков Plotly с аккуратной вёрсткой и тёмной темой.
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


# === Темизация: принудительный тёмный шаблон ===
def _ensure_dark(fig: go.Figure) -> go.Figure:
    """
    Применяет тёмный шаблон и фон на момент вызова, не завися от порядка импортов.
    Если в приложении зарегистрирован "nardo_choco_dark", используем его,
    иначе — текущий default у Plotly.
    """
    tmpl = "nardo_choco_dark" if "nardo_choco_dark" in pio.templates else pio.templates.default
    try:
        fig.update_layout(
            template=tmpl,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#262a2f",
            hoverlabel=dict(bgcolor="rgba(34,36,40,0.9)"),
        )
    except Exception:
        pass
    return fig


_DEF_MARGIN = dict(l=8, r=8, t=48, b=8)


def _apply_layout(fig: go.Figure, title: Optional[str], *, show_legend: bool = True) -> None:
    fig.update_layout(
        title=title or None,
        margin=_DEF_MARGIN,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        hovermode="x unified",
        showlegend=show_legend,
        font=dict(family="Inter, sans-serif", size=12, color="#e6e6e6"),
        title_font=dict(family="Inter, sans-serif", size=16, color="#d4a373"),
        plot_bgcolor="#262a2f",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.2)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)", zeroline=False, linecolor="rgba(255,255,255,0.2)"),
    )


def _apply_y_format(
    fig: go.Figure,
    *,
    y_is_currency: bool = False,
    y_is_percent: bool = False,
    currency: str = "₽",
    decimals: int = 0,
) -> None:
    if y_is_percent:
        fig.update_yaxes(ticksuffix="%", tickformat=f".{max(decimals,0)}f")
    elif y_is_currency:
        fig.update_yaxes(tickprefix=currency, tickformat=f",.{max(decimals,0)}f")
    else:
        fig.update_yaxes(tickformat=f",.{max(decimals,0)}f")


def _apply_hover_format(fig: go.Figure, y_is_currency: bool, y_is_percent: bool) -> go.Figure:
    # Единые тултипы на оси Y (цифры читаемые на тёмном)
    if y_is_currency:
        fig.update_traces(hovertemplate="%{y:,.0f} ₽")
    elif y_is_percent:
        fig.update_traces(hovertemplate="%{y:.1f} %")
    else:
        fig.update_traces(hovertemplate="%{y:,.0f}")
    return fig


# ---------- Базовые примитивы ----------

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
) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, markers=markers, title=title)
    if show_values and isinstance(y, str):
        fig.update_traces(text=df[y], textposition="top center")
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(fig, y_is_currency=y_is_currency, y_is_percent=y_is_percent, currency=currency, decimals=decimals)
    _apply_hover_format(fig, y_is_currency, y_is_percent)
    return _ensure_dark(fig)


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
) -> go.Figure:
    fig = px.area(df, x=x, y=y, color=color, title=title)
    if show_values:
        fig.update_traces(text=df[y], textposition="top center")
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(fig, y_is_currency=y_is_currency, y_is_percent=y_is_percent, currency=currency, decimals=decimals)
    _apply_hover_format(fig, y_is_currency, y_is_percent)
    return _ensure_dark(fig)


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
) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, size=size, hover_data=hover_data, trendline=trendline, title=title)
    fig.update_traces(mode="markers")
    _apply_layout(fig, title, show_legend=show_legend)
    return _ensure_dark(fig)


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
) -> go.Figure:
    fig = px.bar(
        df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        color=color,
        title=title,
    )
    if show_values and isinstance(y, str):
        fig.update_traces(text=df[y], textposition="auto")
    fig.update_layout(barmode=barmode)
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(fig, y_is_currency=y_is_currency, y_is_percent=y_is_percent, currency=currency, decimals=decimals)
    _apply_hover_format(fig, y_is_currency, y_is_percent)
    return _ensure_dark(fig)


def hist(
    df: pd.DataFrame,
    x: str,
    *,
    color: Optional[str] = None,
    nbins: Optional[int] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    fig = px.histogram(df, x=x, color=color, nbins=nbins, title=title)
    fig.update_traces(hovertemplate="%{x}: %{y}")
    _apply_layout(fig, title, show_legend=show_legend)
    return _ensure_dark(fig)


def box(
    df: pd.DataFrame,
    y: str,
    *,
    x: Optional[str] = None,
    color: Optional[str] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
) -> go.Figure:
    fig = px.box(df, x=x, y=y, color=color, title=title, points="outliers")
    _apply_layout(fig, title, show_legend=show_legend)
    return _ensure_dark(fig)


def heatmap_pivot(
    pivot: pd.DataFrame,
    *,
    title: Optional[str] = None,
    colorscale: str = "Blues",
    show_legend: bool = True,
    show_values: bool = True,
) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=colorscale,
            colorbar=dict(title=""),
            text=pivot.values if show_values else None,
            texttemplate="%{z}" if show_values else None,
        )
    )
    _apply_layout(fig, title, show_legend=show_legend)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return _ensure_dark(fig)


# --- Сложные примитивы ---

def waterfall(labels: list[str], values: list[float], title: str | None = None) -> go.Figure:
    measures = ["relative"] * (len(values) - 1) + ["total"]
    fig = go.Figure(go.Waterfall(
        name="",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        connector={"line": {"width": 1}},
    ))
    fig.update_traces(hovertemplate="%{y:,.0f} ₽")
    fig.update_layout(title=title, margin=_DEF_MARGIN)
    return _ensure_dark(fig)


def treemap_revenue(df: pd.DataFrame, path_cols, value_col, color_col: str | None = None, title: str | None = None) -> go.Figure:
    fig = px.treemap(df, path=path_cols, values=value_col, color=color_col, title=title, color_continuous_scale="RdYlGn")
    fig.update_layout(margin=_DEF_MARGIN)
    return _ensure_dark(fig)


def pareto(df: pd.DataFrame, x_col: str, y_col: str, title: str = "Pareto 80/20", x_tickangle: int = -45) -> go.Figure:
    """
    Pareto 80/20: столбики = метрика (например, total_rev) по SKU/категории,
    линия = накопительная доля, правая ось Y (0..100%).
    """
    d = (df.groupby(x_col, as_index=False)[y_col]
           .sum()
           .sort_values(y_col, ascending=False))
    if d.empty:
        return _ensure_dark(go.Figure())

    d["cum_pct"] = d[y_col].cumsum() / d[y_col].sum() * 100
    xcat = d[x_col].astype(str)  # одинаковый тип оси X для bar и scatter

    fig = go.Figure()
    fig.add_bar(x=xcat, y=d[y_col], name=y_col)
    fig.add_trace(go.Scatter(
        x=xcat, y=d["cum_pct"],
        mode="lines+markers",
        name="Накопительный %",
        yaxis="y2",
    ))

    fig.update_traces(selector=dict(type="bar"), hovertemplate="%{y:,.0f}")
    fig.update_traces(selector=dict(type="scatter"), hovertemplate="%{y:.1f} %")

    fig.update_layout(
        title=title,
        margin=_DEF_MARGIN,
        xaxis=dict(tickangle=x_tickangle),
        yaxis=dict(title=y_col),
        yaxis2=dict(title="%", overlaying="y", side="right", range=[0, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return _ensure_dark(fig)


def heatmap_calendar(pivot: pd.DataFrame, title: str | None = None) -> go.Figure:
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            coloraxis="coloraxis",
        )
    )
    fig.update_layout(title=title, coloraxis=dict(colorscale="Blues"), margin=_DEF_MARGIN)
    return _ensure_dark(fig)


def scatter_margin_returns(df: pd.DataFrame, x: str = "returns_pct", y: str = "margin", color: str = "category", hover=None, title: str | None = None) -> go.Figure:
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover, title=title)
    fig.update_layout(margin=_DEF_MARGIN)
    return _ensure_dark(fig)


def target_line(fig: go.Figure, x0, x1, y, name: str = "План") -> go.Figure:
    fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(dash="dash", width=2), name=name)
    return fig


# Алиасы для совместимости с предыдущими версиями
def line_chart(df, x, y, **kwargs):  # noqa: D401
    """Алиас для line() для совместимости с предыдущими версиями."""
    return line(df, x, y, **kwargs)


def bar_chart(df, x, y, **kwargs):  # noqa: D401
    """Алиас для bar() для совместимости с предыдущими версиями."""
    return bar(df, x, y, **kwargs)