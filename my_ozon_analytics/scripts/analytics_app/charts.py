from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ---- Единые настройки оформления ----
_DEF_MARGIN = dict(l=8, r=8, t=48, b=8)


def _ensure_dark(fig: go.Figure) -> go.Figure:
    """Принудительно применяет тёмный шаблон и фон (без зависимости от порядка импортов)."""
    tmpl = "nardo_choco_dark" if "nardo_choco_dark" in pio.templates else pio.templates.default
    try:
        fig.update_layout(
            template=tmpl,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="#262a2f",
            font=dict(family="Inter, sans-serif", size=12, color="#e6e6e6"),
            title_font=dict(family="Inter, sans-serif", size=16, color="#d4a373"),
            hoverlabel=dict(bgcolor="rgba(34,36,40,0.9)"),
            margin=_DEF_MARGIN,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            hovermode="x unified",
        )
    except Exception:
        pass
    return fig


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


def _apply_layout(fig: go.Figure, title: Optional[str], *, show_legend: bool = True) -> None:
    fig.update_layout(showlegend=show_legend, title=title or None)
    # сетка под тёмный фон
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.2)",
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.08)",
        zeroline=False,
        linecolor="rgba(255,255,255,0.2)",
    )


def _apply_hover_format(fig: go.Figure, y_is_currency: bool, y_is_percent: bool) -> go.Figure:
    if y_is_currency:
        fig.update_traces(hovertemplate="%{y:.0f} ₽")
    elif y_is_percent:
        fig.update_traces(hovertemplate="%{y:.1f} %")
    return fig


# ---- Базовые примитивы ----
def line(
    df: pd.DataFrame,
    x,
    y,
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
    fig = px.line(df, x=x, y=y, color=color, markers=markers, title=title)
    if show_values and isinstance(y, str):
        fig.update_traces(text=df[y], textposition="top center")
    fig.update_traces(hovertemplate="%{y}")
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(
        fig,
        y_is_currency=y_is_currency,
        y_is_percent=y_is_percent,
        currency=currency,
        decimals=decimals,
    )
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
):
    fig = px.area(df, x=x, y=y, color=color, title=title)
    if show_values:
        fig.update_traces(text=df[y], textposition="top center")
    fig.update_traces(hovertemplate="%{y}")
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(
        fig,
        y_is_currency=y_is_currency,
        y_is_percent=y_is_percent,
        currency=currency,
        decimals=decimals,
    )
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
    trendline: Optional[str] = None,
    show_legend: bool = True,
):
    fig = px.scatter(
        df, x=x, y=y, color=color, size=size, hover_data=hover_data, trendline=trendline, title=title
    )
    fig.update_traces(mode="markers", hovertemplate="%{y}")
    _apply_layout(fig, title, show_legend=show_legend)
    return _ensure_dark(fig)


def bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    *,
    color: Optional[str] = None,
    title: Optional[str] = None,
    orientation: str = "v",
    barmode: str = "group",
    y_is_currency: bool = False,
    y_is_percent: bool = False,
    currency: str = "₽",
    decimals: int = 0,
    show_legend: bool = True,
    show_values: bool = False,
):
    fig = px.bar(
        df,
        x=x if orientation == "v" else y,
        y=y if orientation == "v" else x,
        color=color,
        title=title,
    )
    if show_values and isinstance(y, str):
        fig.update_traces(text=df[y], textposition="auto")
    fig.update_traces(hovertemplate="%{y}")
    fig.update_layout(barmode=barmode)
    _apply_layout(fig, title, show_legend=show_legend)
    _apply_y_format(
        fig,
        y_is_currency=y_is_currency,
        y_is_percent=y_is_percent,
        currency=currency,
        decimals=decimals,
    )
    return _ensure_dark(fig)


def hist(
    df: pd.DataFrame,
    x: str,
    *,
    color: Optional[str] = None,
    nbins: Optional[int] = None,
    title: Optional[str] = None,
    show_legend: bool = True,
):
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
):
    fig = px.box(df, x=x, y=y, color=color, title=title, points="outliers")
    fig.update_traces(hovertemplate="%{y}")
    _apply_layout(fig, title, show_legend=show_legend)
    return _ensure_dark(fig)


def heatmap_pivot(
    pivot: pd.DataFrame, *, title: Optional[str] = None, colorscale: str = "Blues", show_values: bool = True
):
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            colorscale=colorscale,
            colorbar=dict(title=""),
            text=pivot.values if show_values else None,
            texttemplate="%{z}" if show_values else None,
            textfont=dict(color="#111"),
        )
    )
    _apply_layout(fig, title, show_legend=True)
    return _ensure_dark(fig)


# ---- Частые композиции ----
def waterfall(labels, values, title=None):
    measures = ["relative"] * (len(values) - 1) + ["total"]
    fig = go.Figure(
        go.Waterfall(
            name="",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            connector={"line": {"width": 1}},
        )
    )
    fig.update_traces(hovertemplate="%{y:.0f} ₽")
    _apply_layout(fig, title, show_legend=True)
    return _ensure_dark(fig)


def treemap_revenue(df, path_cols, value_col, color_col=None, title=None):
    fig = px.treemap(df, path=path_cols, values=value_col, color=color_col, title=title, color_continuous_scale="RdYlGn")
    _apply_layout(fig, title, show_legend=True)
    return _ensure_dark(fig)


def pareto(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    title: str = "Pareto 80/20 по выручке",
    *,
    top_n: int | None = 50,
    x_tickangle: int = -45,
    currency: str = "₽",
    show_currency: bool = True,
) -> go.Figure:
    """Pareto: столбики = метрика, линия = накопительный % (правая ось)."""
    d = (
        df.groupby(x_col, as_index=False)[value_col]
        .sum()
        .sort_values(value_col, ascending=False)
    )

    if d.empty:
        return _ensure_dark(go.Figure())

    # ограничение Top-N с "Прочее"
    if top_n and len(d) > top_n:
        head = d.iloc[:top_n].copy()
        tail_sum = d.iloc[top_n:][value_col].sum()
        if tail_sum > 0:
            head.loc[len(head)] = {x_col: "Прочее", value_col: tail_sum}
        d = head

    d["cum_pct"] = d[value_col].cumsum() / d[value_col].sum() * 100
    xcat = d[x_col].astype(str)  # критично: категорическая ось

    fig = go.Figure()

    # бары
    fig.add_bar(
        x=xcat,
        y=d[value_col],
        name=value_col,
        marker=dict(color="#d4a373"),
        hovertemplate=(f"%{{y:,.0f}} {currency}" if show_currency else "%{y:,.0f}"),
    )

    # линия накопленного процента (правая ось)
    fig.add_trace(
        go.Scatter(
            x=xcat,
            y=d["cum_pct"],
            mode="lines+markers",
            name="Накопительный %",
            yaxis="y2",
            line=dict(width=2.5, color="#8ab4f8"),
            marker=dict(size=4),
            hovertemplate="%{y:.1f} %",
        )
    )

    # горизонтальная отметка 80% по правой оси
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(xcat) - 0.5,
        y0=80,
        y1=80,
        yref="y2",
        line=dict(color="rgba(255,255,255,0.35)", width=1.5, dash="dash"),
    )

    fig.update_layout(
        xaxis=dict(
            type="category",
            categoryorder="array",
            categoryarray=list(xcat),
            tickangle=x_tickangle,
        ),
        yaxis=dict(title=value_col),
        yaxis2=dict(title="%", overlaying="y", side="right", range=[0, 100]),
        bargap=0.25,
    )

    _apply_layout(fig, title, show_legend=True)
    return _ensure_dark(fig)


def heatmap_calendar(pivot, title=None):
    fig = go.Figure(
        data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), coloraxis="coloraxis")
    )
    fig.update_layout(coloraxis=dict(colorscale="Blues"))
    _apply_layout(fig, title, show_legend=True)
    return _ensure_dark(fig)


def scatter_margin_returns(df, x="returns_pct", y="margin", color="category", hover=None, title=None):
    fig = px.scatter(df, x=x, y=y, color=color, hover_data=hover, title=title)
    _apply_layout(fig, title, show_legend=True)
    return _ensure_dark(fig)


def target_line(fig, x0, x1, y, name="План"):
    fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, line=dict(dash="dash", width=2), name=name)
    return fig


# ---- Алиасы совместимости ----
def line_chart(df, x, y, **kwargs):
    return line(df, x, y, **kwargs)


def bar_chart(df, x, y, **kwargs):
    return bar(df, x, y, **kwargs)