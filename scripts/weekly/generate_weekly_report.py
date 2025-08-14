#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_weekly_report.py
Еженедельный отчёт по фактам (GOLD: fact_sku_daily.csv).

Вход (fact_sku_daily.csv) — как в build_gold.py:
- date (YYYY-MM-DD)
- sku
- shipped_qty (int)
- promo_rub (float)
- order_value_rub_sum (float)
- shipments (int)
- returns_qty (int)
- returns_rub (float)

Что делает:
- Берёт прошлую завершённую неделю (пн–вс) относительно "сегодня" (или заданной --as-of).
- Фильтрует daily и агрегирует:
  * Summary (KPI недели + WoW сопоставление с предыдущей неделей)
  * По SKU (лидерборд, метрики: отправлено, возвраты, % возвратов, выручка, промо, интенсивность промо, отправления, AOV)
  * По дням (динамика за неделю)
  * Топ-N SKU по выручке и по отправленным штукам
- Сохраняет Excel и Markdown.

Пример:
python generate_weekly_report.py \
  --input gold/fact_sku_daily.csv \
  --output-dir output/weekly \
  --top 15

Опционально:
  --as-of 2025-08-12 (якорная дата; неделя берётся предыдущая, пн–вс)
  --start 2025-08-04 --end 2025-08-10 (явный период)
  --only xlsx  (или --only md)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ========== Утилиты дат ==========

def _last_completed_week(today: date) -> tuple[date, date]:
    """Возвращает (start,end) предыдущей завершённой ISO-недели (пн–вс)."""
    # понедельник текущей недели:
    monday_this = today - timedelta(days=today.weekday())
    # конец прошлой недели — воскресенье:
    end = monday_this - timedelta(days=1)
    start = end - timedelta(days=6)
    return start, end


def _coerce_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


# ========== Метрики/форматирование ==========

@dataclass
class Totals:
    shipped_qty: int = 0
    returns_qty: int = 0
    promo_rub: float = 0.0
    returns_rub: float = 0.0
    order_value_rub_sum: float = 0.0
    shipments: int = 0

    @property
    def return_rate_pct(self) -> float:
        return (self.returns_qty / self.shipped_qty * 100.0) if self.shipped_qty else 0.0

    @property
    def promo_intensity_pct(self) -> float:
        return (self.promo_rub / self.order_value_rub_sum * 100.0) if self.order_value_rub_sum else 0.0

    @property
    def net_revenue(self) -> float:
        return self.order_value_rub_sum - self.returns_rub

    @property
    def aov(self) -> float:
        return (self.order_value_rub_sum / self.shipments) if self.shipments else 0.0


def _fmt_int(x: float | int) -> str:
    try:
        return f"{int(round(float(x))):,}".replace(",", " ")
    except Exception:
        return "0"


def _fmt_money(x: float, currency: str = "₽") -> str:
    try:
        return f"{x:,.0f}".replace(",", " ") + f" {currency}"
    except Exception:
        return f"0 {currency}"


def _fmt_pct(x: float) -> str:
    try:
        return f"{x:.1f}%"
    except Exception:
        return "0.0%"


def _safe_to_markdown(df: pd.DataFrame) -> str:
    try:
        # требует tabulate; если нет — упадём в except
        return df.to_markdown(index=False)
    except Exception:
        # простой pipe-table без табличных библиотек
        cols = list(df.columns)
        lines = []
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, r in df.iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        return "\n".join(lines)


# ========== Основная логика отчёта ==========

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    # делаем отсутствующие колонки нулями/пустыми — на случай старых выгрузок
    needed = {
        "date": None,
        "sku": "",
        "shipped_qty": 0,
        "promo_rub": 0.0,
        "order_value_rub_sum": 0.0,
        "shipments": 0,
        "returns_qty": 0,
        "returns_rub": 0.0,
    }
    for c, fill in needed.items():
        if c not in df.columns:
            df[c] = fill
    return df


def _read_fact(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    df = _ensure_columns(df)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    # типы
    for c in ["shipped_qty", "returns_qty", "shipments"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["promo_rub", "order_value_rub_sum", "returns_rub"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["sku"] = df["sku"].astype(str).str.strip()
    return df


def _period_slice(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    mask = (df["date"] >= start) & (df["date"] <= end)
    return df.loc[mask].copy()


def _totals_from_df(df: pd.DataFrame) -> Totals:
    if df.empty:
        return Totals()
    agg = df[["shipped_qty", "returns_qty", "promo_rub", "returns_rub", "order_value_rub_sum", "shipments"]].sum()
    return Totals(
        shipped_qty=int(agg.get("shipped_qty", 0)),
        returns_qty=int(agg.get("returns_qty", 0)),
        promo_rub=float(agg.get("promo_rub", 0.0)),
        returns_rub=float(agg.get("returns_rub", 0.0)),
        order_value_rub_sum=float(agg.get("order_value_rub_sum", 0.0)),
        shipments=int(agg.get("shipments", 0)),
    )


def _by_sku(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "sku", "shipped_qty", "returns_qty", "return_rate_%", "order_value_rub",
            "promo_rub", "promo_intensity_%", "shipments", "aov_rub"
        ])
    g = (
        df.groupby("sku", as_index=False)
          .agg(
              shipped_qty=("shipped_qty", "sum"),
              returns_qty=("returns_qty", "sum"),
              order_value_rub=("order_value_rub_sum", "sum"),
              promo_rub=("promo_rub", "sum"),
              shipments=("shipments", "sum"),
          )
    )
    g["return_rate_%"] = (g["returns_qty"] / g["shipped_qty"] * 100.0).where(g["shipped_qty"] > 0, 0.0)
    g["promo_intensity_%"] = (g["promo_rub"] / g["order_value_rub"] * 100.0).where(g["order_value_rub"] > 0, 0.0)
    g["aov_rub"] = (g["order_value_rub"] / g["shipments"]).where(g["shipments"] > 0, 0.0)
    # сорт по выручке, затем по отгрузкам
    g = g.sort_values(["order_value_rub", "shipped_qty"], ascending=False).reset_index(drop=True)
    return g


def _by_day(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[
            "date", "shipped_qty", "returns_qty", "return_rate_%", "order_value_rub", "promo_rub"
        ])
    d = (
        df.groupby("date", as_index=False)
          .agg(
              shipped_qty=("shipped_qty", "sum"),
              returns_qty=("returns_qty", "sum"),
              order_value_rub=("order_value_rub_sum", "sum"),
              promo_rub=("promo_rub", "sum"),
          )
          .sort_values("date")
          .reset_index(drop=True)
    )
    d["return_rate_%"] = (d["returns_qty"] / d["shipped_qty"] * 100.0).where(d["shipped_qty"] > 0, 0.0)
    return d


def _wow_summary(now: Totals, prev: Totals) -> pd.DataFrame:
    rows = []
    def add_row(name: str, cur: float, old: float, kind: str):
        delta = cur - old
        pct = (delta / old * 100.0) if old else 0.0
        if kind == "int":
            cur_s, old_s, delta_s = _fmt_int(cur), _fmt_int(old), _fmt_int(delta)
        elif kind == "money":
            cur_s, old_s, delta_s = _fmt_money(cur), _fmt_money(old), _fmt_money(delta)
        else:
            cur_s, old_s, delta_s = _fmt_pct(cur), _fmt_pct(old), _fmt_pct(delta)
        rows.append({
            "metric": name,
            "current": cur_s,
            "previous": old_s,
            "Δ": delta_s,
            "Δ%": _fmt_pct(pct),
        })

    add_row("Отгружено, шт.", now.shipped_qty, prev.shipped_qty, "int")
    add_row("Возвраты, шт.", now.returns_qty, prev.returns_qty, "int")
    add_row("Доля возвратов", now.return_rate_pct, prev.return_rate_pct, "pct")
    add_row("Выручка (order_value), ₽", now.order_value_rub_sum, prev.order_value_rub_sum, "money")
    add_row("Чистая выручка (минус возвраты), ₽", now.net_revenue, prev.net_revenue, "money")
    add_row("Промо, ₽", now.promo_rub, prev.promo_rub, "money")
    add_row("Интенсивность промо", now.promo_intensity_pct, prev.promo_intensity_pct, "pct")
    add_row("Отправления, шт.", now.shipments, prev.shipments, "int")
    add_row("AOV, ₽", now.aov, prev.aov, "money")
    return pd.DataFrame(rows)


def build_report(fact_daily: pd.DataFrame, start: date, end: date, top_n: int = 15) -> dict:
    week = _period_slice(fact_daily, start, end)
    prev_start, prev_end = start - timedelta(days=7), start - timedelta(days=1)
    week_prev = _period_slice(fact_daily, prev_start, prev_end)

    cur_tot = _totals_from_df(week)
    prev_tot = _totals_from_df(week_prev)

    summary = pd.DataFrame({
        "Период": [f"{start} — {end}"],
        "Отгружено, шт.": [_fmt_int(cur_tot.shipped_qty)],
        "Возвраты, шт.": [_fmt_int(cur_tot.returns_qty)],
        "Доля возвратов": [_fmt_pct(cur_tot.return_rate_pct)],
        "Выручка (order_value), ₽": [_fmt_money(cur_tot.order_value_rub_sum)],
        "Чистая выручка, ₽": [_fmt_money(cur_tot.net_revenue)],
        "Промо, ₽": [_fmt_money(cur_tot.promo_rub)],
        "Интенсивность промо": [_fmt_pct(cur_tot.promo_intensity_pct)],
        "Отправления, шт.": [_fmt_int(cur_tot.shipments)],
        "AOV, ₽": [_fmt_money(cur_tot.aov)],
    })

    wow = _wow_summary(cur_tot, prev_tot)
    bysku = _by_sku(week)
    byday = _by_day(week)

    top_by_revenue = bysku.nlargest(top_n, "order_value_rub").copy()
    top_by_revenue["order_value_rub"] = top_by_revenue["order_value_rub"].round(0)
    top_by_shipped = bysku.nlargest(top_n, "shipped_qty").copy()

    return {
        "period": (start, end),
        "summary": summary,
        "wow": wow,
        "by_sku": bysku,
        "by_day": byday,
        "top_by_revenue": top_by_revenue,
        "top_by_shipped": top_by_shipped,
    }


# ========== Экспорт ==========

def save_excel(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out_path, engine="xlsxwriter") as xw:
        report["summary"].to_excel(xw, sheet_name="Summary", index=False)
        report["wow"].to_excel(xw, sheet_name="WoW", index=False)
        report["by_sku"].to_excel(xw, sheet_name="By_SKU", index=False)
        report["by_day"].to_excel(xw, sheet_name="By_Day", index=False)
        report["top_by_revenue"].to_excel(xw, sheet_name="Top_Revenue", index=False)
        report["top_by_shipped"].to_excel(xw, sheet_name="Top_Shipped", index=False)
        # apply basic formats
        wb = xw.book
        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_money = wb.add_format({"num_format": "#,##0"})
        for sheet in ("By_SKU", "By_Day", "Top_Revenue", "Top_Shipped"):
            ws = xw.sheets[sheet]
            # попытка найти типичные столбцы и применить формат
            for col_idx, col_name in enumerate(ws.get_default_row_height() or []):
                pass  # безопасная заглушка, реального API для имён колонок тут нет


def save_markdown(report: dict, out_path: Path) -> None:
    start, end = report["period"]
    md = []
    md.append(f"# Еженедельный отчёт · {start} — {end}\n")

    md.append("## Итоги недели\n")
    md.append(_safe_to_markdown(report["summary"]))
    md.append("\n")

    md.append("## Неделя к неделе (WoW)\n")
    md.append(_safe_to_markdown(report["wow"]))
    md.append("\n")

    md.append("## Топ SKU по выручке\n")
    t1 = report["top_by_revenue"].copy()
    t1["order_value_rub"] = t1["order_value_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    t1["promo_rub"] = t1["promo_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    t1["aov_rub"] = t1["aov_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    md.append(_safe_to_markdown(t1.head(50)))  # чтобы не распухало
    md.append("\n")

    md.append("## Топ SKU по отгрузкам\n")
    t2 = report["top_by_shipped"].copy()
    t2["order_value_rub"] = t2["order_value_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    t2["promo_rub"] = t2["promo_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    t2["aov_rub"] = t2["aov_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    md.append(_safe_to_markdown(t2.head(50)))
    md.append("\n")

    md.append("## Динамика по дням\n")
    d = report["by_day"].copy()
    d["order_value_rub"] = d["order_value_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    d["promo_rub"] = d["promo_rub"].map(lambda v: _fmt_money(float(v)).replace(" ₽",""))
    md.append(_safe_to_markdown(d))
    md.append("\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(md), encoding="utf-8")


# ========== CLI ==========

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Еженедельный отчёт из fact_sku_daily.csv (прошлая завершённая неделя)")
    ap.add_argument("--input", required=True, help="Путь к fact_sku_daily.csv")
    ap.add_argument("--output-dir", default="output/weekly", help="Куда сохранять файлы")
    ap.add_argument("--as-of", default=None, help="Якорная дата (YYYY-MM-DD). Неделя берётся предыдущая, пн–вс.")
    ap.add_argument("--start", default=None, help="Начало периода (YYYY-MM-DD). Если задано, используется вместе с --end/авто +6 дней.")
    ap.add_argument("--end", default=None, help="Конец периода (YYYY-MM-DD). Если не задан, берём start+6.")
    ap.add_argument("--top", type=int, default=15, help="Сколько SKU показывать в ТОПах")
    ap.add_argument("--only", choices=["xlsx", "md"], default=None, help="Сохранить только один формат")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Определяем период
    if args.start:
        start = _coerce_date(args.start)
        end = _coerce_date(args.end) if args.end else (start + timedelta(days=6))
    else:
        today = _coerce_date(args.as_of) if args.as_of else date.today()
        start, end = _last_completed_week(today)

    in_path = Path(args.input)
    if in_path.is_dir():
        candidate = in_path / "fact_sku_daily.csv"
        if not candidate.exists():
            raise FileNotFoundError(f"В папке {in_path} не найден fact_sku_daily.csv (GOLD)")
        in_path = candidate
    if not in_path.exists():
        raise FileNotFoundError(f"Файл не найден: {in_path}")
    fact = _read_fact(in_path)
    report = build_report(fact, start, end, top_n=int(args.top))

    print(f"Период: {start} — {end}")
    print(f"Строк в исходном daily: {len(fact):,}")
    print(f"Строк в недельном срезе: {len(report['by_day']):,}")
    print(f"SKU в недельном срезе: {report['by_sku']['sku'].nunique() if not report['by_sku'].empty else 0:,}")

    out_dir = Path(args.output_dir)
    period_tag = f"{start}_{end}"
    xlsx_path = out_dir / f"weekly_{period_tag}.xlsx"
    md_path = out_dir / f"weekly_{period_tag}.md"

    if args.only in (None, "xlsx"):
        save_excel(report, xlsx_path)
        print(f"✔ Excel сохранён: {xlsx_path.resolve()}")

    if args.only in (None, "md"):
        save_markdown(report, md_path)
        print(f"✔ Markdown сохранён: {md_path.resolve()}")


if __name__ == "__main__":
    main()