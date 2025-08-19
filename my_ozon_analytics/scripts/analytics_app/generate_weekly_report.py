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
import io


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



# ========== ВАЛИДАЦИЯ GOLD (акт сверки) ==========

def _rel_delta(a: float, b: float) -> float:
    try:
        if b == 0:
            return 0.0 if a == 0 else 100.0
        return (a - b) / b * 100.0
    except Exception:
        return 0.0


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_gold_tables(gold_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (daily, monthly, mart). Отсутствующие файлы заменяются пустыми df."""
    def _safe_read_csv(p: Path) -> pd.DataFrame:
        try:
            return pd.read_csv(p, encoding="utf-8-sig", low_memory=False)
        except Exception:
            return pd.DataFrame()

    daily = _safe_read_csv(gold_dir / "fact_sku_daily.csv")
    monthly = _safe_read_csv(gold_dir / "fact_sku_monthly.csv")
    mart = _safe_read_csv(gold_dir / "mart_unit_econ.csv")
    return daily, monthly, mart


def _normalize_period_monthly(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "period" not in df.columns:
        return pd.DataFrame()
    out = df.copy()
    out["period"] = out["period"].astype(str)
    return out


def run_validation(
    gold_dir: Path,
    reference_path: Path | None = None,
    *,
    threshold_pct: float = 1.0,
    out_dir: Path = Path("reports/validation")
) -> dict:
    """
    Проводит 3 проверки:
      1) Акт сверки с эталоном (если задан reference_path).
      2) Список SKU без себестоимости (production_cost_per_unit NaN/0).
      3) Несоответствие daily vs monthly по месяцам (> threshold_pct).

    Возвращает словарь с путями к отчётам и краткой сводкой.
    """
    _ensure_dir(out_dir)

    daily, monthly, mart = _load_gold_tables(Path(gold_dir))
    monthly = _normalize_period_monthly(monthly)

    summary = {"threshold_pct": threshold_pct, "issues": []}
    log_lines = []
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_lines.append(f"=== VALIDATION START · {ts} ===")

    # --- 1) Сверка с эталоном ---
    ref_totals = None
    if reference_path and Path(reference_path).exists():
        try:
            if str(reference_path).lower().endswith((".xls", ".xlsx")):
                ref = pd.read_excel(reference_path)
            else:
                ref = pd.read_csv(reference_path, encoding="utf-8-sig", low_memory=False)
            # допустимые имена
            candidates = {
                "revenue": ["total_rev", "revenue", "gross_revenue", "order_value_rub_sum"],
                "qty": ["total_qty", "qty", "shipped_qty"],
                "returns_rub": ["returns_rub", "returns_amount"],
                "returns_qty": ["returns_qty"],
                "cogs": ["cogs", "production_cost_total"],
                "margin": ["margin", "profit"],
            }
            def pick_sum(df, names):
                for n in names:
                    if n in df.columns:
                        return float(pd.to_numeric(df[n], errors="coerce").fillna(0).sum())
                return 0.0
            ref_totals = {k: pick_sum(ref, v) for k, v in candidates.items()}
        except Exception as e:
            log_lines.append(f"[WARN] Не удалось прочитать эталон: {e}")

    gold_totals = {
        "revenue": float(pd.to_numeric(daily.get("order_value_rub_sum", pd.Series(dtype=float)), errors="coerce").sum()) if not daily.empty else 0.0,
        "qty": float(pd.to_numeric(daily.get("shipped_qty", pd.Series(dtype=float)), errors="coerce").sum()) if not daily.empty else 0.0,
        "returns_rub": float(pd.to_numeric(daily.get("returns_rub", pd.Series(dtype=float)), errors="coerce").sum()) if not daily.empty else 0.0,
        "returns_qty": float(pd.to_numeric(daily.get("returns_qty", pd.Series(dtype=float)), errors="coerce").sum()) if not daily.empty else 0.0,
        "cogs": float(pd.to_numeric(mart.get("cogs", pd.Series(dtype=float)), errors="coerce").sum()) if not mart.empty else 0.0,
        "margin": float(pd.to_numeric(mart.get("margin", pd.Series(dtype=float)), errors="coerce").sum()) if not mart.empty else 0.0,
    }

    if ref_totals is not None:
        log_lines.append("-- Акт сверки с эталоном --")
        for k in ["revenue", "qty", "returns_rub", "returns_qty", "cogs", "margin"]:
            rd = _rel_delta(gold_totals.get(k, 0.0), ref_totals.get(k, 0.0))
            ok = abs(rd) <= threshold_pct
            status = "OK" if ok else "FAIL"
            log_lines.append(f"{k}: gold={gold_totals.get(k,0.0):,.2f} vs ref={ref_totals.get(k,0.0):,.2f} · Δ%={rd:.2f} · {status}")
            if not ok:
                summary["issues"].append({"metric": k, "delta_pct": rd})

    # --- 2) SKU без себестоимости ---
    missing_path = out_dir / "missing_cogs.csv"
    missing_count = 0
    if not mart.empty:
        if "production_cost_per_unit" in mart.columns:
            mask = mart["production_cost_per_unit"].isna() | (pd.to_numeric(mart["production_cost_per_unit"], errors="coerce").fillna(0) <= 0)
            missing = mart.loc[mask, [c for c in ["sku", "production_cost_per_unit", "total_qty", "total_rev"] if c in mart.columns]].copy()
            if not missing.empty:
                missing.to_csv(missing_path, index=False, encoding="utf-8-sig")
                missing_count = len(missing)
                log_lines.append(f"[WARN] SKU без себестоимости: {missing_count} · {missing_path}")
            else:
                log_lines.append("SKU без себестоимости: 0")
        else:
            log_lines.append("[INFO] В mart_unit_econ.csv нет колонки production_cost_per_unit")

    # --- 3) Daily vs Monthly ---
    dvm_path = out_dir / "daily_vs_monthly_mismatches.csv"
    dvm_count = 0
    if not daily.empty and not monthly.empty and {"date", "sku"}.issubset(daily.columns) and {"period", "sku"}.issubset(monthly.columns):
        d = daily.copy()
        d["date"] = pd.to_datetime(d["date"], errors="coerce")
        d["period"] = d["date"].dt.to_period("M").astype(str)
        grp_d = (
            d.groupby(["period", "sku"], as_index=False)
             .agg(shipped_qty=("shipped_qty", "sum"), order_value_rub_sum=("order_value_rub_sum", "sum"))
        )
        m = monthly.copy()
        m["period"] = m["period"].astype(str)
        join = grp_d.merge(m[[c for c in ["period", "sku", "shipped_qty", "sales_rub", "order_value_rub_sum"] if c in m.columns]],
                           on=["period", "sku"], how="outer", suffixes=("_daily", "_monthly"))
        if "order_value_rub_sum_monthly" not in join.columns and "sales_rub_monthly" in join.columns:
            join["order_value_rub_sum_monthly"] = join["sales_rub_monthly"]
        for a, b in [("shipped_qty_daily", "shipped_qty_monthly"), ("order_value_rub_sum_daily", "order_value_rub_sum_monthly")]:
            if a in join.columns and b in join.columns:
                join[a] = pd.to_numeric(join[a], errors="coerce").fillna(0)
                join[b] = pd.to_numeric(join[b], errors="coerce").fillna(0)
        join["delta_qty_pct"] = join.apply(lambda r: _rel_delta(r.get("shipped_qty_daily", 0), r.get("shipped_qty_monthly", 0)), axis=1)
        join["delta_rev_pct"] = join.apply(lambda r: _rel_delta(r.get("order_value_rub_sum_daily", 0), r.get("order_value_rub_sum_monthly", 0)), axis=1)
        mism = join[(join["delta_qty_pct"].abs() > threshold_pct) | (join["delta_rev_pct"].abs() > threshold_pct)].copy()
        if not mism.empty:
            mism.to_csv(dvm_path, index=False, encoding="utf-8-sig")
            dvm_count = len(mism)
            log_lines.append(f"[WARN] Несоответствия daily vs monthly: {dvm_count} · {dvm_path}")
        else:
            log_lines.append("Daily vs Monthly: расхождений нет")

    # --- Завершение ---
    log_path = out_dir / "validation_log.txt"
    with log_path.open("a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")
        f.write("=== VALIDATION END ===\n\n")

    summary.update({
        "log": str(log_path),
        "missing_cogs_csv": str(missing_path) if missing_count else None,
        "dvm_csv": str(dvm_path) if dvm_count else None,
        "missing_cogs_count": int(missing_count),
        "dvm_count": int(dvm_count),
        "gold_totals": gold_totals,
        "ref_totals": ref_totals,
    })
    return summary

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
        # Write sheets
        report["summary"].to_excel(xw, sheet_name="Summary", index=False)
        report["wow"].to_excel(xw, sheet_name="WoW", index=False)
        report["by_sku"].to_excel(xw, sheet_name="By_SKU", index=False)
        report["by_day"].to_excel(xw, sheet_name="By_Day", index=False)
        report["top_by_revenue"].to_excel(xw, sheet_name="Top_Revenue", index=False)
        report["top_by_shipped"].to_excel(xw, sheet_name="Top_Shipped", index=False)

        wb = xw.book
        fmt_int = wb.add_format({"num_format": "#,##0"})
        fmt_money = wb.add_format({"num_format": "#,##0"})
        fmt_pct = wb.add_format({"num_format": "0.0%"})
        fmt_date = wb.add_format({"num_format": "yyyy-mm-dd"})

        # Helper to apply format by column names
        def _fmt_sheet(sheet_name: str, df: pd.DataFrame, int_cols=(), money_cols=(), pct_cols=(), date_cols=()):
            ws = xw.sheets.get(sheet_name)
            if ws is None or df is None or df.empty:
                return
            # set a reasonable default width
            ws.set_column(0, len(df.columns) - 1, 14)
            for col in int_cols:
                if col in df.columns:
                    idx = df.columns.get_loc(col)
                    ws.set_column(idx, idx, 14, fmt_int)
            for col in money_cols:
                if col in df.columns:
                    idx = df.columns.get_loc(col)
                    ws.set_column(idx, idx, 14, fmt_money)
            for col in pct_cols:
                if col in df.columns:
                    idx = df.columns.get_loc(col)
                    ws.set_column(idx, idx, 12, fmt_pct)
            for col in date_cols:
                if col in df.columns:
                    idx = df.columns.get_loc(col)
                    ws.set_column(idx, idx, 12, fmt_date)

        _fmt_sheet(
            "By_SKU",
            report["by_sku"],
            int_cols=("shipped_qty", "returns_qty", "shipments"),
            money_cols=("order_value_rub", "promo_rub", "aov_rub"),
            pct_cols=("return_rate_%", "promo_intensity_%"),
        )
        _fmt_sheet(
            "By_Day",
            report["by_day"],
            int_cols=("shipped_qty", "returns_qty"),
            money_cols=("order_value_rub", "promo_rub"),
            pct_cols=("return_rate_%",),
            date_cols=("date",),
        )
        _fmt_sheet(
            "Top_Revenue",
            report["top_by_revenue"],
            int_cols=("shipped_qty", "returns_qty", "shipments"),
            money_cols=("order_value_rub", "promo_rub", "aov_rub"),
            pct_cols=("return_rate_%", "promo_intensity_%"),
        )
        _fmt_sheet(
            "Top_Shipped",
            report["top_by_shipped"],
            int_cols=("shipped_qty", "returns_qty", "shipments"),
            money_cols=("order_value_rub", "promo_rub", "aov_rub"),
            pct_cols=("return_rate_%", "promo_intensity_%"),
        )


# Helper: Excel as bytes for Streamlit
def build_weekly_excel_bytes(report: dict) -> bytes:
    """Собирает weekly-отчёт в XLSX и возвращает как bytes (для Streamlit download_button)."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        report["summary"].to_excel(xw, sheet_name="Summary", index=False)
        report["wow"].to_excel(xw, sheet_name="WoW", index=False)
        report["by_sku"].to_excel(xw, sheet_name="By_SKU", index=False)
        report["by_day"].to_excel(xw, sheet_name="By_Day", index=False)
        report["top_by_revenue"].to_excel(xw, sheet_name="Top_Revenue", index=False)
        report["top_by_shipped"].to_excel(xw, sheet_name="Top_Shipped", index=False)
    buf.seek(0)
    return buf.getvalue()



def build_markdown_string(report: dict) -> str:
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
    md.append(_safe_to_markdown(t1.head(50)))
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

    return "\n".join(md)

def save_markdown(report: dict, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_markdown_string(report), encoding="utf-8")


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
    ap.add_argument("--validate", action="store_true", help="Запустить валидацию GOLD после построения отчёта")
    ap.add_argument("--gold-dir", default="gold", help="Папка с GOLD-слоем (fact_sku_daily.csv, fact_sku_monthly.csv, mart_unit_econ.csv)")
    ap.add_argument("--reference", default=None, help="Путь к эталонному файлу (ultimate_report_unit.xlsx/.csv)")
    ap.add_argument("--thr", type=float, default=1.0, help="Порог допустимого отклонения, % (по умолчанию 1.0)")
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

    # Валидация GOLD (по желанию)
    if args.validate:
        v = run_validation(Path(args.gold_dir), Path(args.reference) if args.reference else None, threshold_pct=float(args.thr))
        print("\n=== Validation summary ===")
        print(f"Log: {v['log']}")
        if v.get("missing_cogs_csv"):
            print(f"Missing COGS CSV: {v['missing_cogs_csv']} (rows={v['missing_cogs_count']})")
        if v.get("dvm_csv"):
            print(f"Daily vs Monthly mismatches: {v['dvm_csv']} (rows={v['dvm_count']})")
        gt = v.get("gold_totals", {})
        rt = v.get("ref_totals", {})
        if rt:
            print("Totals (gold vs ref):")
            for k in ["revenue", "qty", "returns_rub", "returns_qty", "cogs", "margin"]:
                print(f"  {k}: {gt.get(k,0):,.2f} vs {rt.get(k,0):,.2f}")


if __name__ == "__main__":
    main()