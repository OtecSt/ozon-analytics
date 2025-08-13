# data_loader.py
from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable


import pandas as pd
import numpy as np

# ---- Публичная нормализация SKU (совместимость с другими модулями) ----
def normalize_sku(s):
    """Приводит SKU/OZON ID к унифицированной строке без завершающих `.0` и пробелов.
    Безопасна к смешанным типам.
    """
    s = pd.Series(s, copy=False).astype(str).str.strip()
    s = s.str.replace(r"\.0$", "", regex=True)
    s = s.where(~s.eq("<NA>"), pd.NA)
    return s


# ---- Конфигурация ----

# Папка, куда складывается GOLD-слой. Можно переопределить: export GOLD_DIR=/path/to/gold
GOLD_DIR = Path(os.getenv("GOLD_DIR", "gold")).resolve()

# Базовые имена файлов, которые мы ожидаем встретить в GOLD
# Первые четыре — из build_gold.py, остальные — опционально (из планировщика)
KNOWN_FILES = {
    "fact_sku_daily.csv",
    "fact_sku_monthly.csv",
    "mart_unit_econ.csv",
    "data_dictionary.csv",
    "forecast_sku_monthly.csv",          # optional
    "production_justification.csv",      # optional
    "assumptions.csv",                    # optional
    "backtest.csv",                       # optional
}


# ---- Утилиты нормализации ----

def _unify_sku(series: pd.Series) -> pd.Series:
    """
    Нормализует артикулы/SKU:
    - пытаемся привести к Int64 (чтобы "00123" и "123.0" стали 123),
    - затем обратно в строку без пробелов,
    - "<NA>" -> pandas.NA.
    Безопасна к смешанным типам.
    """
    if series is None or len(series) == 0:
        return series
    return normalize_sku(series)


def _to_period_str(s: pd.Series) -> pd.Series:
    """
    Приводит столбец периодов к строке вида YYYY-MM:
    - принимает Period, datetime, строку; ошибки -> NA;
    - всегда возвращает dtype=object со строками YYYY-MM.
    """
    if s is None or s.empty:
        return s
    # Если уже похоже на YYYY-MM, просто обрежем пробелы
    raw = s.astype(str).str.strip()
    # Попробуем превратить в даты, затем в период M
    dt = pd.to_datetime(raw, errors="coerce")
    per = dt.dt.to_period("M")
    out = per.astype(str)
    # Где не распарсили — оставим исходное (вдруг уже YYYY-MM), затем снова фильтр
    out = out.where(~out.isin(["NaT", "<NA>"]), raw)
    # Нормализуем окончательно: всё, что не match YYYY-MM, в NA
    mask = out.str.match(r"^\d{4}-\d{2}$", na=False)
    out = out.where(mask, pd.NA)
    return out


def _to_datetime(s: pd.Series) -> pd.Series:
    """Мягко преобразует к datetime[ns]; ошибки -> NaT."""
    if s is None or s.empty:
        return s
    return pd.to_datetime(s, errors="coerce")


def _basic_hygiene(df: pd.DataFrame) -> pd.DataFrame:
    """Базовая гигиена над таблицей: нормализация sku/period/date где есть."""
    if df.empty:
        return df

    out = df.copy()

    # Aliases: если есть колонка OZON ID — переименуем в sku (если sku ещё нет)
    if "OZON ID" in out.columns and "sku" not in out.columns:
        out.rename(columns={"OZON ID": "sku"}, inplace=True)

    # SKU
    if "sku" in out.columns:
        out["sku"] = normalize_sku(out["sku"])  # публичная нормализация (совместимо с другими модулями)

    # Period (месячная гранулярность) — встречается в fact_sku_monthly / forecast
    if "period" in out.columns:
        out["period"] = _to_period_str(out["period"])

    # Date (дневная гранулярность) — fact_sku_daily
    if "date" in out.columns:
        out["date"] = _to_datetime(out["date"])

    return out


# ---- Чтение файлов ----

def _read_csv(path: Path) -> pd.DataFrame:
    """
    Безопасно читает CSV с utf-8-sig и low_memory=False.
    При отсутствии файла возвращает пустой DataFrame.
    """
    if not path.exists():
        return pd.DataFrame()
    # первичная попытка — стандартный CSV
    try:
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    except Exception as e:
        print(f"[data_loader] Warning: failed to read {path.name}: {e}")
        return pd.DataFrame()

    # эвристика: если спарсили в одну колонку, а в файле явно есть ';' — перечитываем с sep=';'
    try:
        if df.shape[1] == 1:
            with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
                head = f.read(4096)
            if ";" in head:
                try:
                    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, sep=";")
                except Exception:
                    df = pd.read_csv(path, encoding="utf-8", low_memory=False, sep=";")
    except Exception:
        # не критично — оставляем как есть
        pass

    return df


@lru_cache(maxsize=64)
def load_table(name: str) -> pd.DataFrame:
    """
    Загружает ОДНУ таблицу из GOLD_DIR по имени файла, с нормализацией.
    Кэшируется. Для обновления см. clear_cache().
    """
    path = GOLD_DIR / name
    df = _read_csv(path)
    return _basic_hygiene(df)


def list_available() -> Iterable[str]:
    """Возвращает имена известных CSV, которые реально присутствуют в GOLD_DIR."""
    if not GOLD_DIR.exists():
        return []
    present = set(p.name for p in GOLD_DIR.glob("*.csv"))
    return sorted(KNOWN_FILES.intersection(present))


def clear_cache() -> None:
    """Сбрасывает кэш чтения CSV (если перезаписал файлы — дерни это)."""
    load_table.cache_clear()  # type: ignore[attr-defined]


# ---- Главный вход: загрузить всё разом ----

@lru_cache(maxsize=1)
def load_all() -> Dict[str, pd.DataFrame]:
    """
    Читает все ключевые CSV из GOLD_DIR и возвращает словарь:
      - daily              -> fact_sku_daily.csv
      - monthly            -> fact_sku_monthly.csv
      - mart               -> mart_unit_econ.csv
      - dd                 -> data_dictionary.csv
      - forecast           -> forecast_sku_monthly.csv (optional)
      - prodjust           -> production_justification.csv (optional)
      - assumptions        -> assumptions.csv (optional)
      - backtest           -> backtest.csv (optional)

    Все датафреймы проходят мягкую нормализацию (sku/period/date).
    """
    GOLD_DIR.mkdir(parents=True, exist_ok=True)  # на всякий случай

    daily = load_table("fact_sku_daily.csv")
    monthly = load_table("fact_sku_monthly.csv")
    mart = load_table("mart_unit_econ.csv")
    dd = load_table("data_dictionary.csv")

    # опциональные
    forecast = load_table("forecast_sku_monthly.csv")
    prodjust = load_table("production_justification.csv")
    assumptions = load_table("assumptions.csv")
    backtest = load_table("backtest.csv")

    # Доп. гигиена: sku -> str.strip() (после унификации)
    for df in (daily, monthly, mart, forecast, prodjust):
        if not df.empty and "sku" in df.columns:
            df["sku"] = df["sku"].astype(str).str.strip()

    # Приведём типы числовых полей в daily/monthly при наличии
    # (мягко: ничего не ломаем, просто to_numeric с coerce)
    def _num(df: pd.DataFrame, cols: Iterable[str]) -> None:
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    if not daily.empty:
        _num(daily, ["shipped_qty", "returns_qty", "promo_rub", "order_value_rub_sum", "returns_rub", "shipments"])

    if not monthly.empty:
        _num(monthly, ["shipped_qty", "returns_qty", "promo_rub", "order_value_rub_sum", "returns_rub"])

    return dict(
        daily=daily,
        monthly=monthly,
        mart=mart,
        dd=dd,
        forecast=forecast,
        prodjust=prodjust,
        assumptions=assumptions,
        backtest=backtest,
    )


# ---- Лёгкая валидация (необязательно, но полезно) ----

def validate_dictionary(verbose: bool = True) -> pd.DataFrame:
    """
    Сверяет data_dictionary.csv с реально загруженными таблицами.
    Возвращает фрейм c несовпадениями (если есть).
    """
    data = load_all()
    dd = data.get("dd", pd.DataFrame())
    if dd.empty or not {"table", "column"}.issubset(dd.columns):
        if verbose:
            print("[data_loader] data_dictionary.csv пустой или не имеет нужных колонок.")
        return pd.DataFrame()

    problems = []
    tables_map = {
        "fact_sku_daily": data["daily"],
        "fact_sku_monthly": data["monthly"],
        "mart_unit_econ": data["mart"],
    }
    for tbl, df in tables_map.items():
        if df.empty:
            continue
        dd_tbl = dd[dd["table"] == tbl]
        if dd_tbl.empty:
            continue
        dd_cols = set(dd_tbl["column"].astype(str))
        real_cols = set(map(str, df.columns))
        missing = dd_cols - real_cols
        extra = real_cols - dd_cols
        if missing or extra:
            problems.append({"table": tbl, "missing_in_data": sorted(missing), "not_in_dictionary": sorted(extra)})

    return pd.DataFrame(problems)


# ---- Удобные алиасы (необязательно) ----

def get_daily() -> pd.DataFrame:
    return load_all()["daily"].copy()

def get_monthly() -> pd.DataFrame:
    return load_all()["monthly"].copy()

def get_mart() -> pd.DataFrame:
    return load_all()["mart"].copy()

def get_forecast() -> pd.DataFrame:
    return load_all()["forecast"].copy()

def get_prodjust() -> pd.DataFrame:
    return load_all()["prodjust"].copy()

def get_assumptions() -> pd.DataFrame:
    return load_all()["assumptions"].copy()

def get_backtest() -> pd.DataFrame:
    return load_all()["backtest"].copy()