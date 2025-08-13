#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_planner_ru.py — консольный запуск планировщика на русском.

Использование (из корня проекта):
  python scripts/run_planner_ru.py \
    --заказы "data/cleaned_data/25 заказы январь-июль.csv" \
    --реализация "data/cleaned_data/Отчет о реализации товара январь-июль.xlsx" \
    --возвраты "data/cleaned_data/2025 возвраты январь-июль.xlsx" \
    --себестоимость "data/cleaned_data/production_costs.xlsx" \
    --горизонт 3 --модель ets --бэктест 2 \
    --дрейф-цены 0.03 --дельта-промо-пп 0.0 --дельта-комиссии-пп 0.0 --дельта-возвратов-пп 0.0 \
    --мин-маржа-проц 5 --мин-партия 1 \
    --выход output/forecast_report.xlsx --папка-csv output/csv

Примечания:
- Флаги дублируются: русские и латиницей (например, --заказы/--orders).
- Дрейф цены указывать долей (0.03 = +3%). Дельты с пометкой «пп» — в процентных пунктах.
- Скрипт НЕ трогает исходные файлы, он только читает их и пишет отчёт(ы).
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Dict, Any

import pandas as pd

# Надёжный импорт враппера: пробуем как пакет и как модуль
try:
    from scripts.planner import запустить_планировщик  # запуск как файл из корня
except Exception:
    try:
        from scripts.planner import запустить_планировщик  # запуск как модуль из папки scripts
    except Exception as e:
        raise ImportError(
            "Не удалось импортировать planner_ru. Убедитесь, что запускаете из корня проекта и что scripts/ содержит __init__.py"
        ) from e


def _save_excel(res: Dict[str, Any], excel_path: Path) -> None:
    excel_path = Path(excel_path)
    excel_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(excel_path) as writer:
        # порядок листов и безопасная запись даже если DF пустой
        (res.get("прогноз") or pd.DataFrame()).to_excel(writer, sheet_name="Прогноз", index=False)
        (res.get("обоснование_производства") or pd.DataFrame()).to_excel(writer, sheet_name="Обоснование производства", index=False)
        (res.get("бэктест") or pd.DataFrame()).to_excel(writer, sheet_name="Бэктест", index=False)
        (res.get("допущения") or pd.DataFrame()).to_excel(writer, sheet_name="Допущения", index=False)
        (res.get("аналитика") or pd.DataFrame()).to_excel(writer, sheet_name="Аналитика", index=False)


def _save_csvs(res: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "аналитика": "analytics.csv",
        "прогноз": "forecast.csv",
        "обоснование_производства": "production_justification.csv",
        "бэктест": "backtest.csv",
        "допущения": "assumptions.csv",
    }
    for key, fname in mapping.items():
        df = res.get(key)
        if isinstance(df, pd.DataFrame):
            df.to_csv(out_dir / fname, index=False, encoding="utf-8-sig")


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Запуск прогнозного планировщика с русскими флагами и сохранением отчётов",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Пути к данным
    p.add_argument("--orders", "--заказы", dest="orders", required=True, help="CSV с заказами (\"25 заказы ....csv\")")
    p.add_argument("--sales", "--реализация", dest="sales", required=True, help="XLSX отчёт о реализации")
    p.add_argument("--returns", "--возвраты", dest="returns", required=True, help="XLSX возвраты")
    p.add_argument("--costs", "--себестоимость", dest="costs", required=True, help="XLSX себестоимость (production_costs.xlsx)")
    p.add_argument("--planned-inbound", "--план-поставок", dest="planned_inbound", default=None, help="XLSX с планом поставок (опционально)")

    # Параметры модели/горизонта
    p.add_argument("--horizon", "--горизонт", dest="horizon", type=int, default=3, help="Горизонт прогноза в месяцах")
    p.add_argument("--model", "--модель", dest="model", choices=["ets", "arima"], default="ets", help="Тип модели: ETS или ARIMA")
    p.add_argument("--backtest", "--бэктест", dest="backtest", type=int, default=0, help="Сколько последних месяцев проверить бэктестом")

    # Допущения (what-if)
    p.add_argument("--price-drift", "--дрейф-цены", dest="price_drift", type=float, default=0.0, help="Дрейф средней цены (доля: 0.03 = +3%)")
    p.add_argument("--promo-delta-pp", "--дельта-промо-пп", dest="promo_delta_pp", type=float, default=0.0, help="Изменение промо, п.п.")
    p.add_argument("--commission-delta-pp", "--дельта-комиссии-пп", dest="commission_delta_pp", type=float, default=0.0, help="Изменение комиссии, п.п.")
    p.add_argument("--returns-delta-pp", "--дельта-возвратов-пп", dest="returns_delta_pp", type=float, default=0.0, help="Изменение доли возвратов, п.п.")
    p.add_argument("--capacity-limit", "--лимит-мощности", dest="capacity_limit", type=float, default=None, help="Ограничение мощности (опц.)")

    # Пороги рекомендаций
    p.add_argument("--min-margin-pct", "--мин-маржа-проц", dest="min_margin_pct", type=float, default=5.0, help="Мин. маржа для рекомендации, %")
    p.add_argument("--min-batch", "--мин-партия", dest="min_batch", type=int, default=1, help="Мин. партия для рекомендации, шт")

    # Выводы
    p.add_argument("--output", "--выход", dest="output", default=None, help="Путь к XLSX-отчёту (если указать — создастся многостраничный файл)")
    p.add_argument("--csvdir", "--папка-csv", dest="csvdir", default=None, help="Папка для выгрузки CSV (опционально)")

    return p


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    res = запустить_планировщик(
        путь_заказы=args.orders,
        путь_реализация=args.sales,
        путь_возвраты=args.returns,
        путь_себестоимость=args.costs,
        путь_план_поставок=args.planned_inbound,
        горизонт_мес=args.horizon,
        модель=args.model,
        бэктест_посл_мес=args.backtest,
        дрейф_цены=args.price_drift,
        дельта_промо_пп=args.promo_delta_pp,
        дельта_комиссии_пп=args.commission_delta_pp,
        дельта_возвратов_пп=args.returns_delta_pp,
        лимит_мощности=args.capacity_limit,
        мин_маржа_проц=args.min_margin_pct,
        мин_партия=args.min_batch,
    )

    # Сохранения
    if args.output:
        _save_excel(res, Path(args.output))
        print(f"✔ XLSX сохранён: {Path(args.output).resolve()}")
    if args.csvdir:
        _save_csvs(res, Path(args.csvdir))
        print(f"✔ CSV выгружены в: {Path(args.csvdir).resolve()}")

    # Короткая сводка в консоли
    prog = res.get("обоснование_производства")
    if isinstance(prog, pd.DataFrame) and not prog.empty:
        top = prog.sort_values("total_margin", ascending=False).head(5)
        print("\nТоп-5 SKU по ожидаемой марже:")
        with pd.option_context("display.max_columns", 0, "display.width", 140):
            print(top)
    else:
        print("\nОбоснование производства пусто (возможно, нет прогнозируемых данных).")


if __name__ == "__main__":
    main()