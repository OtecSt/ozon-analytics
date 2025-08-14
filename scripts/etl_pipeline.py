"""
sku_analytics.py
=================

Скрипт для построения аналитики по SKU из нескольких исходных файлов Ozon.

Файлы и их структура
--------------------
В проекте ожидаются следующие отчёты:

* **Заказы** (CSV) – содержит отправленные заказы за период. Столбцы могут
  отличаться в разных выгрузках, поэтому скрипт пытается распознать
  идентификатор товара и количество отгруженных единиц. Для расчёта
  затрат на промо используется столбец со скидкой в рублях.
* **Отчёт о реализации товара** (XLSX) – фактические продажи по каждому SKU
  с указанием количества проданного товара, суммы выручки, комиссии Ozon и
  выплаты продавцу.
* **Возвраты** (XLSX) – сведения о количестве и сумме возвращённых
  товаров по каждому SKU.
* **Себестоимость (production_costs)** (XLSX) – справочник себестоимости
  единицы товара по SKU.

Скрипт агрегирует эти данные, рассчитывает ключевые показатели (чистая
выручка, совокупные затраты, маржа, доля возвратов, интенсивность промо
и т. д.), присваивает рекомендацию (SCALE/FIX/STOP) и сохраняет
результат в Excel с несколькими листами: полная аналитика, TOP‑5
прибыли, TOP‑5 убытков, проблемные SKU, помесячные тренды и период
анализа.

Пример использования:

```
python sku_analytics.py --orders "data/25 заказы январь-июль.csv" \
                       --sales "data/Отчет о реализации товара январь-июль.xlsx" \
                       --returns "data/2025 возвраты январь-июль.xlsx" \
                       --costs "data/production_costs.xlsx" \
                       --output "output/analytic_report_by_sku.xlsx"
```

"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _to_numeric(series: pd.Series) -> pd.Series:
    """Приводит строковую колонку с числами в русской локали к числовому типу.

    Удаляет все символы, кроме цифр, знаков минус, запятых и точек, затем
    заменяет запятые на точки и преобразует к float. Неуспешные значения
    превращаются в NaN.

    Args:
        series: исходная серия.

    Returns:
        pd.Series: серия с типом float.
    """
    cleaned = (
        series.astype(str)
        .str.replace(r"[^0-9,\-\.]", "", regex=True)
        .str.replace(",", ".")
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _detect_column(df: pd.DataFrame, keywords: List[str]) -> Optional[str]:
    """Ищет первую колонку, имя которой содержит любое из ключевых слов.

    Поиск нечувствителен к регистру и пробелам. Если колонка не
    найдена, возвращает None.

    Args:
        df: DataFrame, в котором ведётся поиск.
        keywords: список ключевых подстрок, которые ищутся в имени колонки.

    Returns:
        str | None: имя найденной колонки или None.
    """
    lowered = {c.lower().strip(): c for c in df.columns}
    for key_lower, orig in lowered.items():
        for kw in keywords:
            if kw in key_lower:
                return orig
    return None


def load_orders(path: Path) -> pd.DataFrame:
    """Загружает и очищает файл с заказами.

    Формирует единообразные колонки: sku, qty_shipped, discount_rub,
    date_ship и month_ship. При отсутствии столбца с количеством
    отгруженных товаров qty_shipped по умолчанию устанавливается в 1.

    Args:
        path: путь к CSV-файлу заказов.

    Returns:
        DataFrame: нормализованный набор заказов.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    # SKU
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул", "код товара"])
    if not sku_col:
        raise KeyError(f"Не удалось найти колонку с SKU в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Дата отгрузки
    date_col = _detect_column(df, ["дата отгруз", "shipment date"])
    if date_col:
        df = df.rename(columns={date_col: "date_ship"})
        df["date_ship"] = pd.to_datetime(df["date_ship"], dayfirst=True, errors="coerce")
        df["month_ship"] = df["date_ship"].dt.to_period("M")
    else:
        df["date_ship"] = pd.NaT
        df["month_ship"] = pd.NaT
    # Количество отправленных товаров
    qty_col = _detect_column(df, ["кол", "qty"])
    if qty_col:
        df = df.rename(columns={qty_col: "qty_shipped"})
        df["qty_shipped"] = pd.to_numeric(df["qty_shipped"], errors="coerce").fillna(1).astype(int)
    else:
        df["qty_shipped"] = 1
    # Скидка в рублях
    disc_col = _detect_column(df, ["скидка", "discount"])
    if disc_col:
        df = df.rename(columns={disc_col: "discount_rub"})
        df["discount_rub"] = _to_numeric(df["discount_rub"]).fillna(0)
    else:
        df["discount_rub"] = 0.0
    df["sku"] = df["sku"].astype(str).str.strip()
    return df[["sku", "qty_shipped", "discount_rub", "date_ship", "month_ship"]]


def load_sales(path: Path) -> pd.DataFrame:
    """Загружает и очищает файл с отчётом о реализации товаров.

    Args:
        path: путь к файлу Excel отчёта о реализации.

    Returns:
        DataFrame: нормализованный отчёт о продажах.
    """
    df = pd.read_excel(path)
    # SKU
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул", "код товара"])
    if not sku_col:
        raise KeyError(f"Не удалось найти колонку с SKU в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Qty sold
    qty_col = _detect_column(df, ["кол", "qty_sold", "quantity"])
    if qty_col:
        df = df.rename(columns={qty_col: "qty_sold"})
    else:
        raise KeyError("В отчёте о реализации отсутствует колонка количества")
    # Revenue
    rev_col = _detect_column(df, ["выруч", "revenue", "на сумму"])
    if rev_col:
        df = df.rename(columns={rev_col: "revenue_rub"})
    else:
        raise KeyError("Не найден столбец выручки в отчёте о реализации")
    # Ozon fee
    # Иногда в отчёте есть несколько колонок с вознаграждением/комиссией.
    # Ищем приоритетно колонку с «ozon_fee» в имени.
    fee_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["ozon_fee", "озон fee", "комис", "вознаграж"])]
    fee_col = None
    for c in fee_candidates:
        if "ozon_fee" in c.lower():
            fee_col = c
            break
    if fee_col is None and fee_candidates:
        fee_col = fee_candidates[0]
    if fee_col:
        df = df.rename(columns={fee_col: "ozon_fee_rub"})
    else:
        raise KeyError("Не найден столбец комиссии Ozon в отчёте о реализации")
    # Payout (net payout to seller)
    # Ищем колонку, содержащую "net_payout" или ближайший аналог.
    payout_candidates = [
        c for c in df.columns if any(kw in c.lower() for kw in ["net_payout", "к выплат", "итого к начис", "итого к выплат", "к начис"])
    ]
    payout_col = None
    for c in payout_candidates:
        if "net_payout" in c.lower() or ("к выплат" in c.lower() and "итого" not in c.lower()):
            payout_col = c
            break
    if payout_col is None and payout_candidates:
        payout_col = payout_candidates[0]
    if payout_col:
        df = df.rename(columns={payout_col: "net_payout_rub"})
    else:
        raise KeyError("Не найден столбец итоговой выплаты в отчёте о реализации")
    df["sku"] = df["sku"].astype(str).str.strip()
    df["qty_sold"] = pd.to_numeric(df["qty_sold"], errors="coerce").fillna(0).astype(int)
    df["revenue_rub"] = _to_numeric(df["revenue_rub"]).fillna(0)
    df["ozon_fee_rub"] = _to_numeric(df["ozon_fee_rub"]).fillna(0)
    df["net_payout_rub"] = _to_numeric(df["net_payout_rub"]).fillna(0)
    # Удаляем дубликаты имён колонок (появляются, если несколько колонок
    # переименованы в одинаковое имя). Сохраняется первое вхождение.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]
    return df[["sku", "qty_sold", "revenue_rub", "ozon_fee_rub", "net_payout_rub"]]


def load_returns(path: Path) -> pd.DataFrame:
    """Загружает и очищает файл с информацией о возвратах.

    Args:
        path: путь к файлу Excel возвратов.

    Returns:
        DataFrame: агрегированный по SKU файл возвратов.
    """
    df = pd.read_excel(path)
    # SKU
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не удалось найти колонку с SKU в файле возвратов {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Qty returns
    qty_col = _detect_column(df, ["количество", "returns_qty", "возвращ"])
    if qty_col:
        df = df.rename(columns={qty_col: "returns_qty"})
    else:
        raise KeyError("Не найден столбец с количеством возвратов")
    # Return value
    val_col = _detect_column(df, ["стоим", "returns_rub", "price"])
    if val_col:
        df = df.rename(columns={val_col: "returns_rub"})
    else:
        raise KeyError("Не найден столбец с суммой возвратов")
    # Date of return
    date_col = _detect_column(df, ["дата возв", "date"])
    if date_col:
        df = df.rename(columns={date_col: "date_return"})
        df["date_return"] = pd.to_datetime(df["date_return"], dayfirst=True, errors="coerce")
        df["month_return"] = df["date_return"].dt.to_period("M")
    else:
        df["date_return"] = pd.NaT
        df["month_return"] = pd.NaT
    df["sku"] = df["sku"].astype(str).str.strip()
    df["returns_qty"] = pd.to_numeric(df["returns_qty"], errors="coerce").fillna(0).astype(float)
    df["returns_rub"] = _to_numeric(df["returns_rub"]).fillna(0)
    grouped = df.groupby("sku", as_index=False).agg(
        returns_qty=("returns_qty", "sum"),
        returns_rub=("returns_rub", "sum"),
    )
    return grouped


def load_costs(path: Path) -> pd.DataFrame:
    """Загружает справочник себестоимости.

    Args:
        path: путь к файлу себестоимости.

    Returns:
        DataFrame: таблица с колонками sku и production_cost.
    """
    df = pd.read_excel(path)
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не найдено поле SKU в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    cost_col = _detect_column(df, ["себест", "cost", "prod"])
    if not cost_col:
        raise KeyError(f"Не найден столбец себестоимости в файле {path}")
    df = df.rename(columns={cost_col: "production_cost"})
    df["sku"] = df["sku"].astype(str).str.strip()
    df["production_cost"] = pd.to_numeric(df["production_cost"], errors="coerce").fillna(0)
    return df.groupby("sku", as_index=False)["production_cost"].mean()


def compute_promos(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет сумму скидок (расходы на промо) по каждому SKU из заказов."""
    promos = orders_df.groupby("sku", as_index=False)["discount_rub"].sum()
    promos = promos.rename(columns={"discount_rub": "promo_cost"})
    return promos


def compute_analytics(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    promos_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    orders_df: pd.DataFrame,
) -> pd.DataFrame:
    """Расчёт сводной аналитики по SKU."""
    agg_sales = (
        sales_df.groupby("sku", as_index=False)
        .agg(
            total_qty=("qty_sold", "sum"),
            total_rev=("revenue_rub", "sum"),
            total_fee=("ozon_fee_rub", "sum"),
            total_payout=("net_payout_rub", "sum"),
        )
    )
    merged = (
        agg_sales
        .merge(returns_df, on="sku", how="left")
        .merge(promos_df, on="sku", how="left")
        .merge(costs_df, on="sku", how="left")
    )
    merged[["returns_qty", "returns_rub", "promo_cost", "production_cost"]] = (
        merged[["returns_qty", "returns_rub", "promo_cost", "production_cost"]].fillna(0)
    )
    merged["net_qty"] = merged["total_qty"] - merged["returns_qty"]
    merged["net_revenue"] = merged["total_rev"] - merged["returns_rub"]
    merged["production_cost_total"] = merged["production_cost"] * merged["total_qty"]
    merged["comb_cost"] = merged["total_fee"] + merged["production_cost_total"] + merged["promo_cost"]
    merged["margin"] = merged["net_revenue"] - merged["comb_cost"]
    merged["returns_pct"] = merged.apply(
        lambda r: (r["returns_qty"] / r["total_qty"] * 100) if r["total_qty"] else 0, axis=1
    )
    merged["promo_intensity_pct"] = merged.apply(
        lambda r: (r["promo_cost"] / r["total_rev"] * 100) if r["total_rev"] else 0, axis=1
    )
    def rec_action(r: pd.Series) -> str:
        if r["margin"] < 0:
            return "STOP"
        if r["returns_pct"] > 20 or r["promo_intensity_pct"] > 10:
            return "FIX"
        return "SCALE"
    merged["recommended_action"] = merged.apply(rec_action, axis=1)
    if "date_ship" in orders_df.columns:
        ship_map = (
            orders_df.dropna(subset=["date_ship"])
            .groupby("sku", as_index=False)
            .agg(date_ship=("date_ship", "min"), month_ship=("month_ship", "first"))
        )
        merged = merged.merge(ship_map, on="sku", how="left")
    else:
        merged["date_ship"] = pd.NaT
        merged["month_ship"] = pd.NaT
    merged["date_sale"] = pd.NaT
    merged["month_sale"] = pd.NaT
    merged["date"] = merged["date_sale"].combine_first(merged["date_ship"])
    merged["month"] = merged["date"].dt.to_period("M")
    return merged


def save_report(
    analytics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Сохраняет итоговый отчёт в Excel."""
    rus_mapping = {
        "sku": "SKU",
        "total_qty": "Общее количество",
        "total_rev": "Валовая выручка, ₽",
        "total_fee": "Комиссия Ozon, ₽",
        "total_payout": "К выплате, ₽",
        "returns_qty": "Кол-во возвратов, шт.",
        "returns_rub": "Сумма возвратов, ₽",
        "promo_cost": "Расходы на промо, ₽",
        "production_cost": "Себестоимость, ₽",
        "production_cost_total": "Итого себестоимость, ₽",
        "comb_cost": "Общие затраты, ₽",
        "net_qty": "Чистое количество, шт.",
        "net_revenue": "Чистая выручка, ₽",
        "margin": "Маржа, ₽",
        "returns_pct": "Доля возвратов, %",
        "promo_intensity_pct": "Интенсивность промо, %",
        "recommended_action": "Рекомендация",
        "date_sale": "Дата платежа",
        "month_sale": "Месяц платежа",
        "date_ship": "Дата отгрузки",
        "month_ship": "Месяц отгрузки",
        "date": "Дата",
        "month": "Месяц",
    }
    full_df = analytics_df.rename(columns=rus_mapping)
    top5 = full_df.sort_values("Маржа, ₽", ascending=False).head(5)
    flop5 = full_df.sort_values("Маржа, ₽", ascending=True).head(5)
    problems = full_df[full_df["Рекомендация"] != "SCALE"]
    # Тренды: по месяцам суммируем отгруженное количество; возвраты дат не содержат
    monthly_ship = (
        orders_df.dropna(subset=["month_ship"])
        .groupby("month_ship", as_index=False)["qty_shipped"].sum()
        .rename(columns={"month_ship": "month", "qty_shipped": "shipped"})
    )
    # Так как дат возврата нет, отображаем нули
    monthly_ret = pd.DataFrame({"month": monthly_ship["month"], "returned": 0})
    trends = monthly_ship.merge(monthly_ret, on="month", how="left").fillna(0)
    period_df = pd.DataFrame({
        "Показатель": ["Дата отгрузки", "Дата платежа", "Единая дата"],
        "Начало": [
            orders_df["date_ship"].min(),
            full_df["Дата платежа"].min(),
            full_df["Дата"].min(),
        ],
        "Конец": [
            orders_df["date_ship"].max(),
            full_df["Дата платежа"].max(),
            full_df["Дата"].max(),
        ],
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        full_df.to_excel(writer, sheet_name="Полная аналитика", index=False)
        top5.to_excel(writer, sheet_name="TOP5 прибыльные", index=False)
        flop5.to_excel(writer, sheet_name="TOP5 убыточные", index=False)
        problems.to_excel(writer, sheet_name="Проблемные SKU", index=False)
        trends.to_excel(writer, sheet_name="Monthly Trends", index=False)
        period_df.to_excel(writer, sheet_name="Интервал", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение аналитики по SKU из выгрузок Ozon")
    parser.add_argument("--orders", type=str, required=True, help="Путь к CSV-файлу с заказами")
    parser.add_argument("--sales", type=str, required=True, help="Путь к XLSX-файлу с отчётом о реализации")
    parser.add_argument("--returns", type=str, required=True, help="Путь к XLSX-файлу с возвратами")
    parser.add_argument("--costs", type=str, required=True, help="Путь к XLSX-файлу с себестоимостью")
    parser.add_argument("--output", type=str, default="analytic_report_by_sku.xlsx", help="Имя выходного Excel-файла")
    args = parser.parse_args()
    orders_path = Path(args.orders)
    sales_path = Path(args.sales)
    returns_path = Path(args.returns)
    costs_path = Path(args.costs)
    output_path = Path(args.output)
    orders_df = load_orders(orders_path)
    sales_df = load_sales(sales_path)
    returns_df = load_returns(returns_path)
    costs_df = load_costs(costs_path)
    promos_df = compute_promos(orders_df)
    analytics_df = compute_analytics(sales_df, returns_df, promos_df, costs_df, orders_df)
    save_report(analytics_df, orders_df, returns_df, output_path)
    print(f"✔ Отчёт сохранён в {output_path}")


if __name__ == "__main__":
    main()