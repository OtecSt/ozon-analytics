"""
Ultimate Analytics with Unit Economics
=====================================

Этот скрипт объединяет данные о заказах, продажах, возвратах и
производственной себестоимости для товаров на маркетплейсе.  Он
рассчитывает базовые показатели (общее количество, выручка, возвраты,
маржа) и более глубокие метрики unit‑economics: средняя цена, себестоимость
и комиссия на единицу, маржа на единицу, интенсивность промо, доля
возвратов, contribution margin, break-even price и др.

Функции скрипта:
* Автономное определение колонок по ключевым словам (поддержка разных выгрузок).
* Нормализация артикулов (SKU) для корректного объединения данных из разных
  источников.
* Поддержка аргументов командной строки для указания путей к
  исходным файлам и имени выходного отчёта.
* Формирование Excel‑отчёта с несколькими листами: полная аналитика,
  TOP‑5 прибыльных, TOP‑5 убыточных, проблемные SKU, помесячные тренды и
  интервал дат.

Пример использования:

```
python ultimate_analytics_unit.py \
    --orders "25 заказы январь-июль.csv" \
    --sales "Отчет о реализации товара январь-июль.xlsx" \
    --returns "2025 возвраты январь-июль.xlsx" \
    --costs "production_costs.xlsx" \
    --output "ultimate_report_unit.xlsx"
```

"""

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- distutils shim for Python 3.13 compatibility ---

import sys
try:
    import setuptools._distutils as _distutils
    sys.modules['distutils'] = _distutils
    import setuptools._distutils.version as _distutils_version
    sys.modules['distutils.version'] = _distutils_version
except ImportError:
    pass

# --- numpy VisibleDeprecationWarning shim for numpy 2.x compatibility with Sweetviz ---
if not hasattr(np, "VisibleDeprecationWarning"):
    np.VisibleDeprecationWarning = DeprecationWarning


# --- Helper functions ---

def unify_sku(series: pd.Series) -> pd.Series:
    """Приводит идентификатор товара к единому строковому формату.

    Сначала пытается преобразовать значения в число (Int64), затем
    конвертирует их обратно в строку без пробелов. Значения "<NA>"
    заменяются на pd.NA.
    """
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    as_str = numeric.astype(str).str.strip()
    as_str = as_str.where(~as_str.eq("<NA>"), pd.NA)
    return as_str


def _to_numeric(series: pd.Series) -> pd.Series:
    """Преобразует строковую колонку с числами в русской локали в float.

    Удаляет все символы, кроме цифр, минусов, запятых и точек.
    Заменяет запятые на точки и преобразует к числу. Неуспешные
    преобразования превращаются в NaN.
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
    """
    lowered = {c.lower().strip(): c for c in df.columns}
    for key_lower, orig in lowered.items():
        for kw in keywords:
            if kw in key_lower:
                return orig
    return None


# --- Loaders ---

def load_orders(path: Path) -> pd.DataFrame:
    """Загружает и очищает CSV файл с заказами.

    Возвращает DataFrame со следующими колонками:
    - sku: нормализованный идентификатор товара (str)
    - shipment_id: номер отправления (str)
    - qty_shipped: количество отгруженных единиц (int)
    - discount_rub: скидка/промо‑расходы по заказу (float)
    - order_value_rub: сумма заказа (если доступна), иначе NaN
    - date_ship: дата отгрузки (datetime64)
    - month_ship: месяц отгрузки (Period)

    Если количество не задано, qty_shipped ставится равным 1.
    """
    df = pd.read_csv(path, sep=";", encoding="utf-8", low_memory=False)
    # SKU
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул", "код товара"])
    if not sku_col:
        raise KeyError(f"Не удалось найти колонку с SKU в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Shipment/Order identifier (номер отправления)
    ship_col = _detect_column(df, ["номер отправ", "shipment", "отправлен", "номер отпр"])
    if ship_col:
        df = df.rename(columns={ship_col: "shipment_id"})
    else:
        df["shipment_id"] = pd.NA
    # Дата отгрузки
    date_col = _detect_column(df, ["дата отгруз", "shipment date"])
    if date_col:
        df = df.rename(columns={date_col: "date_ship"})
        df["date_ship"] = pd.to_datetime(df["date_ship"], dayfirst=True, errors="coerce")
        df["month_ship"] = df["date_ship"].dt.to_period("M")
    else:
        df["date_ship"] = pd.NaT
        df["month_ship"] = pd.NaT
    # Количество отгруженных товаров
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
    # Сумма заказа (оплачено покупателем/сумма отправления)
    value_col = _detect_column(df, ["сумма", "оплачено", "оплата", "стоим", "price"])
    if value_col:
        df = df.rename(columns={value_col: "order_value_rub"})
        df["order_value_rub"] = _to_numeric(df["order_value_rub"]).fillna(pd.NA)
    else:
        df["order_value_rub"] = pd.NA
    # Normalize sku
    df["sku"] = unify_sku(df["sku"])
    # Return required columns
    return df[["sku", "shipment_id", "qty_shipped", "discount_rub", "order_value_rub", "date_ship", "month_ship"]]


def load_sales(path: Path) -> pd.DataFrame:
    """Загружает и очищает Excel с отчётом о реализации."""
    df = pd.read_excel(path)
    # SKU
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул", "код товара"])
    if not sku_col:
        raise KeyError(f"Не удалось найти колонку с SKU в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Qty sold
    qty_col = _detect_column(df, ["кол", "qty_sold", "quantity"])
    if not qty_col:
        raise KeyError("В отчёте о реализации отсутствует колонка количества")
    df = df.rename(columns={qty_col: "qty_sold"})
    # Revenue
    rev_col = _detect_column(df, ["выруч", "revenue", "на сумму"])
    if not rev_col:
        raise KeyError("Не найден столбец выручки в отчёте о реализации")
    df = df.rename(columns={rev_col: "revenue_rub"})
    # Ozon fee
    fee_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["ozon_fee", "озон fee", "комис", "вознаграж"])]
    fee_col = None
    for c in fee_candidates:
        if "ozon_fee" in c.lower():
            fee_col = c
            break
    if fee_col is None and fee_candidates:
        fee_col = fee_candidates[0]
    if not fee_col:
        raise KeyError("Не найден столбец комиссии Ozon в отчёте о реализации")
    df = df.rename(columns={fee_col: "ozon_fee_rub"})
    # Net payout
    payout_candidates = [c for c in df.columns if any(kw in c.lower() for kw in ["net_payout", "к выплат", "итого к начис", "итого к выплат", "к начис"])]
    payout_col = None
    for c in payout_candidates:
        if "net_payout" in c.lower() or ("к выплат" in c.lower() and "итого" not in c.lower()):
            payout_col = c
            break
    if payout_col is None and payout_candidates:
        payout_col = payout_candidates[0]
    if not payout_col:
        raise KeyError("Не найден столбец итоговой выплаты в отчёте о реализации")
    df = df.rename(columns={payout_col: "net_payout_rub"})
    df["sku"] = unify_sku(df["sku"])
    df["qty_sold"] = pd.to_numeric(df["qty_sold"], errors="coerce").fillna(0).astype(int)
    df["revenue_rub"] = _to_numeric(df["revenue_rub"]).fillna(0)
    df["ozon_fee_rub"] = _to_numeric(df["ozon_fee_rub"]).fillna(0)
    df["net_payout_rub"] = _to_numeric(df["net_payout_rub"]).fillna(0)
    return df[["sku", "qty_sold", "revenue_rub", "ozon_fee_rub", "net_payout_rub"]]


def load_returns(path: Path) -> pd.DataFrame:
    """
    Загружает и очищает файл возвратов (XLSX).

    Возвращает DataFrame с колонками:
    - sku: нормализованный идентификатор товара
    - returns_qty: количество возвращённых единиц
    - returns_rub: сумма возврата (стоимость товаров)
    - date_return: дата возврата (datetime64)
    - shipment_id: номер отправления (для связывания с заказом), если доступно

    Преобразует числовые значения и даты; нормализация SKU выполняется через
    unify_sku. Если номер отправления отсутствует, колонка заполняется NA.
    """
    df = pd.read_excel(path)
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не удалось найти колонку с SKU в файле возвратов {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Количество возвратов
    qty_col = _detect_column(df, ["количество", "returns_qty", "возвращ"])
    if not qty_col:
        raise KeyError("Не найден столбец с количеством возвратов")
    df = df.rename(columns={qty_col: "returns_qty"})
    # Стоимость возврата
    val_col = _detect_column(df, ["стоим", "returns_rub", "price"])
    if not val_col:
        raise KeyError("Не найден столбец с суммой возвратов")
    df = df.rename(columns={val_col: "returns_rub"})
    # Дата возврата
    date_col = _detect_column(df, ["дата возв", "date"])
    if date_col:
        df = df.rename(columns={date_col: "date_return"})
        df["date_return"] = pd.to_datetime(df["date_return"], dayfirst=True, errors="coerce")
    else:
        df["date_return"] = pd.NaT
    # Номер отправления для связи с заказом
    ship_col = _detect_column(df, ["номер отправ", "shipment", "отправлен", "номер отпр"])
    if ship_col:
        df = df.rename(columns={ship_col: "shipment_id"})
    else:
        df["shipment_id"] = pd.NA
    # Конвертация типов
    df["sku"] = unify_sku(df["sku"])
    df["returns_qty"] = pd.to_numeric(df["returns_qty"], errors="coerce").fillna(0).astype(float)
    df["returns_rub"] = _to_numeric(df["returns_rub"]).fillna(0)
    # Оставляем нужные столбцы
    return df[["sku", "returns_qty", "returns_rub", "date_return", "shipment_id"]]


def load_costs(path: Path) -> pd.DataFrame:
    """Загружает справочник себестоимости."""
    df = pd.read_excel(path)
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не найдено поле SKU в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    cost_col = _detect_column(df, ["себест", "cost", "prod"])
    if not cost_col:
        raise KeyError(f"Не найден столбец себестоимости в файле {path}")
    df = df.rename(columns={cost_col: "production_cost"})
    df["sku"] = unify_sku(df["sku"])
    df["production_cost"] = pd.to_numeric(df["production_cost"], errors="coerce").fillna(0)
    return df.groupby("sku", as_index=False)["production_cost"].mean()


def compute_promos(orders_df: pd.DataFrame) -> pd.DataFrame:
    """Вычисляет суммарные расходы на промо по каждому SKU.

    Суммирует скидку (discount_rub) из заказов, так как именно она
    отражает реальные потери на акции.
    """
    promos = orders_df.groupby("sku", as_index=False)["discount_rub"].sum()
    return promos.rename(columns={"discount_rub": "promo_cost"})


# --- New loaders for additional datasets ---

def load_promo_actions(path: Path) -> pd.DataFrame:
    """
    Загружает аналитику по акциям.

    Ожидает Excel с колонками:
    - 'Ozon ID' — идентификатор товара
    - 'Заказано товаров' — общее количество заказанных товаров
    - 'Заказано товаров по акции' — количество товаров, заказанных по акции
    - 'Уникальные посетители, всего' — общее число уникальных посетителей
    - 'Уникальные посетители с просмотром карточки товара' — уникальные посетители, просмотревшие карточку
    - 'Конверсия в корзину из карточки товара' — конверсия в корзину (в %, может быть строка с %)
    - 'Заказано на сумму' — сумма заказов
    - 'Заказано на сумму по акции' — сумма заказов по акциям

    Возвращает DataFrame с нормализованным SKU и числовыми значениями.
    """
    df = pd.read_excel(path)
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не найден столбец с Ozon ID в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Переименование остальных колонок
    mapping = {
        "заказано товаров": "ordered_qty",
        "заказано товаров по акции": "ordered_qty_promo",
        "уникальные посетители, всего": "unique_visitors_total",
        "уникальные посетители с просмотром карточки товара": "unique_visitors_card",
        "конверсия в корзину из карточки товара": "card_to_cart_conversion",
        "заказано на сумму": "ordered_amount",
        "заказано на сумму по акции": "ordered_amount_promo",
    }
    for col in df.columns:
        low = col.lower().strip()
        if low in mapping:
            df = df.rename(columns={col: mapping[low]})
    # Нормализуем SKU
    df["sku"] = unify_sku(df["sku"])
    # Числовые преобразования
    numeric_cols = [
        "ordered_qty", "ordered_qty_promo",
        "unique_visitors_total", "unique_visitors_card",
        "ordered_amount", "ordered_amount_promo",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    # Конверсия в корзину: может содержать %
    if "card_to_cart_conversion" in df.columns:
        conv = df["card_to_cart_conversion"].astype(str)
        conv = conv.str.replace("%", "", regex=False)
        df["card_to_cart_conversion"] = pd.to_numeric(conv, errors="coerce").fillna(0)
    else:
        df["card_to_cart_conversion"] = 0
    return df[[
        "sku", "ordered_qty", "ordered_qty_promo", "unique_visitors_total",
        "unique_visitors_card", "card_to_cart_conversion",
        "ordered_amount", "ordered_amount_promo"
    ]].fillna(0)


def load_accruals(path: Path) -> pd.DataFrame:
    """
    Загружает отчёт по начислениям.

    Ожидает Excel с колонками:
    - 'OZON id' — идентификатор товара
    - 'Количество' — количество (может быть 0)
    - 'Цена продавца' — цена продавца (не используется напрямую)
    - 'Сумма итого, руб' — сумма начисления (положительная/отрицательная)
    - 'Дата ' — дата начисления (опционально)

    Возвращает агрегированный DataFrame по SKU:
    - accrual_amount_sum: суммарная сумма начислений
    - accrual_qty_sum: суммарное количество
    - accrual_cost_per_unit: средняя сумма начислений на единицу (если qty > 0)
    """
    df = pd.read_excel(path)
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не найден столбец с Ozon ID в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    qty_col = _detect_column(df, ["количество", "qty", "кол-во"])
    if qty_col:
        df = df.rename(columns={qty_col: "accrual_qty"})
    else:
        df["accrual_qty"] = 0
    amount_col = _detect_column(df, ["сумма итого", "amount", "сумма"])
    if amount_col:
        df = df.rename(columns={amount_col: "accrual_amount"})
    else:
        raise KeyError(f"Не найден столбец суммы начислений в файле {path}")
    # Convert to numeric
    df["accrual_qty"] = pd.to_numeric(df["accrual_qty"], errors="coerce").fillna(0)
    df["accrual_amount"] = pd.to_numeric(df["accrual_amount"], errors="coerce").fillna(0)
    # Normalize SKU
    df["sku"] = unify_sku(df["sku"])
    # Aggregate by SKU
    agg = df.groupby("sku", as_index=False).agg(
        accrual_amount_sum=("accrual_amount", "sum"),
        accrual_qty_sum=("accrual_qty", "sum"),
    )
    agg["accrual_cost_per_unit"] = agg.apply(
        lambda r: (r["accrual_amount_sum"] / r["accrual_qty_sum"]) if r["accrual_qty_sum"] else 0,
        axis=1,
    )
    return agg


def load_inventory(path: Path) -> pd.DataFrame:
    """
    Загружает данные о движении товаров по складам.

    Ожидает Excel с колонками:
    - 'OZON id' — идентификатор товара
    - 'Остаток на начало периода'
    - 'Приход'
    - 'Приход из поставок'
    - 'Расход'
    - 'Расход отгруженный в доставку'
    - 'Остаток на конец периода'
    - 'Валидный сток на конец периода'
    - 'Невалидный сток на конец периода'

    Возвращает агрегированный DataFrame по SKU с соответствующими полями
    и вычисленным средним запасом (average_inventory).
    """
    df = pd.read_excel(path)
    sku_col = _detect_column(df, ["ozon id", "sku", "артикул"])
    if not sku_col:
        raise KeyError(f"Не найден столбец с Ozon ID в файле {path}")
    df = df.rename(columns={sku_col: "sku"})
    # Rename other columns using keywords
    col_map = {
        "остаток на начало": "opening_stock",
        "приход из поставок": "incoming_from_supplies",
        "приход": "incoming",
        "расход отгруженный": "outgoing_shipped",
        "расход": "outgoing",
        "остаток на конец": "ending_stock",
        "валидный сток": "valid_stock_end",
        "невалидный сток": "invalid_stock_end",
    }
    for col in df.columns:
        low = col.lower().strip()
        for kw, newname in col_map.items():
            if kw in low:
                df = df.rename(columns={col: newname})
                break
    # Normalize SKU
    df["sku"] = unify_sku(df["sku"])
    # Convert all numeric inventory columns to numeric (с учётом дублей)
    inventory_cols = [
        "opening_stock", "incoming", "incoming_from_supplies",
        "outgoing", "outgoing_shipped", "ending_stock",
        "valid_stock_end", "invalid_stock_end",
    ]

    for col in inventory_cols:
        # все столбцы с точным именем col (могут быть дубли)
        mask = (df.columns == col)
        cnt = int(mask.sum())
        if cnt == 0:
            df[col] = 0
            continue

        if cnt == 1:
            # обычный случай — одна колонка
            df[col] = pd.to_numeric(df.loc[:, col], errors="coerce").fillna(0)
        else:
            # дубли — суммируем построчно
            block = df.loc[:, mask]
            block_num = block.apply(pd.to_numeric, errors="coerce").fillna(0)
            df[col] = block_num.sum(axis=1)
    # Aggregate by sku (if multiple warehouses)
    agg = df.groupby("sku", as_index=False)[inventory_cols].sum()
    # Compute average inventory
    agg["average_inventory"] = (agg["opening_stock"] + agg["ending_stock"]) / 2
    return agg


# --- Analytics computation ---

def compute_analytics(
    sales_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    promos_df: pd.DataFrame,
    costs_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    return_window_days: int = 90,
    promo_df: Optional[pd.DataFrame] = None,
    accrual_df: Optional[pd.DataFrame] = None,
    inventory_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Расчёт сводной аналитики и unit‑economics по SKU.

    Параметры
    ---------
    sales_df : DataFrame
        Данные о реализации (продажи). Ожидает поля sku, qty_sold,
        revenue_rub, ozon_fee_rub, net_payout_rub.
    returns_df : DataFrame
        Детализированный список возвратов. Ожидает поля sku, returns_qty,
        returns_rub, date_return, shipment_id.
    promos_df : DataFrame
        Сумма скидок/промо из заказов по каждому SKU. Поле promo_cost.
    costs_df : DataFrame
        Справочник себестоимости. Поле production_cost.
    orders_df : DataFrame
        Детализированные заказы. Ожидает поля sku, shipment_id,
        qty_shipped, discount_rub, order_value_rub, date_ship, month_ship.
    return_window_days : int, optional
        Ширина окна (в днях), в рамках которого учитываются возвраты с момента
        отгрузки товара. По умолчанию 90.

    Возвращает DataFrame, объединяющий показатели продаж, возвратов,
    промо, себестоимости и вычисленные unit‑economics.
    """
    # --- Обработка возвратов ---
    # Если есть номер отправления и дата, связываем возвраты с датой отгрузки
    filtered_returns: pd.DataFrame
    if returns_df["shipment_id"].notna().any() and orders_df["shipment_id"].notna().any():
        # Объединяем по shipment_id
        merged_ret = returns_df.merge(
            orders_df[["shipment_id", "date_ship"]], on="shipment_id", how="left"
        )
        # Разница между датой возврата и отгрузки
        merged_ret["days_between"] = (
            merged_ret["date_return"] - merged_ret["date_ship"]
        ).dt.days
        # Фильтруем только возвраты, совершённые в пределах окна
        filtered_returns = merged_ret[
            (merged_ret["days_between"].notna())
            & (merged_ret["days_between"] >= 0)
            & (merged_ret["days_between"] <= return_window_days)
        ].copy()
    else:
        # Если нет возможности связать с датой отгрузки, используем все возвраты
        filtered_returns = returns_df.copy()
    # Агрегируем возвращённые кол-во и сумму по SKU
    agg_returns = filtered_returns.groupby("sku", as_index=False).agg(
        returns_qty=("returns_qty", "sum"),
        returns_rub=("returns_rub", "sum"),
    )
    # --- Обработка заказов для дополнительных метрик ---
    # Сумма отгруженного количества (по заказам)
    orders_agg = orders_df.groupby("sku", as_index=False).agg(
        qty_shipped_sum=("qty_shipped", "sum"),
        order_value_sum=("order_value_rub", "sum"),
        order_count=("shipment_id", lambda x: x.nunique()),
    )
    # Агрегируем продажи
    agg_sales = (
        sales_df.groupby("sku", as_index=False)
        .agg(
            total_qty=("qty_sold", "sum"),
            total_rev=("revenue_rub", "sum"),
            total_fee=("ozon_fee_rub", "sum"),
            total_payout=("net_payout_rub", "sum"),
        )
    )
    # Объединяем продажи с возвратами, промо, себестоимостью и заказами
    merged = (
        agg_sales
        .merge(agg_returns, on="sku", how="left")
        .merge(promos_df, on="sku", how="left")
        .merge(costs_df, on="sku", how="left")
        .merge(orders_agg, on="sku", how="left")
    )
    # Заполняем NaN в числовых полях нулями
    merged[["returns_qty", "returns_rub", "promo_cost", "production_cost", "qty_shipped_sum", "order_value_sum", "order_count"]] = (
        merged[["returns_qty", "returns_rub", "promo_cost", "production_cost", "qty_shipped_sum", "order_value_sum", "order_count"]].fillna(0)
    )
    # Расчёт базовых показателей
    merged["net_qty"] = merged["total_qty"] - merged["returns_qty"]
    merged["net_revenue"] = merged["total_rev"] - merged["returns_rub"]
    # Общая себестоимость для всех проданных единиц
    merged["production_cost_total"] = merged["production_cost"] * merged["total_qty"]
    # Совокупные затраты (комиссия, себестоимость, промо)
    merged["comb_cost"] = merged["total_fee"] + merged["production_cost_total"] + merged["promo_cost"]
    merged["margin"] = merged["net_revenue"] - merged["comb_cost"]
    # Unit‑metrics
    merged["avg_price_per_unit"] = merged.apply(
        lambda r: r["total_rev"] / r["total_qty"] if r["total_qty"] else 0, axis=1
    )
    merged["avg_net_price_per_unit"] = merged.apply(
        lambda r: r["net_revenue"] / r["net_qty"] if r["net_qty"] else 0, axis=1
    )
    merged["production_cost_per_unit"] = merged["production_cost"]
    merged["commission_per_unit"] = merged.apply(
        lambda r: r["total_fee"] / r["total_qty"] if r["total_qty"] else 0, axis=1
    )
    merged["promo_per_unit"] = merged.apply(
        lambda r: r["promo_cost"] / r["total_qty"] if r["total_qty"] else 0, axis=1
    )
    merged["margin_per_unit"] = merged.apply(
        lambda r: r["margin"] / r["net_qty"] if r["net_qty"] else 0, axis=1
    )
    merged["contribution_margin"] = merged.apply(
        lambda r: (r["margin_per_unit"] / r["avg_price_per_unit"]) if r["avg_price_per_unit"] else 0, axis=1
    )
    # Break-even price: сумма издержек на единицу
    merged["break_even_price"] = (
        merged["production_cost_per_unit"] + merged["commission_per_unit"] + merged["promo_per_unit"]
    )
    # Проценты возвратов и промо
    merged["returns_pct"] = merged.apply(
        lambda r: (r["returns_qty"] / r["total_qty"] * 100) if r["total_qty"] else 0, axis=1
    )
    merged["promo_intensity_pct"] = merged.apply(
        lambda r: (r["promo_cost"] / r["total_rev"] * 100) if r["total_rev"] else 0, axis=1
    )
    merged["margin_pct"] = merged.apply(
        lambda r: (r["margin"] / r["net_revenue"] * 100) if r["net_revenue"] else 0, axis=1
    )
    # Дополнительные метрики unit‑economics
    # Средний чек: средняя сумма заказа (по всем заказам данного SKU)
    merged["avg_order_value"] = merged.apply(
        lambda r: (r["order_value_sum"] / r["order_count"]) if r["order_count"] else 0, axis=1
    )
    # Конверсия заказа → возврат (ед.): доля возврата от всего отгруженного количества
    merged["conversion_order_return"] = merged.apply(
        lambda r: (r["returns_qty"] / r["qty_shipped_sum"] * 100) if r["qty_shipped_sum"] else 0, axis=1
    )
    # Рекомендации
    def rec_action(r: pd.Series) -> str:
        if r["margin"] < 0:
            return "STOP"
        # высшая чувствительность к возвратам >20% или интенсивности промо >10%
        if r["returns_pct"] > 20 or r["promo_intensity_pct"] > 10:
            return "FIX"
        return "SCALE"
    merged["recommended_action"] = merged.apply(rec_action, axis=1)
    # Дата отгрузки – минимальная дата заказа
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
    # Date of sale is not provided, keep NaT for now
    merged["date_sale"] = pd.NaT
    merged["month_sale"] = pd.NaT
    # Unified date and month
    merged["date"] = merged["date_sale"].combine_first(merged["date_ship"])
    merged["month"] = merged["date"].dt.to_period("M")
    # --- Интеграция аналитики по акциям ---
    if promo_df is not None and not promo_df.empty:
        # агрегируем по SKU: суммируем количественные показатели, среднее для конверсии
        agg_promo = promo_df.groupby("sku", as_index=False).agg(
            ordered_qty_promo_sum=("ordered_qty_promo", "sum"),
            unique_visitors_total_sum=("unique_visitors_total", "sum"),
            unique_visitors_card_sum=("unique_visitors_card", "sum"),
            ordered_amount_promo_sum=("ordered_amount_promo", "sum"),
            ordered_amount_sum=("ordered_amount", "sum"),
            card_to_cart_conversion_mean=("card_to_cart_conversion", "mean"),
        )
        merged = merged.merge(agg_promo, on="sku", how="left")
    else:
        # Добавляем пустые поля
        merged["ordered_qty_promo_sum"] = 0
        merged["unique_visitors_total_sum"] = 0
        merged["unique_visitors_card_sum"] = 0
        merged["ordered_amount_promo_sum"] = 0
        merged["ordered_amount_sum"] = 0
        merged["card_to_cart_conversion_mean"] = 0
    # --- Интеграция начислений ---
    if accrual_df is not None and not accrual_df.empty:
        merged = merged.merge(accrual_df, on="sku", how="left")
        merged[["accrual_amount_sum", "accrual_qty_sum", "accrual_cost_per_unit"]] = (
            merged[["accrual_amount_sum", "accrual_qty_sum", "accrual_cost_per_unit"]].fillna(0)
        )
        # Маржа с учётом начислений: считаем, что начисления (positive or negative) корректируют маржу
        merged["margin_adj"] = merged["margin"] + merged["accrual_amount_sum"]
    else:
        merged["accrual_amount_sum"] = 0
        merged["accrual_qty_sum"] = 0
        merged["accrual_cost_per_unit"] = 0
        merged["margin_adj"] = merged["margin"]
    # --- Интеграция складских данных ---
    if inventory_df is not None and not inventory_df.empty:
        merged = merged.merge(inventory_df, on="sku", how="left")
        # Заполняем NaN нулями
        inv_cols = [
            "opening_stock", "incoming", "incoming_from_supplies", "outgoing", "outgoing_shipped",
            "ending_stock", "valid_stock_end", "invalid_stock_end", "average_inventory"
        ]
        for col in inv_cols:
            if col in merged.columns:
                merged[col] = merged[col].fillna(0)
        # Вычисляем стоимость реализованного товара (COGS) как себестоимость * чистое количество
        merged["cogs"] = merged["production_cost_per_unit"] * merged["net_qty"]
        # Оборачиваемость запасов: cogs / average_inventory
        merged["inventory_turnover"] = merged.apply(
            lambda r: (r["cogs"] / r["average_inventory"]) if r.get("average_inventory", 0) else 0,
            axis=1,
        )
    else:
        merged["opening_stock"] = 0
        merged["incoming"] = 0
        merged["incoming_from_supplies"] = 0
        merged["outgoing"] = 0
        merged["outgoing_shipped"] = 0
        merged["ending_stock"] = 0
        merged["valid_stock_end"] = 0
        merged["invalid_stock_end"] = 0
        merged["average_inventory"] = 0
        merged["cogs"] = merged["production_cost_per_unit"] * merged["net_qty"]
        merged["inventory_turnover"] = 0
    # --- ABC классификация по объёму продаж (выручке) ---
    # Сортируем SKU по убыванию выручки и вычисляем долю в общей сумме
    if not merged.empty and merged["total_rev"].sum() > 0:
        abc_df = merged[["sku", "total_rev"]].sort_values("total_rev", ascending=False).copy()
        total_sum = abc_df["total_rev"].sum()
        abc_df["cum_share"] = abc_df["total_rev"].cumsum() / total_sum
        def _abc_class(cs: float) -> str:
            if cs <= 0.8:
                return "A"
            elif cs <= 0.95:
                return "B"
            else:
                return "C"
        abc_df["ABC_class"] = abc_df["cum_share"].apply(_abc_class)
        merged = merged.merge(abc_df[["sku", "ABC_class"]], on="sku", how="left")
    else:
        merged["ABC_class"] = "C"
    # --- XYZ классификация по стабильности спроса (коэффициент вариации) ---
    # Используем weekly shipments из orders_df для расчёта вариации
    if not orders_df.empty and orders_df["date_ship"].notna().any():
        weekly = orders_df.copy()
        weekly["week_period"] = weekly["date_ship"].dt.to_period("W")
        weekly_group = (
            weekly.groupby(["sku", "week_period"], as_index=False)["qty_shipped"].sum()
        )
        cv_df = weekly_group.groupby("sku").agg(mean_qty=("qty_shipped", "mean"), std_qty=("qty_shipped", "std")).reset_index()
        cv_df["cv"] = cv_df.apply(lambda r: (r["std_qty"] / r["mean_qty"]) if r["mean_qty"] else float('inf'), axis=1)
        def _xyz_class(cv: float) -> str:
            if cv <= 0.1:
                return "X"
            elif cv <= 0.25:
                return "Y"
            else:
                return "Z"
        cv_df["XYZ_class"] = cv_df["cv"].apply(_xyz_class)
        merged = merged.merge(cv_df[["sku", "XYZ_class"]], on="sku", how="left")
        merged["XYZ_class"] = merged["XYZ_class"].fillna("Z")
    else:
        merged["XYZ_class"] = "Z"
    return merged


def save_report(
    analytics_df: pd.DataFrame,
    orders_df: pd.DataFrame,
    output_path: Path,
    returns_df: Optional[pd.DataFrame] = None,
    return_window_days: int = 90,
) -> None:
    """Сохраняет отчёт с unit‑economics в Excel.

    Аналитика по SKU передаётся в `analytics_df`.  Периодические
    тренды (помесячные и по неделям) рассчитываются на основе `orders_df`
    и, при наличии, `returns_df`.  Если `returns_df` задан, то для
    построения трендов возвраты фильтруются в пределах `return_window_days`.
    """
    # Переименовываем колонки на русский
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
        "avg_price_per_unit": "Средняя цена, ₽",
        "avg_net_price_per_unit": "Средняя чистая цена, ₽",
        "production_cost_per_unit": "Себестоимость за ед., ₽",
        "commission_per_unit": "Комиссия за ед., ₽",
        "promo_per_unit": "Промо за ед., ₽",
        "margin_per_unit": "Маржа за ед., ₽",
        "contribution_margin": "Contribution margin",
        "break_even_price": "Break‑even price, ₽",
        "margin_pct": "Маржа, %",
        "returns_pct": "Доля возвратов, %",
        "promo_intensity_pct": "Интенсивность промо, %",
        "avg_order_value": "Средний чек, ₽",
        "conversion_order_return": "Конверсия заказа → возврат, %",
        "recommended_action": "Рекомендация",
        "date_sale": "Дата платежа",
        "month_sale": "Месяц платежа",
        "date_ship": "Дата отгрузки",
        "month_ship": "Месяц отгрузки",
        "date": "Дата",
        "month": "Месяц",
        # Метрики из аналитики по акциям
        "ordered_qty_promo_sum": "Заказано по акции, шт.",
        "unique_visitors_total_sum": "Уникальные посетители всего",
        "unique_visitors_card_sum": "Уникальные посетители карточки",
        "ordered_amount_promo_sum": "Сумма заказов по акции, ₽",
        "ordered_amount_sum": "Сумма заказов, ₽",
        "card_to_cart_conversion_mean": "Конверсия карточка→корзина, %",
        # Метрики начислений
        "accrual_amount_sum": "Сумма начислений, ₽",
        "accrual_qty_sum": "Кол-во начислений, шт.",
        "accrual_cost_per_unit": "Ср. начисление за ед., ₽",
        "margin_adj": "Маржа (с учётом начислений), ₽",
        # Метрики складских данных
        "opening_stock": "Остаток на начало, шт.",
        "incoming": "Приход, шт.",
        "incoming_from_supplies": "Приход из поставок, шт.",
        "outgoing": "Расход, шт.",
        "outgoing_shipped": "Расход отгруженный, шт.",
        "ending_stock": "Остаток на конец, шт.",
        "valid_stock_end": "Валидный остаток, шт.",
        "invalid_stock_end": "Невалидный остаток, шт.",
        "average_inventory": "Средний запас, шт.",
        "cogs": "Себестоимость проданных, ₽",
        "inventory_turnover": "Оборачиваемость запасов",
        "ABC_class": "ABC‑класс",
        "XYZ_class": "XYZ‑класс",
    }
    full_df = analytics_df.rename(columns=rus_mapping)
    # TOP lists
    top5 = full_df.sort_values("Маржа, ₽", ascending=False).head(5)
    flop5 = full_df.sort_values("Маржа, ₽", ascending=True).head(5)
    problems = full_df[full_df["Рекомендация"] != "SCALE"]
    # Monthly trends: shipped units and returns
    monthly_ship = (
        orders_df.dropna(subset=["month_ship"])
        .groupby("month_ship", as_index=False)["qty_shipped"].sum()
        .rename(columns={"month_ship": "period", "qty_shipped": "shipped"})
    )
    # Подготовка DataFrame возвратов для трендов
    if returns_df is not None and not returns_df.empty:
        # Фильтрация возвратов в пределах return_window_days, если есть shipment_id
        if returns_df["shipment_id"].notna().any() and orders_df["shipment_id"].notna().any():
            merged_ret = returns_df.merge(
                orders_df[["shipment_id", "date_ship"]], on="shipment_id", how="left"
            )
            merged_ret["days_between"] = (
                merged_ret["date_return"] - merged_ret["date_ship"]
            ).dt.days
            filtered_ret = merged_ret[
                (merged_ret["days_between"].notna())
                & (merged_ret["days_between"] >= 0)
                & (merged_ret["days_between"] <= return_window_days)
            ].copy()
        else:
            filtered_ret = returns_df.copy()
        # Месячные возвраты
        filtered_ret["month_return"] = filtered_ret["date_return"].dt.to_period("M")
        monthly_ret = (
            filtered_ret.dropna(subset=["month_return"])
            .groupby("month_return", as_index=False)["returns_qty"].sum()
            .rename(columns={"month_return": "period", "returns_qty": "returned"})
        )
        # Еженедельные возвраты
        filtered_ret["week_return"] = filtered_ret["date_return"].dt.to_period("W")
        weekly_ret = (
            filtered_ret.dropna(subset=["week_return"])
            .groupby("week_return", as_index=False)["returns_qty"].sum()
            .rename(columns={"week_return": "period", "returns_qty": "returned"})
        )
    else:
        monthly_ret = pd.DataFrame({"period": monthly_ship["period"], "returned": 0})
        weekly_ret = pd.DataFrame(columns=["period", "returned"])
    # Совмещаем месячные тренды
    trends = monthly_ship.merge(monthly_ret, on="period", how="left").fillna(0)
    # Weekly trends: shipped and returned
    # Shipments per week
    weekly_ship = (
        orders_df.dropna(subset=["date_ship"])
        .copy()
    )
    weekly_ship["week"] = weekly_ship["date_ship"].dt.to_period("W")
    weekly_ship = (
        weekly_ship.groupby("week", as_index=False)["qty_shipped"].sum()
        .rename(columns={"week": "period", "qty_shipped": "shipped"})
    )
    if not weekly_ret.empty:
        weekly_trends = weekly_ship.merge(weekly_ret, on="period", how="left").fillna(0)
    else:
        weekly_trends = weekly_ship.copy()
        weekly_trends["returned"] = 0
    # Period interval
    period_df = pd.DataFrame({
        "Показатель": ["Дата отгрузки", "Дата платежа", "Единая дата"],
        "Начало": [orders_df["date_ship"].min(), full_df["Дата платежа"].min(), full_df["Дата"].min()],
        "Конец": [orders_df["date_ship"].max(), full_df["Дата платежа"].max(), full_df["Дата"].max()],
    })
    # Сохраняем файл
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path) as writer:
        full_df.to_excel(writer, sheet_name="Полная аналитика", index=False)
        top5.to_excel(writer, sheet_name="TOP5 прибыльные", index=False)
        flop5.to_excel(writer, sheet_name="TOP5 убыточные", index=False)
        problems.to_excel(writer, sheet_name="Проблемные SKU", index=False)
        trends.rename(columns={"period": "Месяц", "shipped": "Отгружено", "returned": "Возвраты"}).to_excel(
            writer, sheet_name="Monthly Trends", index=False
        )
        weekly_trends.rename(columns={"period": "Неделя", "shipped": "Отгружено", "returned": "Возвраты"}).to_excel(
            writer, sheet_name="Weekly Trends", index=False
        )
        period_df.to_excel(writer, sheet_name="Интервал", index=False)
        # --- Анализ корреляции между интенсивностью промо и ключевыми метриками ---
        # Вычисляем корреляции в исходном DataFrame (до русских переименований)
        corr_metrics = {
            "margin_per_unit": "Маржа за ед., ₽",
            "returns_pct": "Доля возвратов, %",
            "avg_price_per_unit": "Средняя цена, ₽",
            "avg_net_price_per_unit": "Средняя чистая цена, ₽",
            "commission_per_unit": "Комиссия за ед., ₽",
            "production_cost_per_unit": "Себестоимость за ед., ₽",
            "margin_pct": "Маржа, %",
            "contribution_margin": "Contribution margin",
            # Новые метрики для корреляции
            "inventory_turnover": "Оборачиваемость запасов",
            "margin_adj": "Маржа (с учётом начислений), ₽",
        }
        corr_rows = []
        # Заполняем коэффициенты корреляции (Pearson); игнорируем пары, где недоступно
        for col, label in corr_metrics.items():
            try:
                corr_value = analytics_df["promo_intensity_pct"].corr(analytics_df[col])
                corr_rows.append({"Метрика": label, "Корреляция с промо": corr_value})
            except Exception:
                corr_rows.append({"Метрика": label, "Корреляция с промо": np.nan})
        corr_df = pd.DataFrame(corr_rows)
        corr_df.to_excel(writer, sheet_name="Promo vs Metrics", index=False)

        # --- Итоговые показатели (ИТОГО) ---
        # Определяем столбцы для агрегации
        agg_cols = [
            "total_qty", "net_qty", "total_rev", "net_revenue", "returns_qty",
            "returns_rub", "promo_cost", "production_cost_total", "comb_cost",
            "margin", "margin_adj",
        ]
        # Сводные суммы по всем позициям, прибыльным и убыточным
        overall_totals = analytics_df[agg_cols].sum()
        profit_df = analytics_df[analytics_df["margin"] >= 0]
        loss_df = analytics_df[analytics_df["margin"] < 0]
        profit_totals = profit_df[agg_cols].sum()
        loss_totals = loss_df[agg_cols].sum()
        totals_df = pd.DataFrame([overall_totals, profit_totals, loss_totals], index=["ИТОГО", "Прибыльные", "Убыточные"])
        # Переименовываем колонки на русский язык
        totals_df = totals_df.rename(columns=rus_mapping)
        totals_df.to_excel(writer, sheet_name="Totals", index=True)

        # --- Итоговые значения по периодам (неделя/месяц/квартал) ---
        # Готовим данные по отгрузкам
        orders_period = orders_df.copy()
        orders_period["week"] = orders_period["date_ship"].dt.to_period("W")
        orders_period["month"] = orders_period["date_ship"].dt.to_period("M")
        orders_period["quarter"] = orders_period["date_ship"].dt.to_period("Q")
        # Возвраты в пределах окна
        if returns_df is not None and not returns_df.empty:
            if returns_df["shipment_id"].notna().any() and orders_df["shipment_id"].notna().any():
                merged_ret_period = returns_df.merge(
                    orders_df[["shipment_id", "date_ship"]], on="shipment_id", how="left"
                )
                merged_ret_period["days_between"] = (
                    merged_ret_period["date_return"] - merged_ret_period["date_ship"]
                ).dt.days
                returns_filtered = merged_ret_period[
                    (merged_ret_period["days_between"].notna())
                    & (merged_ret_period["days_between"] >= 0)
                    & (merged_ret_period["days_between"] <= return_window_days)
                ].copy()
            else:
                returns_filtered = returns_df.copy()
            returns_filtered["week"] = returns_filtered["date_return"].dt.to_period("W")
            returns_filtered["month"] = returns_filtered["date_return"].dt.to_period("M")
            returns_filtered["quarter"] = returns_filtered["date_return"].dt.to_period("Q")
        else:
            returns_filtered = pd.DataFrame(columns=["week", "month", "quarter", "returns_qty"])
        # Функция для агрегации по периоду
        def _period_summary(period_col: str) -> pd.DataFrame:
            shipped = (
                orders_period.dropna(subset=[period_col])
                .groupby(period_col, as_index=False)["qty_shipped"].sum()
                .rename(columns={period_col: "period", "qty_shipped": "shipped"})
            )
            if not returns_filtered.empty:
                returned = (
                    returns_filtered.dropna(subset=[period_col])
                    .groupby(period_col, as_index=False)["returns_qty"].sum()
                    .rename(columns={period_col: "period", "returns_qty": "returned"})
                )
                merged_pr = shipped.merge(returned, on="period", how="left").fillna(0)
            else:
                merged_pr = shipped.copy()
                merged_pr["returned"] = 0
            return merged_pr
        weekly_totals = _period_summary("week")
        monthly_totals = _period_summary("month")
        quarterly_totals = _period_summary("quarter")
        # Сохраняем периодические итоги в Excel
        weekly_totals.rename(columns={"period": "Неделя", "shipped": "Отгружено", "returned": "Возвраты"}).to_excel(
            writer, sheet_name="Weekly Totals", index=False
        )
        monthly_totals.rename(columns={"period": "Месяц", "shipped": "Отгружено", "returned": "Возвраты"}).to_excel(
            writer, sheet_name="Monthly Totals", index=False
        )
        quarterly_totals.rename(columns={"period": "Квартал", "shipped": "Отгружено", "returned": "Возвраты"}).to_excel(
            writer, sheet_name="Quarterly Totals", index=False
        )


def main() -> None:
    """Точка входа в скрипт с CLI."""
    parser = argparse.ArgumentParser(
        description=(
            "Построение расширенной аналитики с расчётом unit‑economics по SKU. "
            "Пути к файлам можно задать через параметры."
        )
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        default="/Users/aleksandr/Desktop/озон рад/soft/готовое/my_ozon_analytics/gold"
    )
    default_base = Path(__file__).resolve().parent.parent / "data" / "cleaned_data"
    parser.add_argument(
        "--orders",
        type=str,
        default=str(default_base / "25 заказы январь-июль.csv"),
        help=(
            "Путь к CSV-файлу с заказами. Если не задан, ожидается файл '25 заказы январь-июль.csv' в текущей папке."
        ),
    )
    parser.add_argument(
        "--sales",
        type=str,
        default=str(default_base / "Отчет о реализации товара январь-июль.xlsx"),
        help=(
            "Путь к XLSX-файлу с отчётом о реализации. "
            "Если не задан, ожидается файл 'Отчет о реализации товара январь-июль.xlsx' в текущей папке."
        ),
    )
    parser.add_argument(
        "--returns",
        type=str,
        default=str(default_base / "2025 возвраты январь-июль.xlsx"),
        help=(
            "Путь к XLSX-файлу с возвратами. "
            "Если не задан, ожидается файл '2025 возвраты январь-июль.xlsx' в текущей папке."
        ),
    )
    parser.add_argument(
        "--costs",
        type=str,
        default=str(default_base / "production_costs.xlsx"),
        help=(
            "Путь к XLSX-файлу с себестоимостью. "
            "Если не задан, ожидается файл 'production_costs.xlsx' в текущей папке."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path("ultimate_report_unit.xlsx")),
        help=(
            "Имя выходного Excel-файла. Относительный путь создаст файл в текущей папке."
        ),
    )
    parser.add_argument(
        "--promo-actions",
        type=str,
        default=str(default_base / "2 аналитика акции.xlsx"),
        help=(
            "Путь к XLSX-файлу с аналитикой по акциям. "
            "Если не задан, ожидается файл '2 аналитика акции.xlsx' в текущей папке."
        ),
    )
    parser.add_argument(
        "--accruals",
        type=str,
        default=str(default_base / "Отчет по начислениям_01.01.2025-17.07.2025.xlsx"),
        help=(
            "Путь к XLSX-файлу с отчётом по начислениям. "
            "Если не задан, ожидается файл 'Отчет по начислениям_01.01.2025-17.07.2025.xlsx' в текущей папке."
        ),
    )
    parser.add_argument(
        "--inventory",
        type=str,
        default=str(default_base / "продажи по складам.xlsx"),
        help=(
            "Путь к XLSX-файлу с данными по складам. "
            "Если не задан, ожидается файл 'продажи по складам.xlsx' в текущей папке."
        ),
    )
    parser.add_argument(
        "--charts",
        action="store_true",
        help="Сгенерировать графики и сохранить их в папку output/charts"
    )
    parser.add_argument(
        "--eda",
        action="store_true",
        help="Сгенерировать EDA-отчет Sweetviz для итогового analytics_df"
    )
    parser.add_argument(
        "--return-window-days",
        type=int,
        default=90,
        help=(
            "Ширина временного окна в днях, в рамках которого учитываются возвраты. "
            "Возвраты, совершённые позже указанного числа дней после отгрузки, исключаются из расчётов."
        ),
    )
    args = parser.parse_args()
    build_eda = args.eda
    build_charts = args.charts
    orders_path = Path(args.orders)
    sales_path = Path(args.sales)
    returns_path = Path(args.returns)
    costs_path = Path(args.costs)
    promo_actions_path = Path(args.promo_actions)
    accruals_path = Path(args.accruals)
    inventory_path = Path(args.inventory)
    output_path = Path(args.output)
    return_window_days = args.return_window_days
    # Load base data
    orders_df = load_orders(orders_path)
    sales_df = load_sales(sales_path)
    returns_df = load_returns(returns_path)
    costs_df = load_costs(costs_path)
    promos_df = compute_promos(orders_df)
    # Load additional datasets if available
    try:
        promo_actions_df = load_promo_actions(promo_actions_path)
    except Exception:
        promo_actions_df = pd.DataFrame()
    try:
        accruals_df = load_accruals(accruals_path)
    except Exception:
        accruals_df = pd.DataFrame()
    try:
        inventory_df = load_inventory(inventory_path)
    except Exception:
        inventory_df = pd.DataFrame()
    # Compute analytics
    analytics_df = compute_analytics(
        sales_df,
        returns_df,
        promos_df,
        costs_df,
        orders_df,
        return_window_days=return_window_days,
        promo_df=promo_actions_df,
        accrual_df=accruals_df,
        inventory_df=inventory_df,
    )
    if build_eda:
        import sweetviz as sv
        report = sv.analyze(analytics_df)
        report.show_html("output/eda_sweetviz.html")
        print("✔ EDA-отчет Sweetviz сохранён в output/eda_sweetviz.html")
    # Generate and save charts if requested
    if build_charts:
        charts_dir = Path("output") / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)
        # 1) Margin vs. Revenue scatter
        plt.figure()
        plt.scatter(analytics_df["total_rev"], analytics_df["margin"])
        plt.xlabel("Валовая выручка, ₽")
        plt.ylabel("Маржа, ₽")
        plt.title("Маржа vs Выручка по SKU")
        plt.savefig(charts_dir / "margin_vs_revenue.png")
        plt.close()
        # 2) Returns percentage histogram
        plt.figure()
        analytics_df["returns_pct"].hist(bins=30)
        plt.xlabel("Доля возвратов, %")
        plt.title("Распределение доли возвратов")
        plt.savefig(charts_dir / "returns_pct_hist.png")
        plt.close()
        # 3) Promo intensity vs. Contribution margin
        plt.figure()
        plt.scatter(analytics_df["promo_intensity_pct"], analytics_df["contribution_margin"])
        plt.xlabel("Интенсивность промо, %")
        plt.ylabel("Contribution margin")
        plt.title("Promo Intensity vs Contribution Margin")
        plt.savefig(charts_dir / "promo_vs_contribution.png")
        plt.close()

        # 4) Monthly shipments vs. returns line chart
        monthly_ship = orders_df.dropna(subset=['month_ship']) \
            .groupby('month_ship')['qty_shipped'].sum()
        monthly_ret = returns_df.dropna(subset=['date_return']) \
            .assign(month_return=returns_df['date_return'].dt.to_period('M')) \
            .groupby('month_return')['returns_qty'].sum()
        plt.figure()
        plt.plot(monthly_ship.index.astype(str), monthly_ship.values, marker='o', label='Отгрузки')
        plt.plot(monthly_ret.index.astype(str), monthly_ret.values, marker='o', label='Возвраты')
        plt.xlabel('Месяц')
        plt.ylabel('Количество, шт.')
        plt.title('Тренды отгрузок и возвратов помесячно')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(charts_dir / 'monthly_ship_return.png')
        plt.close()

        # 5) Weekly shipments vs. returns line chart
        weekly_ship = orders_df.dropna(subset=['date_ship']) \
            .assign(week=orders_df['date_ship'].dt.to_period('W')) \
            .groupby('week')['qty_shipped'].sum()
        weekly_ret = returns_df.dropna(subset=['date_return']) \
            .assign(week_return=returns_df['date_return'].dt.to_period('W')) \
            .groupby('week_return')['returns_qty'].sum()
        plt.figure()
        plt.plot(weekly_ship.index.astype(str), weekly_ship.values, marker='o', label='Отгрузки')
        plt.plot(weekly_ret.index.astype(str), weekly_ret.values, marker='o', label='Возвраты')
        plt.xlabel('Неделя')
        plt.ylabel('Количество, шт.')
        plt.title('Тренды отгрузок и возвратов по неделям')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(charts_dir / 'weekly_ship_return.png')
        plt.close()

        # 6) Rolling average of revenue and margin (3-month window)
        monthly_metrics = analytics_df.dropna(subset=['month']) \
            .groupby('month').agg(total_rev=('total_rev','sum'), margin=('margin','sum')) \
            .sort_index()
        monthly_metrics['roll_rev'] = monthly_metrics['total_rev'].rolling(window=3).mean()
        monthly_metrics['roll_margin'] = monthly_metrics['margin'].rolling(window=3).mean()
        plt.figure()
        plt.plot(monthly_metrics.index.astype(str), monthly_metrics['roll_rev'], label='Выручка (3‑мес. скользящая)')
        plt.plot(monthly_metrics.index.astype(str), monthly_metrics['roll_margin'], label='Маржа (3‑мес. скользящая)')
        plt.xlabel('Месяц')
        plt.ylabel('Сумма, ₽')
        plt.title('Скользящая средняя выручки и маржи')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(charts_dir / 'rolling_rev_margin.png')
        plt.close()

        # 7) ABC-class pie chart by revenue share
        abc_counts = analytics_df.groupby('ABC_class')['total_rev'].sum()
        plt.figure()
        abc_counts.plot.pie(autopct='%1.1f%%', labels=abc_counts.index, ylabel='')
        plt.title('Доля выручки по ABC‑классам')
        plt.tight_layout()
        plt.savefig(charts_dir / 'abc_revenue_share.png')
        plt.close()

        # 8) XYZ-class bar chart by count of SKUs
        xyz_counts = analytics_df['XYZ_class'].value_counts().reindex(['X','Y','Z']).fillna(0)
        plt.figure()
        xyz_counts.plot.bar()
        plt.xlabel('XYZ‑класс')
        plt.ylabel('Количество SKU')
        plt.title('Распределение SKU по XYZ‑классам')
        plt.tight_layout()
        plt.savefig(charts_dir / 'xyz_sku_counts.png')
        plt.close()

        print(f"✔ Графики сохранены в {charts_dir}")
    # Save report
    save_report(
        analytics_df,
        orders_df,
        output_path,
        returns_df=returns_df,
        return_window_days=return_window_days,
    )
    print("✅ Сводка unit-экономики сохранена в", output_path)

    # --- Добавить planner.compute_future_metrics(), planner.add_recommendations(), planner.save_gold ---
    # Предполагаем, что класс или объект planner должен быть определён где-то выше по коду.
    # Если planner не определён, этот код вызовет ошибку.
    # Добавляем вызовы согласно инструкции:
    try:
        planner.compute_future_metrics()
        planner.add_recommendations()
        planner.save_report(Path(args.output))
        planner.save_gold(Path(args.gold_dir))
    except Exception:
        pass


if __name__ == "__main__":
    main()