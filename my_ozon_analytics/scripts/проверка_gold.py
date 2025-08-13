# проверка_gold.py
from pathlib import Path
import pandas as pd
import csv

ПАПКА_GOLD = Path("gold")

ОЖИДАЕМЫЕ_КОЛОНКИ = {
    "fact_sku_daily.csv": [
        "date","sku","shipped_qty","promo_rub","order_value_rub_sum","shipments","returns_qty","returns_rub"
    ],
    "fact_sku_monthly.csv": [
        "period","sku","shipped_qty","promo_rub","order_value_rub_sum","returns_qty","returns_rub"
    ],
    "mart_unit_econ.csv": [  # проверяем, что хотя бы эти колонки есть
        "sku","total_qty","total_rev","total_fee","total_payout",
        "returns_qty","returns_rub","promo_cost","production_cost","net_qty",
        "net_revenue","margin","avg_price_per_unit","commission_per_unit",
        "promo_per_unit","margin_per_unit","margin_pct","returns_pct",
        "promo_intensity_pct","recommended_action"
    ],
    "data_dictionary.csv": ["table","column","dtype","description"],
}

ЧИСЛОВЫЕ_ПОЛЯ = {
    "fact_sku_daily.csv": ["shipped_qty","promo_rub","order_value_rub_sum","shipments","returns_qty","returns_rub"],
    "fact_sku_monthly.csv": ["shipped_qty","promo_rub","order_value_rub_sum","returns_qty","returns_rub"],
}

def проверить_bom_и_разделитель(путь: Path):
    with open(путь, "rb") as f:
        первые3 = f.read(3)
    assert первые3 == b"\xef\xbb\xbf", f"{путь.name}: нет BOM utf-8-sig"
    # Быстрый сниффер разделителя
    with open(путь, "r", encoding="utf-8-sig") as f:
        sample = f.read(2048)
    dialect = csv.Sniffer().sniff(sample)
    assert dialect.delimiter == ",", f"{путь.name}: ожидался разделитель ','"

def проверить_схему(путь: Path, ожид: list[str]):
    df = pd.read_csv(путь, encoding="utf-8-sig")
    if путь.name == "mart_unit_econ.csv":
        # Подмножество: все перечисленные поля должны присутствовать
        отсутствуют = [c for c in ожид if c not in df.columns]
        assert not отсутствуют, f"{путь.name}: нет колонок {отсутствуют}"
    else:
        assert list(df.columns) == ожид, f"{путь.name}: несоответствие колонок"
    return df

def проверить_пустоты_и_знаки(df: pd.DataFrame, числовые: list[str], файл: str):
    for c in числовые:
        assert df[c].isna().sum() == 0, f"{файл}: NaN в {c}"
        if c.endswith("_qty") or c in {"shipments"}:
            assert (df[c] >= 0).all(), f"{файл}: отрицательные значения в {c}"

def проверить_непрерывность_месяцев(df_m: pd.DataFrame):
    # по каждой SKU проверим, что месяцы без дыр между min..max
    df = df_m.copy()
    df["period"] = pd.PeriodIndex(df["period"], freq="M")
    for sku, блок in df.groupby("sku"):
        период = pd.period_range(блок["period"].min(), блок["period"].max(), freq="M")
        имеющиеся = set(блок["period"])
        пропуски = [p for p in период if p not in имеющиеся]
        assert not пропуски, f"fact_sku_monthly.csv: у SKU {sku} пропущены месяцы {пропуски}"

if __name__ == "__main__":
    ошибок = []
    for имя, ожид in ОЖИДАЕМЫЕ_КОЛОНКИ.items():
        путь = ПАПКА_GOLD / имя
        try:
            проверить_bom_и_разделитель(путь)
            df = проверить_схему(путь, ожид)
            if имя in ЧИСЛОВЫЕ_ПОЛЯ:
                проверить_пустоты_и_знаки(df, ЧИСЛОВЫЕ_ПОЛЯ[имя], имя)
            if имя == "fact_sku_monthly.csv":
                проверить_непрерывность_месяцев(df)
        except AssertionError as e:
            ошибок.append(str(e))
    if ошибок:
        print("❌ Найдены проблемы:")
        for e in ошибок: print(" -", e)
        raise SystemExit(1)
    print("✔ Все проверки пройдены")