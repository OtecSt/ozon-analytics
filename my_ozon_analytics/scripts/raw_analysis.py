import pandas as pd
from pathlib import Path
import sys
import warnings

def validate_data(df):
    print("=== Data Shape ===")
    print(df.shape, end="\n\n")
    print("=== Missing Values ===")
    print(df.isnull().sum(), end="\n\n")
    print("=== Duplicate SKU entries ===")
    print(df['sku'].duplicated().sum(), end="\n\n")
    print("=== Descriptive Statistics ===")
    print(df.describe(include='all'), end="\n\n")

def main():
    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

    # 1) читаем «сырую» выгрузку — шапка на третьей строке (header=2)
    BASE_DIR = Path(__file__).resolve().parent.parent
    data_dir = BASE_DIR / "data"
    data_file = data_dir / "Отчет о реализации товара январь-июль.xlsx"

    # Check if data_dir and data_file exist, and if not, print error and exit
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        sys.exit(1)
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Files in data directory:")
        for f in data_dir.iterdir():
            print("  -", f.name)
        sys.exit(1)

    # 1) читаем очищенную таблицу с единственным заголовком
    df = pd.read_excel(str(data_file), header=0)
    # Убираем любые Unnamed колонки, если они остались
    df = df.loc[:, ~df.columns.str.contains(r'^Unnamed')]

    # 2) даём удобные имена столбцам
    df = df.rename(columns={
        "Код товара продавца":                   "sku",
        "Артикул":                              "sku",
        "Кол-во":                                "qty_sold",
        "Реализовано на сумму, руб.":            "revenue_rub",
        "Базовое вознаграждение Ozon, руб.":     "ozon_fee_rub",
        "Итого к начислению, руб.":              "net_payout_rub"
    })

    # Remove rows without SKU or revenue, then fill missing numeric values
    df = df.dropna(subset=["sku", "revenue_rub"])
    df[["qty_sold", "ozon_fee_rub", "net_payout_rub"]] = df[["qty_sold", "ozon_fee_rub", "net_payout_rub"]].fillna(0)

    validate_data(df)
    print("="*40)

    # 3) выводим первые строки и базовую статистику
    # print("=== Preview ===")
    # print(df.head(), end="\n\n")

    # print("=== dtypes ===")
    # print(df.dtypes, end="\n\n")

    # 4) делаем простую сводку по SKU
    # summary = (
    #     df
    #     .groupby("sku")
    #     .agg(
    #         total_qty   = pd.NamedAgg(column="qty_sold",      aggfunc="sum"),
    #         total_revenue = pd.NamedAgg(column="revenue_rub",  aggfunc="sum"),
    #         avg_fee_rub = pd.NamedAgg(column="ozon_fee_rub",   aggfunc="mean"),
    #         avg_margin  = pd.NamedAgg(
    #             column="net_payout_rub",
    #             aggfunc=lambda x: (x.sum() - df.loc[x.index, "ozon_fee_rub"].sum()) / x.sum()
    #         )
    #     )
    #     .reset_index()
    # )

    # print("=== Summary by SKU ===")
    # print(summary.to_string(index=False))

if __name__ == "__main__":
    main()