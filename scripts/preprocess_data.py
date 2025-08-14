DATE_SYNONYMS = [
    "дата", "дата отгрузки", "дата возврата", "дата оформления", "дата оформления заказа",
    "shipment_date", "return_date", "order_date"
]
import pandas as pd
from pathlib import Path

# Mapping of raw column name fragments to standardized column names
COLUMN_SYNONYMS = {
    "код товара продавца": "sku",
    "артикул": "sku",
    "sku": "sku",
    "штрих-код": "barcode",
    "кол-во": "qty_sold",
    "реализовано на сумму": "revenue_rub",
    "выплаты по механикам лояльности": "loyalty_payout_rub",
    "баллы за скидки": "discount_points",
    "цена реализации": "price_rub",
    "базовое вознаграждение ozon": "ozon_fee_rub",
    "итого к начислению": "net_payout_rub",
    "продажи": "revenue_rub",
    "возврат": "returns_rub",
    # Add more synonyms as needed...
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns by matching fragments in COLUMN_SYNONYMS keys,
    mapping them to standardized names.
    """
    rename_map = {}
    for raw_frag, std_name in COLUMN_SYNONYMS.items():
        for col in df.columns:
            if raw_frag in str(col).strip().lower():
                rename_map[col] = std_name
    df = df.rename(columns=rename_map)
    if rename_map:
        print(f"[preprocess] Renamed columns: {rename_map}")
    else:
        print("[preprocess] No columns renamed")
    # Приведение всех дат к формату datetime
    for col in df.columns:
        for d_frag in DATE_SYNONYMS:
            if d_frag in col.lower():
                df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    return df

def run_preprocessing(input_dir: Path = None, output_dir: Path = None):
    """
    Process all .xlsx and .csv files in input_dir,
    normalize their column names, and save to output_dir.
    """
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = input_dir or (base_dir / "data")
    clean_dir = output_dir or (data_dir / "cleaned_data")

    if not data_dir.exists():
        raise FileNotFoundError(f"Input data directory not found: {data_dir}")
    clean_dir.mkdir(parents=True, exist_ok=True)

    for file_path in data_dir.iterdir():
        # Skip Excel temp files
        if file_path.name.startswith("~$"):
            continue
        if not file_path.is_file():
            continue
        suffix = file_path.suffix.lower()
        if suffix not in {".xlsx", ".csv"}:
            continue

        print(f"[preprocess] Processing {file_path.name}")
        if suffix == ".xlsx":
            try:
                df = pd.read_excel(file_path)
            except Exception as e:
                print(f"[preprocess] Skipping {file_path.name}: cannot read Excel file ({e})")
                continue
        else:
            df = pd.read_csv(file_path)

        df_clean = normalize_columns(df)

        out_path = clean_dir / file_path.name
        if suffix == ".xlsx":
            df_clean.to_excel(out_path, index=False)
        else:
            df_clean.to_csv(out_path, index=False)
        print(f"[preprocess] Saved cleaned file to: {out_path}")

if __name__ == "__main__":
    run_preprocessing()