#!/usr/bin/env python3
"""
Единый конвейер для запуска по порядку:
1) sku_analytics.py
2) forecast_planning.py
3) build_gold.py

Читает параметры из pipeline_config.yaml.
Логи падают в logs/pipeline.log. Возвращает ненулевой код при ошибке любого шага.
"""

from __future__ import annotations
import subprocess
import sys
import yaml
from pathlib import Path
from datetime import datetime

THIS = Path(__file__).resolve()
ROOT = THIS.parent
LOGS = ROOT / "logs"
LOGS.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOGS / "pipeline.log"


def log(msg: str) -> None:
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    log(f"RUN: {' '.join(cmd)}  (cwd={cwd or ROOT})")
    proc = subprocess.Popen(cmd, cwd=str(cwd or ROOT))
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"Команда завершилась с кодом {ret}: {' '.join(cmd)}")


def main() -> int:
    cfg_path = ROOT / "pipeline_config.yaml"
    if not cfg_path.exists():
        log(f"Не найден конфиг: {cfg_path}")
        return 2

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    py = cfg.get("python_bin", sys.executable)

    # Пути
    scripts_dir = Path(cfg["paths"]["scripts_dir"]).expanduser()
    data_dir    = Path(cfg["paths"]["data_dir"]).expanduser()
    gold_dir    = Path(cfg["paths"]["gold_dir"]).expanduser()
    reports_dir = Path(cfg["paths"]["reports_dir"]).expanduser()
    planned_inbound = cfg["paths"].get("planned_inbound")

    gold_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Файлы данных
    orders   = str((data_dir / cfg["files"]["orders"]).expanduser())
    sales    = str((data_dir / cfg["files"]["sales"]).expanduser())
    returns  = str((data_dir / cfg["files"]["returns"]).expanduser())
    costs    = str((data_dir / cfg["files"]["costs"]).expanduser())
    promo    = str((data_dir / cfg["files"]["promo_actions"]).expanduser())
    accruals = str((data_dir / cfg["files"]["accruals"]).expanduser())
    inv      = str((data_dir / cfg["files"]["inventory"]).expanduser())

    # Скрипты
    sku_analytics = str((scripts_dir / "sku_analytics.py").expanduser())
    forecast_planning = str((scripts_dir / "forecast_planning.py").expanduser())
    build_gold = str((scripts_dir / "build_gold.py").expanduser())

    # Параметры
    return_window_days = str(cfg.get("params", {}).get("return_window_days", 90))
    horizon   = str(cfg.get("forecast", {}).get("horizon", 3))
    model     = str(cfg.get("forecast", {}).get("model", "ets"))
    backtest  = str(cfg.get("forecast", {}).get("backtest", 0))
    price_drift = str(cfg.get("assumptions", {}).get("price_drift", 0.0))
    promo_delta = str(cfg.get("assumptions", {}).get("promo_delta_pp", 0.0))
    comm_delta  = str(cfg.get("assumptions", {}).get("commission_delta_pp", 0.0))
    ret_delta   = str(cfg.get("assumptions", {}).get("returns_delta_pp", 0.0))
    capacity    = cfg.get("assumptions", {}).get("capacity_limit")
    min_margin  = str(cfg.get("assumptions", {}).get("min_margin_pct", 0.05))
    min_batch   = str(cfg.get("assumptions", {}).get("min_batch", 1))

    gold_dir_path = str(gold_dir)

    # Выходные файлы
    unit_report = str((reports_dir / "ultimate_report_unit.xlsx").expanduser())
    forecast_report = str((reports_dir / "forecast_planning_report.xlsx").expanduser())

    # 1) sku_analytics
    cmd1 = [
        py, sku_analytics,
        "--orders", orders,
        "--sales", sales,
        "--returns", returns,
        "--costs", costs,
        "--promo-actions", promo,
        "--accruals", accruals,
        "--inventory", inv,
        "--output", unit_report,
        "--return-window-days", return_window_days
    ]
    if cfg.get("analytics", {}).get("charts", False):
        cmd1.append("--charts")
    if cfg.get("analytics", {}).get("eda", False):
        cmd1.append("--eda")
    run(cmd1)

    # 2) forecast_planning
    cmd2 = [
        py, forecast_planning,
        "--orders", orders,
        "--sales", sales,
        "--returns", returns,
        "--costs", costs,
        "--horizon", horizon,
        "--model", model,
        "--backtest", backtest,
        "--price-drift", price_drift,
        "--promo-delta-pp", promo_delta,
        "--commission-delta-pp", comm_delta,
        "--returns-delta-pp", ret_delta,
        "--min-margin-pct", min_margin,
        "--min-batch", min_batch,
        "--output", forecast_report,
        "--gold-dir", gold_dir_path
    ]
    if planned_inbound:
        cmd2 += ["--planned-inbound", str(Path(planned_inbound).expanduser())]
    run(cmd2)

    # 3) build_gold
    cmd3 = [
        py, build_gold,
        "--orders", orders,
        "--sales", sales,
        "--returns", returns,
        "--costs", costs,
        "--promo-actions", promo,
        "--accruals", accruals,
        "--inventory", inv,
        "--output-dir", str(gold_dir),
        "--return-window-days", return_window_days
    ]
    run(cmd3)

    log("Пайплайн завершён успешно.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        log(f"ОШИБКА: {e}")
        sys.exit(1)
