# components/__init__.py
# Инициализация пакета components

from .kpis import kpi_row, kpi_card  # импорт нужных функций/классов
from .charts import render_chart     # импорт ключевых функций из charts

__all__ = [
    "kpi_row",
    "kpi_card",
    "render_chart"
]