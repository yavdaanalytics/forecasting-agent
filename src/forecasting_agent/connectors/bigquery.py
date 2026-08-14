from __future__ import annotations

from forecasting_agent.domain.types import SalesSeries


class UnconfiguredBigQueryStore:
    """Placeholder until GCP credentials and table names are wired."""

    def __init__(self, project: str | None = None, table: str | None = None) -> None:
        self.project = project
        self.table = table

    def load(self, brand: str | None = None) -> list[SalesSeries]:
        raise RuntimeError(
            "BigQuery store is not configured. Use CsvSalesStore or MemorySalesStore, "
            "or implement a query against the brand history table."
        )
