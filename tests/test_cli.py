from pathlib import Path

from forecasting_agent.cli import main
from forecasting_agent.connectors.csv_store import CsvSalesStore


def test_csv_store_and_cli(tmp_path: Path):
    csv_path = tmp_path / "sales.csv"
    csv_path.write_text(
        "sku,date,qty,brand\n"
        "PLANT-001,2024-01-01,10,TAOS\n"
        "PLANT-001,2024-01-02,11,TAOS\n"
        "PLANT-001,2024-01-03,9,TAOS\n"
        + "".join(f"PLANT-001,2024-01-{d:02d},{10},TAOS\n" for d in range(4, 32))
        + "".join(f"PLANT-001,2024-02-{d:02d},{10},TAOS\n" for d in range(1, 29))
        + "".join(f"PLANT-001,2024-03-{d:02d},{10},TAOS\n" for d in range(1, 32)),
        encoding="utf-8",
    )
    series = CsvSalesStore(csv_path).load("TAOS")
    assert len(series) == 1
    assert series[0].sku == "PLANT-001"
    out = tmp_path / "out.json"
    assert main(["--input", str(csv_path), "--brand", "TAOS", "--output", str(out)]) == 0
    assert out.exists()
    assert "TAOS" in out.read_text(encoding="utf-8")
