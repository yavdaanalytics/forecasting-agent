# Development

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux
pip install -e ".[dev]"
pytest
```

Optional extras:

- `pip install -e ".[prophet]"` — local Prophet
- `pip install -e ".[gcp]"` — BigQuery connector (not required for CSV/MVP)
- `pip install -e ".[agent]"` — LangGraph adapter around `ForecastPipeline`

Layout: domain types and metrics have no I/O; connectors implement `SalesStore`; methods implement `ForecastMethod`; `ForecastPipeline` only wires those ports.

```bash
python -m forecasting_agent --input examples/sample_taos_input.csv --brand TAOS
```
