from __future__ import annotations

import argparse
import json
from pathlib import Path

from forecasting_agent.connectors.csv_store import CsvSalesStore
from forecasting_agent.orchestration.pipeline import ForecastPipeline
from forecasting_agent.reporting.accuracy import pipeline_as_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the forecasting agent pipeline")
    parser.add_argument("--input", required=True, help="CSV with sku,date,qty columns")
    parser.add_argument("--brand", default="TAOS")
    parser.add_argument("--output", default=None, help="Write JSON summary here")
    parser.add_argument("--no-validate", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = CsvSalesStore(args.input)
    pipeline = ForecastPipeline(store, brand=args.brand, validate=not args.no_validate)
    result = pipeline.run()
    payload = pipeline_as_dict(result)
    text = json.dumps(payload, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
