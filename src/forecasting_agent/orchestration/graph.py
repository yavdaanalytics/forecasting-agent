from __future__ import annotations

from typing import Any, TypedDict

from forecasting_agent.orchestration.pipeline import ForecastPipeline


class ForecastState(TypedDict, total=False):
    brand: str
    result: Any
    error: str | None


def build_graph(pipeline: ForecastPipeline):
    """LangGraph adapter. Optional extra: pip install 'forecasting-agent[agent]'."""
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "langgraph is not installed. pip install 'forecasting-agent[agent]'"
        ) from exc

    def run_node(state: ForecastState) -> ForecastState:
        result = pipeline.run(state.get("brand"))
        return {"brand": pipeline.brand_name, "result": result, "error": None}

    graph = StateGraph(ForecastState)
    graph.add_node("pipeline", run_node)
    graph.set_entry_point("pipeline")
    graph.add_edge("pipeline", END)
    return graph.compile()
