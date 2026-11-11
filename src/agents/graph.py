"""LangGraph multi-agent orchestration graph definition for OmniComply."""

from __future__ import annotations

from typing import Any

import structlog
from langgraph.graph import END, StateGraph

from src.agents.audio_agent import AudioAgent
from src.agents.compliance_agent import ComplianceAgent
from src.agents.document_agent import DocumentAgent
from src.agents.query_agent import QueryAgent
from src.agents.report_agent import ReportAgent
from src.agents.router import RouterAgent
from src.agents.state import AgentState

logger = structlog.get_logger(__name__)


def build_compliance_graph():
    """Construct and compile the stateful LangGraph multi-agent workflow."""

    # 1. Instantiate specialized agents
    router = RouterAgent()
    doc_agent = DocumentAgent()
    audio_agent = AudioAgent()
    query_agent = QueryAgent()
    compliance_agent = ComplianceAgent()
    report_agent = ReportAgent()

    # 2. Initialize StateGraph with AgentState
    workflow = StateGraph(AgentState)

    # 3. Define node functions
    async def node_router(state: AgentState) -> AgentState:
        destination = router.route(state)
        state["next_step"] = destination
        return state

    async def node_document(state: AgentState) -> AgentState:
        return await doc_agent.process(state)

    async def node_audio(state: AgentState) -> AgentState:
        return await audio_agent.process(state)

    async def node_query(state: AgentState) -> AgentState:
        return await query_agent.process(state)

    async def node_compliance(state: AgentState) -> AgentState:
        return await compliance_agent.process(state)

    async def node_report(state: AgentState) -> AgentState:
        return await report_agent.process(state)

    # 4. Add nodes to graph
    workflow.add_node("router", node_router)
    workflow.add_node("document_agent", node_document)
    workflow.add_node("audio_agent", node_audio)
    workflow.add_node("query_agent", node_query)
    workflow.add_node("compliance_agent", node_compliance)
    workflow.add_node("report_agent", node_report)

    # 5. Set entry point
    workflow.set_entry_point("router")

    # 6. Define conditional edges from router
    def route_decision(state: AgentState) -> str:
        return state.get("next_step", "query_agent")

    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "document_agent": "document_agent",
            "audio_agent": "audio_agent",
            "query_agent": "query_agent",
            "compliance_agent": "compliance_agent",
            "report_agent": "report_agent",
        },
    )

    # Ingestion leads to compliance analysis
    workflow.add_edge("document_agent", "compliance_agent")
    workflow.add_edge("audio_agent", "compliance_agent")

    # Compliance conditionally routes to ReportAgent or END
    def compliance_decision(state: AgentState) -> str:
        if state.get("next_step") == "report_agent":
            return "report_agent"
        return END

    workflow.add_conditional_edges(
        "compliance_agent",
        compliance_decision,
        {
            "report_agent": "report_agent",
            END: END,
        },
    )

    # Terminations
    workflow.add_edge("query_agent", END)
    workflow.add_edge("report_agent", END)

    # 7. Compile executable app
    app = workflow.compile()
    logger.info("compliance_graph_compiled_successfully")
    return app


# Pre-compiled application instance
compliance_graph = build_compliance_graph()


async def run_graph(initial_state: dict[str, Any]) -> dict[str, Any]:
    """Execute the multi-agent graph with an initial dictionary state."""
    state_payload: AgentState = {
        "messages": initial_state.get("messages", []),
        "input_type": initial_state.get("input_type", "query"),
        "file_path": initial_state.get("file_path"),
        "query": initial_state.get("query"),
        "metadata": initial_state.get("metadata", {}),
        "retry_count": 0,
    }
    result = await compliance_graph.ainvoke(state_payload)
    return result
