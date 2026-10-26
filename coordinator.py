from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent, AgentState
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langchain.messages import HumanMessage, ToolMessage
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelFallbackMiddleware,
    ModelCallLimitMiddleware,
)

from agents import (
    groundwater_expert,
    remote_sensing_expert,
    drought_expert,
)


# =========================================================
# 1. AI MODELS
# =========================================================

primary_model = init_chat_model(
    "gemini-3.8-flash",
    model_provider="google_genai",
    max_retries=3,
    timeout=60,
)

fallback_model = init_chat_model(
    "qwen/qwen3.8-27b",
    model_provider="groq",
    max_retries=2,
    timeout=60,
    max_tokens=800,
)

fallback_middleware = ModelFallbackMiddleware(fallback_model)

call_limit_middleware = ModelCallLimitMiddleware(
    run_limit=12
)


# =========================================================
# 2. COORDINATOR STATE
# =========================================================

class HydroState(AgentState):
    region_name: str
    final_action_plan: str


# =========================================================
# 3. GROUNDWATER ANALYSIS TOOL
# =========================================================

@tool
async def analyze_groundwater(region: str) -> str:
    """Analyze available groundwater well and aquifer evidence."""

    result = await groundwater_expert.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Analyze groundwater evidence for {region}. "
                        "Use available well and agricultural data. "
                        "Keep the report concise and identify "
                        "missing measurements."
                    )
                )
            ]
        }
    )

    return result["messages"][-1].content


# =========================================================
# 4. REMOTE SENSING ANALYSIS TOOL
# =========================================================

@tool
async def analyze_remote_sensing(region: str) -> str:
    """Analyze available satellite and remote-sensing indicators."""

    result = await remote_sensing_expert.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Analyze available satellite and remote-sensing "
                        f"evidence for {region}. Report available "
                        "indicators and dates, explain their limitations, "
                        "and keep the report concise."
                    )
                )
            ]
        }
    )

    return result["messages"][-1].content


# =========================================================
# 5. DROUGHT ANALYSIS TOOL
# =========================================================

@tool
async def analyze_drought(region: str) -> str:
    """Analyze SPI-12, climate conditions, and groundwater evidence."""

    result = await drought_expert.ainvoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        f"Assess meteorological drought for {region}. "
                        "Use SPI-12 and climate tools when available. "
                        "Report the latest value, date, and classification "
                        "only if provided by the tools. Distinguish "
                        "meteorological drought from groundwater depletion. "
                        "Keep the report concise."
                    )
                )
            ]
        }
    )

    return result["messages"][-1].content


# =========================================================
# 6. HUMAN APPROVAL TOOL
# =========================================================

@tool
def execute_action_plan(
    action_plan: str,
    runtime: ToolRuntime,
) -> Command:
    """Submit the proposed water-management plan for human approval."""

    return Command(
        update={
            "final_action_plan": action_plan,
            "messages": [
                ToolMessage(
                    content=(
                        "The action plan was submitted through the "
                        "human-approval workflow. This does not mean "
                        "that any external action was performed."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


hitl_middleware = HumanInTheLoopMiddleware(
    interrupt_on={
        "execute_action_plan": True,
    },
    description_prefix=(
        "WATER RESOURCE DECISION: Requires Human Approval"
    ),
)


# =========================================================
# 7. HYDRODECISION AI COORDINATOR
# =========================================================

coordinator = create_agent(
    model=primary_model,
    tools=[
        analyze_groundwater,
        analyze_remote_sensing,
        analyze_drought,
        execute_action_plan,
    ],
    state_schema=HydroState,
    middleware=[
        hitl_middleware,
        fallback_middleware,
        call_limit_middleware,
    ],
    checkpointer=InMemorySaver(),
    system_prompt="""
You are HydroDecision AI, a scientific groundwater
management decision-support coordinator.

For every requested region, follow this sequence:

STEP 1 — COLLECT EVIDENCE
Call each of these three tools separately:
- analyze_groundwater
- analyze_remote_sensing
- analyze_drought

Pass the requested region name to every tool.
Wait for and read each tool's actual result.

STEP 2 — SYNTHESIZE THE RESULTS
Write a genuine scientific report based on the returned
specialist reports. Do not substitute tool calls,
tool arguments, or JSON instructions for the report.

The report must contain:
1. Groundwater evidence and missing measurements.
2. Climate and SPI-12 evidence, if available.
3. Satellite evidence, if available.
4. Risks and uncertainties.
5. Practical, evidence-based recommendations.

Clearly label measured evidence, interpretations,
assumptions, and missing data.

Never invent observations, values, dates, or sources.
Do not claim that a tool succeeded unless its result
was actually received.

STEP 3 — SUBMIT THE PLAN FOR HUMAN REVIEW
After synthesizing the specialist results, call
execute_action_plan exactly once.

Its action_plan argument must contain the complete
human-readable management plan, not a tool call,
not a JSON object describing a tool, and not a
placeholder.

The human approval step must happen only after the
report and recommendations have been prepared.

Never claim that approving a plan performs real-world
actions. Report honestly what was executed.

If a specialist fails, report that failure and identify
the missing evidence. Do not fabricate its findings.
""",
)
