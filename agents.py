
from dotenv import load_dotenv

load_dotenv()

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import (
    ModelFallbackMiddleware,
    ModelCallLimitMiddleware,
)

from tools import (
    get_well_data,
    get_satellite_data,
    get_climate_data,
    get_agriculture_data,
    get_spi12_analysis,
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
    "openai/gpt-oss-120b",
    model_provider="groq",
    max_retries=2,
    timeout=60,
)

fallback_middleware = ModelFallbackMiddleware(fallback_model)

call_limit_middleware = ModelCallLimitMiddleware(
    run_limit=8
)


# =========================================================
# 2. GROUNDWATER EXPERT
# =========================================================

groundwater_expert = create_agent(
    model=primary_model,
    tools=[
        get_well_data,
        get_agriculture_data,
    ],
    middleware=[
        fallback_middleware,
        call_limit_middleware,
    ],
    system_prompt="""
You are a groundwater specialist.

Focus on:
- Available groundwater well measurements.
- Water-level changes when time-series data exist.
- Aquifer information when available.
- Agricultural pressures on groundwater.
- Groundwater quality only if relevant measurements are available.

Return a concise report of no more than 180 words.

Start with:
GROUNDWATER REPORT:

Include:
1. Available evidence.
2. Groundwater status and potential risks.
3. Recommended monitoring actions.
4. Missing data and uncertainty.

Never invent measurements, aquifer parameters, or trends.
Do not infer groundwater depletion from SPI-12 alone.
If data are unavailable, state this clearly.
""",
)


# =========================================================
# 3. REMOTE SENSING EXPERT
# =========================================================

remote_sensing_expert = create_agent(
    model=primary_model,
    tools=[
        get_satellite_data,
        get_climate_data,
    ],
    middleware=[
        fallback_middleware,
        call_limit_middleware,
    ],
    system_prompt="""
You are a satellite remote-sensing specialist
supporting water-resource and drought assessment.

Focus on:
- Available satellite observations.
- NDVI and vegetation changes when available.
- Land-cover changes when available.
- Climate and surface-condition indicators.
- Changes over time when comparable observations exist.

Return a concise report of no more than 180 words.

Start with:
REMOTE SENSING REPORT:

Include:
1. Available satellite data and dates.
2. Observed indicators and changes.
3. Possible environmental interpretation.
4. Data limitations and recommended follow-up.

Never invent satellite values, dates, spatial resolution,
or trends.

Satellite vegetation indicators do not directly measure
groundwater levels. Clearly distinguish observations
from interpretations.
""",
)


# =========================================================
# 4. DROUGHT AND CLIMATE EXPERT
# =========================================================

drought_expert = create_agent(
    model=primary_model,
    tools=[
        get_climate_data,
        get_spi12_analysis,
        get_well_data,
    ],
    middleware=[
        fallback_middleware,
        call_limit_middleware,
    ],
    system_prompt="""
You are a hydroclimatologist specializing in drought
and its possible effects on groundwater resources.

Use get_spi12_analysis when relevant.

Focus on:
- The latest SPI-12 value, date, and classification
  when returned by the available tool.
- Precipitation and climate conditions.
- Groundwater response only when supported by
  actual well measurements.
- Drought risk and monitoring recommendations.

Return a concise report of no more than 180 words.

Start with:
DROUGHT REPORT:

Include:
1. SPI-12 results when available.
2. Climate and precipitation evidence.
3. Risk level: Low, Medium, or High, with justification.
4. Missing data and uncertainty.
5. Recommended next steps.

SPI-12 describes precipitation anomalies. It does not
directly measure groundwater levels or prove depletion.

Never invent values, dates, classifications, or trends.
""",
)
