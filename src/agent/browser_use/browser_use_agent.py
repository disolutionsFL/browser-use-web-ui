"""
Custom BrowserUseAgent -- thin subclass of browser-use v0.12.0 Agent.

The base Agent now handles signal handling, hooks, pause/resume, GIF generation,
and Playwright script saving. This wrapper exists for future customization hooks.
"""

from __future__ import annotations

import logging
import os

from browser_use.agent.service import Agent, AgentHookFunc
from browser_use.agent.views import (
    ActionResult,
    AgentHistory,
    AgentHistoryList,
    AgentStepInfo,
)

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

SKIP_LLM_API_KEY_VERIFICATION = (
    os.environ.get("SKIP_LLM_API_KEY_VERIFICATION", "false").lower()[0] in "ty1"
)


class BrowserUseAgent(Agent):
    """Thin wrapper over browser-use Agent for project-specific customizations.

    The base Agent.run() now includes:
    - SignalHandler setup/teardown
    - on_step_start / on_step_end callbacks
    - Pause / resume / stop handling
    - Max failures check
    - Playwright script saving
    - GIF generation

    Add custom hooks or overrides here as needed.
    """
    pass
