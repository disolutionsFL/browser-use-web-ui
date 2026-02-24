"""
browser-use MCP Server

Wraps the browser-use agent as MCP tools for OpenClaw integration.
Runs as a separate process from the Gradio web UI, sharing the same venv.

Usage:
    python mcp_server.py                            # Default: 0.0.0.0:8934
    python mcp_server.py --port 8934 --host 0.0.0.0

systemd (BB8):
    xvfb-run --auto-servernum python mcp_server.py --host 0.0.0.0
"""

import asyncio
import logging
import os
import sys
import argparse
from datetime import datetime
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from mcp.server.fastmcp import FastMCP

from browser_use.browser.browser import BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import CustomBrowserContext
from src.controller.custom_controller import CustomController
from src.agent.browser_use.browser_use_agent import BrowserUseAgent
from src.utils.llm_provider import get_llm_model

logger = logging.getLogger("browser-use-mcp")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

# --- Configuration from environment ---
DEFAULT_OLLAMA_ENDPOINT = os.getenv(
    "OLLAMA_ENDPOINT", "http://localhost:11434"
)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen3-coder-30b-96k:latest")
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.6"))
DEFAULT_NUM_CTX = int(os.getenv("DEFAULT_NUM_CTX", "16000"))
TASK_TIMEOUT = int(os.getenv("TASK_TIMEOUT", "300"))

# --- Concurrency control ---
_task_lock = asyncio.Lock()
_current_status: dict = {
    "running": False,
    "task": None,
    "step": 0,
    "max_steps": 0,
    "started_at": None,
    "agent": None,
}

# --- MCP Server ---
# In mcp SDK 1.6.x, host/port are constructor settings; run() only takes transport.
# Parse CLI args early so we can pass host/port to FastMCP constructor.
_parser = argparse.ArgumentParser(description="browser-use MCP Server")
_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
_parser.add_argument("--port", type=int, default=8934, help="Port to listen on")
_args = _parser.parse_args()

server = FastMCP("browser-use", host=_args.host, port=_args.port)


@server.tool()
async def run_task(
    task: str,
    model: Optional[str] = None,
    max_steps: int = 10,
    use_vision: bool = True,
    max_actions_per_step: int = 10,
    ollama_base_url: Optional[str] = None,
    num_ctx: Optional[int] = None,
) -> str:
    """Run a browser automation task using the browser-use agent.

    The agent autonomously navigates websites, fills forms, clicks buttons,
    and extracts information. Returns the final result text.

    Args:
        task: Natural language description of the browser task to perform.
        model: Ollama model name. Defaults to server's DEFAULT_MODEL env var.
        max_steps: Maximum agent reasoning steps before stopping.
        use_vision: Use screenshots for page understanding.
        max_actions_per_step: Max browser actions per reasoning step.
        ollama_base_url: Override Ollama endpoint URL.
        num_ctx: Ollama context window size.
    """
    if _task_lock.locked():
        return (
            "ERROR: A task is already running. "
            "Use get_status to check progress or stop_task to cancel."
        )

    async with _task_lock:
        browser = None
        context = None
        try:
            _current_status.update(
                {
                    "running": True,
                    "task": task,
                    "step": 0,
                    "max_steps": max_steps,
                    "started_at": datetime.now().isoformat(),
                }
            )

            llm = get_llm_model(
                provider="ollama",
                model_name=model or DEFAULT_MODEL,
                temperature=DEFAULT_TEMPERATURE,
                base_url=ollama_base_url or DEFAULT_OLLAMA_ENDPOINT,
                num_ctx=num_ctx or DEFAULT_NUM_CTX,
            )

            browser = CustomBrowser(
                config=BrowserConfig(
                    headless=True,
                    new_context_config=BrowserContextConfig(
                        window_width=1280,
                        window_height=1100,
                    ),
                )
            )

            context = await browser.new_context(
                config=BrowserContextConfig(
                    window_width=1280,
                    window_height=1100,
                )
            )

            controller = CustomController()

            async def on_step_end(agent):
                _current_status["step"] = agent.state.n_steps

            agent = BrowserUseAgent(
                task=task,
                llm=llm,
                browser=browser,
                browser_context=context,
                controller=controller,
                use_vision=use_vision,
                max_actions_per_step=max_actions_per_step,
                enable_memory=False,
            )
            _current_status["agent"] = agent

            logger.info(
                "Running task: %s (model=%s, max_steps=%d)",
                task,
                model or DEFAULT_MODEL,
                max_steps,
            )

            history = await asyncio.wait_for(
                agent.run(max_steps=max_steps, on_step_end=on_step_end),
                timeout=TASK_TIMEOUT,
            )

            final = history.final_result()
            if final:
                return f"Task completed.\n\nResult:\n{final}"

            errors = history.errors()
            if errors:
                return "Task finished with errors:\n" + "\n".join(
                    str(e) for e in errors[-3:]
                )

            return "Task completed but no explicit result was returned by the agent."

        except asyncio.TimeoutError:
            return f"ERROR: Task timed out after {TASK_TIMEOUT} seconds."
        except Exception as e:
            logger.exception("Task failed")
            return f"ERROR: {type(e).__name__}: {e}"
        finally:
            _current_status.update(
                {"running": False, "task": None, "agent": None}
            )
            if context:
                try:
                    await context.close()
                except Exception:
                    pass
            if browser:
                try:
                    await browser.close()
                except Exception:
                    pass


@server.tool()
async def get_status() -> str:
    """Check the status of the currently running browser-use task.

    Returns whether a task is running, which step it's on, and the task description.
    """
    if not _current_status["running"]:
        return "No task is currently running."

    return (
        f"Task: {_current_status['task']}\n"
        f"Step: {_current_status['step']}/{_current_status['max_steps']}\n"
        f"Started: {_current_status['started_at']}"
    )


@server.tool()
async def stop_task() -> str:
    """Stop the currently running browser-use task.

    Signals the agent to stop after its current step completes.
    """
    agent = _current_status.get("agent")
    if not agent or not _current_status["running"]:
        return "No task is currently running."

    agent.state.stopped = True
    return "Stop signal sent. Task will finish after current step completes."


def main():
    logger.info("Starting browser-use MCP server on %s:%d", _args.host, _args.port)
    logger.info("Default model: %s", DEFAULT_MODEL)
    logger.info("Default Ollama endpoint: %s", DEFAULT_OLLAMA_ENDPOINT)
    logger.info("Task timeout: %ds", TASK_TIMEOUT)

    server.run(transport="sse")


if __name__ == "__main__":
    main()
