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
import time
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

# --- Concurrency control & state ---
_task_lock = asyncio.Lock()
_current_status: dict = {
    "running": False,
    "task": None,
    "step": 0,
    "max_steps": 0,
    "started_at": None,
    "agent": None,
    # Persisted after task completes so get_status can report last result
    "last_status": None,       # "completed" | "failed" | "stopped" | "timeout" | "max_steps"
    "last_task": None,
    "last_result": None,
    "last_error": None,
    "last_finished_at": None,
    "last_steps_taken": 0,
    "last_actions": [],
    "last_duration_s": 0,
}

# --- MCP Server ---
# In mcp SDK 1.6.x, host/port are constructor settings; run() only takes transport.
# Parse CLI args early so we can pass host/port to FastMCP constructor.
_parser = argparse.ArgumentParser(description="browser-use MCP Server")
_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
_parser.add_argument("--port", type=int, default=8934, help="Port to listen on")
_args = _parser.parse_args()

server = FastMCP("browser-use", host=_args.host, port=_args.port)


def _format_result(
    status: str,
    task: str,
    result: str | None = None,
    error: str | None = None,
    steps_taken: int = 0,
    max_steps: int = 0,
    actions: list[str] | None = None,
    duration_s: float = 0,
) -> str:
    """Format a structured, unambiguous result string for LLM callers."""
    lines = []
    lines.append(f"STATUS: {status.upper()}")
    lines.append(f"Task: {task}")
    lines.append(f"Steps: {steps_taken}/{max_steps}")
    if duration_s > 0:
        lines.append(f"Duration: {duration_s:.1f}s")
    if actions:
        lines.append(f"Actions: {', '.join(actions)}")
    if error:
        lines.append(f"Error: {error}")
    if result:
        lines.append("")
        lines.append(f"Result:\n{result}")
    return "\n".join(lines)


def _save_last_result(
    status: str,
    task: str,
    result: str | None,
    error: str | None,
    steps_taken: int,
    actions: list[str],
    duration_s: float,
):
    """Cache the last task result so get_status can report it."""
    _current_status.update({
        "last_status": status,
        "last_task": task,
        "last_result": result,
        "last_error": error,
        "last_finished_at": datetime.now().isoformat(),
        "last_steps_taken": steps_taken,
        "last_actions": actions,
        "last_duration_s": duration_s,
    })


@server.tool()
async def run_task(
    task: str,
    model: Optional[str] = None,
    max_steps: int = 10,
    use_vision: bool = True,
    max_actions_per_step: int = 10,
    ollama_base_url: Optional[str] = None,
    num_ctx: Optional[int] = None,
    enable_memory: bool = False,
) -> str:
    """Run a browser automation task using the browser-use agent.

    The agent autonomously navigates websites, fills forms, clicks buttons,
    and extracts information. Returns a structured result with clear STATUS.

    Args:
        task: Natural language description of the browser task to perform.
        model: Ollama model name. Defaults to server's DEFAULT_MODEL env var.
        max_steps: Maximum agent reasoning steps before stopping.
        use_vision: Use screenshots for page understanding.
        max_actions_per_step: Max browser actions per reasoning step.
        ollama_base_url: Override Ollama endpoint URL.
        num_ctx: Ollama context window size.
        enable_memory: Enable agent long-term memory across steps.
    """
    if _task_lock.locked():
        return (
            "STATUS: BUSY\n"
            "A task is already running. "
            "Use get_status to check progress or stop_task to cancel."
        )

    async with _task_lock:
        browser = None
        context = None
        start_time = time.monotonic()
        steps_taken = 0
        action_names = []
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
                enable_memory=enable_memory,
            )
            _current_status["agent"] = agent

            logger.info(
                "Running task: %s (model=%s, max_steps=%d, memory=%s)",
                task,
                model or DEFAULT_MODEL,
                max_steps,
                enable_memory,
            )

            history = await asyncio.wait_for(
                agent.run(max_steps=max_steps, on_step_end=on_step_end),
                timeout=TASK_TIMEOUT,
            )

            duration_s = time.monotonic() - start_time
            steps_taken = len(history.history) if history.history else 0
            try:
                action_names = history.action_names()
            except Exception:
                action_names = []

            final = history.final_result()
            is_done = history.is_done()
            is_successful = history.is_successful()

            if is_done and final:
                status = "completed" if is_successful else "failed"
                _save_last_result(status, task, final, None, steps_taken, action_names, duration_s)
                return _format_result(
                    status=status,
                    task=task,
                    result=final,
                    steps_taken=steps_taken,
                    max_steps=max_steps,
                    actions=action_names,
                    duration_s=duration_s,
                )

            # Agent finished without calling done -- hit max steps
            errors = history.errors()
            error_text = None
            if errors:
                error_msgs = [str(e) for e in errors[-3:] if e]
                if error_msgs:
                    error_text = "; ".join(error_msgs)

            # Try to extract any useful content from the last step
            last_content = None
            if history.history:
                for step in reversed(history.history):
                    for result in reversed(step.result):
                        if result.extracted_content:
                            last_content = result.extracted_content
                            break
                    if last_content:
                        break

            _save_last_result("max_steps", task, last_content, error_text, steps_taken, action_names, duration_s)
            return _format_result(
                status="max_steps_reached",
                task=task,
                result=last_content,
                error=error_text or f"Agent did not call done within {max_steps} steps",
                steps_taken=steps_taken,
                max_steps=max_steps,
                actions=action_names,
                duration_s=duration_s,
            )

        except asyncio.TimeoutError:
            duration_s = time.monotonic() - start_time
            _save_last_result("timeout", task, None, f"Timed out after {TASK_TIMEOUT}s", steps_taken, action_names, duration_s)
            return _format_result(
                status="timeout",
                task=task,
                error=f"Task timed out after {TASK_TIMEOUT} seconds",
                steps_taken=steps_taken,
                max_steps=max_steps,
                duration_s=duration_s,
            )
        except Exception as e:
            duration_s = time.monotonic() - start_time
            logger.exception("Task failed")
            error_msg = f"{type(e).__name__}: {e}"
            _save_last_result("failed", task, None, error_msg, steps_taken, action_names, duration_s)
            return _format_result(
                status="error",
                task=task,
                error=error_msg,
                steps_taken=steps_taken,
                max_steps=max_steps,
                duration_s=duration_s,
            )
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
    """Check the status of the currently running or last completed browser-use task.

    Returns whether a task is running (with step progress), or the result
    of the last completed task if no task is running.
    """
    if _current_status["running"]:
        return (
            f"STATUS: RUNNING\n"
            f"Task: {_current_status['task']}\n"
            f"Step: {_current_status['step']}/{_current_status['max_steps']}\n"
            f"Started: {_current_status['started_at']}"
        )

    # No task running -- report last completed task if available
    last_status = _current_status.get("last_status")
    if last_status:
        lines = [
            f"STATUS: IDLE",
            f"Last task: {_current_status['last_task']}",
            f"Last result status: {last_status.upper()}",
            f"Finished at: {_current_status['last_finished_at']}",
            f"Steps taken: {_current_status['last_steps_taken']}",
            f"Duration: {_current_status['last_duration_s']:.1f}s",
        ]
        if _current_status["last_error"]:
            lines.append(f"Error: {_current_status['last_error']}")
        if _current_status["last_result"]:
            lines.append(f"\nLast result:\n{_current_status['last_result']}")
        return "\n".join(lines)

    return "STATUS: IDLE\nNo task has been run yet."


@server.tool()
async def stop_task() -> str:
    """Stop the currently running browser-use task.

    Signals the agent to stop after its current step completes.
    """
    agent = _current_status.get("agent")
    if not agent or not _current_status["running"]:
        return "STATUS: IDLE\nNo task is currently running."

    agent.state.stopped = True
    return "STATUS: STOPPING\nStop signal sent. Task will finish after current step completes."


def main():
    logger.info("Starting browser-use MCP server on %s:%d", _args.host, _args.port)
    logger.info("Default model: %s", DEFAULT_MODEL)
    logger.info("Default Ollama endpoint: %s", DEFAULT_OLLAMA_ENDPOINT)
    logger.info("Task timeout: %ds", TASK_TIMEOUT)

    server.run(transport="sse")


if __name__ == "__main__":
    main()
