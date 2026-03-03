"""
Custom browser factory for creating BrowserSession instances with project defaults.

Provides a thin wrapper around browser-use v0.12.0's BrowserSession + BrowserProfile
for headless automation on BB8. Extensible for future customizations.
"""

import logging

from browser_use import BrowserSession, BrowserProfile

logger = logging.getLogger(__name__)


# Default profile for headless MCP/agent use
DEFAULT_HEADLESS_PROFILE = BrowserProfile(
    headless=True,
    window_size={"width": 1280, "height": 1100},
    disable_security=False,
)


def create_browser_session(
    headless: bool = True,
    window_width: int = 1280,
    window_height: int = 1100,
    profile: BrowserProfile | None = None,
    **kwargs,
) -> BrowserSession:
    """Create a BrowserSession with sensible defaults for this project.

    Args:
        headless: Run in headless mode (default True for MCP/agent use).
        window_width: Browser viewport width.
        window_height: Browser viewport height.
        profile: Optional pre-built BrowserProfile. If provided, other args are ignored.
        **kwargs: Additional kwargs passed to BrowserSession constructor.

    Returns:
        A configured BrowserSession (not yet started -- call session.start() or let Agent handle it).
    """
    if profile is None:
        profile = BrowserProfile(
            headless=headless,
            window_size={"width": window_width, "height": window_height},
        )

    session = BrowserSession(browser_profile=profile, **kwargs)
    logger.debug(
        "Created BrowserSession (headless=%s, %dx%d)",
        profile.headless,
        (profile.window_size or {}).get("width", 0),
        (profile.window_size or {}).get("height", 0),
    )
    return session
