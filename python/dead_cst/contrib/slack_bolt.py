"""Plugin: keep Slack Bolt apps and their handlers alive."""

from __future__ import annotations

from ..plugins.decl_shapes import DispatchAppPlugin

# Attribute names ``slack_bolt.App`` / ``slack_bolt.async_app.AsyncApp``
# expose for registering a handler. Matched as the rightmost attribute
# of ``@<instance>.<name>(...)`` -- both bare-decorator and
# call-decorator forms are picked up by ``find_handlers``.
#
# Sourced from the public Slack Bolt for Python API
# (https://github.com/slackapi/bolt-python). The sync ``App`` and async
# ``AsyncApp`` classes expose the same decorator surface; both bases are
# listed in ``app_classes`` below.
_REGISTRATION_DECORATORS: frozenset[str] = frozenset(
    {
        "event",
        "message",
        "command",
        "action",
        "shortcut",
        "view",
        "options",
        "error",
        "step",
        "function",
    }
)


def slack_bolt_plugin() -> DispatchAppPlugin:
    """Mark Slack Bolt apps as entrypoints and wire handlers through them.

    Handles direct (``app = App(...)``), aliased
    (``from slack_bolt import App as Bolt; app = Bolt(...)``),
    module-prefixed (``import slack_bolt; app = slack_bolt.App(...)``),
    and factory-style (``app = create_app()``) construction.

    Both sync and async APIs are recognised:

    * ``slack_bolt.App`` -- sync app.
    * ``slack_bolt.async_app.AsyncApp`` -- async app (decorator forms
      identical to the sync surface).
    """
    return DispatchAppPlugin(
        marker_prefix="slack-bolt",
        app_classes=("slack_bolt.App", "slack_bolt.async_app.AsyncApp"),
        registration_decorators=_REGISTRATION_DECORATORS,
    )
