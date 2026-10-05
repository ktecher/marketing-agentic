from datetime import datetime, timezone

from pydantic import BaseModel

from tools.registry import register_tool


class NoArgs(BaseModel):
    pass


@register_tool(
    "current_date",
    "Get today's date (UTC). Use it to judge whether a source or mention is recent.",
    NoArgs,
)
def current_date(*, options: dict) -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d (%A)")
