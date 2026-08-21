"""Prompt policy helpers shared by the OpenAI planner server."""

from __future__ import annotations

from typing import Dict, List, Sequence


def apply_planner_prompt_policy(
    messages: Sequence[Dict[str, str]],
    policy: str,
    agent_names: Sequence[str],
) -> List[Dict[str, str]]:
    """Return messages for the requested planner output-order policy.

    ``reasonplan`` intentionally returns an equivalent copy without injecting
    any output-format instruction. ``planreason`` is the former ``now`` policy
    and retains its plan-first, Agent-field-first prompt. ``mid`` keeps the same
    plan-first prompt for its non-priority-decoding baseline.
    """

    result = [dict(message) for message in messages]
    if policy not in {"mid", "planreason"}:
        return result

    registry = ", ".join(agent_names)
    planner_instruction = {
        "role": "system",
        "content": (
            "Return a normal planner response with no private routing tags. "
            "Write PLAN_JSON before PLANNING_REASONING. In PLAN_JSON, use a "
            "compact JSON array without Markdown fences or indentation. Every "
            "object must put the agent field first and use this exact prefix: "
            '{"agent":"<name>",. Follow it with id, task, reason, and dep. '
            f"Agent names must be selected from: {registry}. Preserve the "
            "query's dates and numeric constraints verbatim. Do not create "
            "duplicate subtasks. Every object must explicitly include reason "
            "and dep; use dep:[] for independent tasks. Follow the caller's "
            "Agent capability and routing descriptions exactly. When a "
            "required external fact is absent from the supplied query or "
            "context, first create a fact-acquisition subtask using the "
            "caller's search, retrieval, or context Agent as appropriate. "
            "Never ask a math, calculation, code, or reasoning Agent to "
            "discover an absent external fact; computational Agents may use "
            "only values present in the request or produced by dependencies. "
            "In particular, a country's population for a named year is an "
            "external fact when its numeric value is not supplied: create a "
            "search subtask for each required country before any arithmetic. "
            "Use exactly one standalone marker of each kind, in this order: "
            "PLAN_JSON, END_PLAN_JSON, PLANNING_REASONING, "
            "END_PLANNING_REASONING. END_PLANNING_REASONING must be the final "
            "output line."
        ),
    }
    first_user = next(
        (
            index
            for index, message in enumerate(result)
            if message["role"] == "user"
        ),
        len(result),
    )
    result.insert(first_user, planner_instruction)
    return result
