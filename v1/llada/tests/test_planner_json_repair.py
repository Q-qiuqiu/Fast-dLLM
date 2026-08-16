import json

from planner_json_repair import repair_plan_json_response


AGENTS = ["search_agent", "math_agent", "code_agent", "commonsense_agent"]


def envelope(plan_text):
    return (
        "PLAN_JSON\n"
        + plan_text
        + "\nEND_PLAN_JSON\n\nPLANNING_REASONING\nreasoning\n"
        "END_PLANNING_REASONING"
    )


def extract_returned_plan(content):
    segment = content.split("PLAN_JSON", 1)[1].split("END_PLAN_JSON", 1)[0]
    return json.loads(segment)


def test_valid_plan_is_returned_byte_for_byte():
    content = envelope(
        '[{"agent":"math_agent","id":1,"task":"calculate",'
        '"reason":"numbers supplied","dep":[]}]'
    )
    repaired, report = repair_plan_json_response(content, AGENTS)
    assert repaired == content
    assert report["applied"] is False
    assert report["method"] == "strict"


def test_repairs_missing_object_close_and_comma_observed_in_case_93():
    content = envelope(
        """[
  {
    "agent": "search_agent",
    "id": 1,
    "task": "Find height one",
    "reason": "Value is absent",
    "dep": []
  },
  {
    "agent": "search_agent",
    "id": 2,
    "task": "Find height two",
    "reason": "Value is absent",
    "dep": [1]
  {
    "agent": "math_agent",
    "id": 3,
    "task": "Calculate average",
    "reason": "Use both heights",
    "dep": [1, 2]
  }
]"""
    )
    repaired, report = repair_plan_json_response(content, AGENTS)
    plan = extract_returned_plan(repaired)
    assert [item["agent"] for item in plan] == [
        "search_agent", "search_agent", "math_agent"
    ]
    assert report["applied"] is True
    assert report["method"] == "minimal_syntax"
    assert report["operations"] == ["close_object_and_add_comma:1"]


def test_repairs_missing_open_brace_and_id_quote():
    content = envelope(
        """[
  {"agent":"search_agent","id":1,"task":"find","reason":"absent","dep":[]},
    "agent":"code_agent",id":2,"task":"compute","reason":"precise","dep":[1]
  }
]"""
    )
    repaired, report = repair_plan_json_response(content, AGENTS)
    plan = extract_returned_plan(repaired)
    assert [item["id"] for item in plan] == [1, 2]
    assert report["method"] == "minimal_syntax"
    assert report["operations"] == [
        "open_next_object:1", "restore_id_open_quote:1"
    ]


def test_schema_rebuild_handles_missing_field_comma_without_guessing_values():
    content = envelope(
        """[
  {
    "agent":"math_agent",
    "id":1,
    "task":"calculate"
    "reason":"all inputs supplied",
    "dep":[]
  }
]"""
    )
    repaired, report = repair_plan_json_response(content, AGENTS)
    assert extract_returned_plan(repaired)[0]["reason"] == "all inputs supplied"
    assert report["method"] == "schema_rebuild"
    assert "rebuild_from_validated_fields" in report["operations"]


def test_does_not_repair_semantically_unsafe_future_dependency():
    content = envelope(
        '[{"agent":"math_agent","id":1,"task":"calculate",'
        '"reason":"numbers supplied","dep":[2]}]'
    )
    repaired, report = repair_plan_json_response(content, AGENTS)
    assert repaired == content
    assert report["applied"] is False
    assert report["method"] == "failed"
    assert "future dependency" in report["repair_error"]

