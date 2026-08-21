from inference import DEFAULT_CATALOG, parse_args, render_policy_prompt


class PromptTokenizer:
    def apply_chat_template(
        self,
        messages,
        add_generation_prompt,
        tokenize,
    ):
        assert add_generation_prompt is True
        assert tokenize is False
        return messages[0]["content"]


def test_policy_prompt_matrix():
    tokenizer = PromptTokenizer()
    raw = render_policy_prompt(tokenizer, "QUERY", DEFAULT_CATALOG, "raw")
    mid = render_policy_prompt(tokenizer, "QUERY", DEFAULT_CATALOG, "mid")
    planreason = render_policy_prompt(
        tokenizer, "QUERY", DEFAULT_CATALOG, "planreason"
    )

    assert raw == "QUERY"
    assert mid == planreason
    assert mid.startswith("QUERY\n\n")
    assert "compact internal layout" in mid
    assert "search_agent, code_agent, summary_agent" in mid


def test_policy_cli_replaces_disable_flag_and_keeps_cache_independent():
    args = parse_args(
        [
            "--policy",
            "mid",
            "--cache-mode",
            "dual",
            "--query",
            "same query",
        ]
    )
    assert args.policy == "mid"
    assert args.cache_mode == "dual"
    assert args.query == "same query"


def test_policy_defaults_to_current_implementation():
    args = parse_args([])
    assert args.policy == "planreason"
    assert args.cache_mode == "dual"
    assert args.agent_log_path is None
