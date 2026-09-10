from agent_platform.context_saver import ContextSaver


def test_compacts_large_historical_tool_output_and_keeps_recent_turns():
    old = "\n".join(["ordinary output line"] * 1000 + ["ERROR: test failed"] + ["/workspace/project/file.py"])
    messages = [
        {"role": "system", "content": "system instructions"},
        {"role": "user", "content": "finish the task"},
        {"role": "assistant", "content": "inspect"},
        {"role": "tool", "content": old},
    ] + [{"role": "assistant", "content": f"recent {i}"} for i in range(8)]

    saver = ContextSaver(target_reduction=0.85, keep_recent=8, max_tool_chars=3000)
    compacted = saver.compact(messages)

    assert compacted[0] == messages[0]
    assert compacted[1] == messages[1]
    assert "ERROR: test failed" in compacted[3]["content"]
    assert compacted[-1] == messages[-1]
    assert saver.savings_ratio(messages, compacted) > 0.80


def test_repeated_tool_output_is_deduplicated():
    value = "same result " * 500
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {"role": "tool", "content": value},
        {"role": "tool", "content": value},
    ] + [{"role": "assistant", "content": "recent"} for _ in range(8)]

    compacted = ContextSaver(keep_recent=8).compact(messages)
    assert "repeated tool result omitted" in compacted[3]["content"]
