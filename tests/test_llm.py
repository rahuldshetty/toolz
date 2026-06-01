from toolz.llm import (
    custom_lm,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    OPENAI_MODEL_NAME
)

def test_lm():
    assert OPENAI_API_KEY not in [None, ""]
    assert OPENAI_MODEL_NAME not in [None, ""]
    assert OPENAI_BASE_URL not in [None, ""]

    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant"
        },
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]

    reply = custom_lm(messages = messages)

    assert len(reply) > 0
    assert "text" in reply[0]
    assert "paris" in reply[0]['text'].lower()
