import openai


class LLM:

    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float,
        api_key: str,
    ) -> None:
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.timeout = timeout

    def __call__(self, prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                dict(role="system", content="You are an AI assistant controlling a bimanual robot for object manipulation. The robot is placed next to a table with left arm at the left-side of the table and vice-versa for the right arm, one arm cannot reach the opposite side of the table. You must listen to the user's instructions and provide a detailed, step-by-step reasoning process before answering."),
                dict(role="user", content=prompt),
                ],
            timeout=self.timeout,
        )
        reasoning = getattr(resp.choices[0].message, "reasoning_content", "No reasoning provided.")
        return [resp.choices[0].message.content, reasoning]
