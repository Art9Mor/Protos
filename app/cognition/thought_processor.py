from app.cognition.thought import Thought, ThoughtIntent
from app.memory.context import ContextMemory


class ThoughtProcessor:
    """
    Когнитивный анализатор.
    На основе текста и текущей памяти формирует мысль Протоса.
    """

    def think(
        self,
        text: str,
        memory: ContextMemory,
    ) -> Thought:

        normalized = text.lower().strip()

        if any(word in normalized for word in ("привет", "здравствуй", "добрый")):
            return Thought(
                original_text=text,
                intent=ThoughtIntent.GREETING,
                confidence=1.0,
            )

        if "как тебя зовут" in normalized:
            return Thought(
                original_text=text,
                intent=ThoughtIntent.ASK_NAME,
                confidence=1.0,
            )

        if "кто ты" in normalized:
            return Thought(
                original_text=text,
                intent=ThoughtIntent.ASK_IDENTITY,
                confidence=1.0,
            )

        if "кто я" in normalized:
            return Thought(
                original_text=text,
                intent=ThoughtIntent.ASK_CREATOR,
                subject="user",
                confidence=1.0,
            )

        if "создател" in normalized:
            return Thought(
                original_text=text,
                intent=ThoughtIntent.ASK_CREATOR,
                confidence=1.0,
            )

        if "директив" in normalized:
            return Thought(
                original_text=text,
                intent=ThoughtIntent.ASK_DIRECTIVE,
                confidence=1.0,
            )

        if normalized.endswith("?"):
            return Thought(
                original_text=text,
                intent=ThoughtIntent.QUESTION,
                confidence=0.5,
            )

        return Thought(
            original_text=text,
            intent=ThoughtIntent.UNKNOWN,
            confidence=0.0,
        )