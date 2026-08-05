from app.cognition.thought import Thought, ThoughtIntent
from app.memory.context import ContextMemory
from app.memory.models import MemoryEvent, MemoryEventType


class MemoryRetriever:
    """
    Отвечает за поиск воспоминаний, связанных с текущей мыслью.
    """

    def retrieve(
        self,
        thought: Thought,
        memory: ContextMemory,
    ) -> list[MemoryEvent]:

        if thought.intent == ThoughtIntent.UNKNOWN:
            return []

        results: list[MemoryEvent] = []

        for event in memory.events:

            if event.event_type != MemoryEventType.SYSTEM:
                continue

            text = event.content.lower()

            match thought.intent:

                case ThoughtIntent.ASK_NAME:
                    if "меня зовут" in text:
                        results.append(event)

                case ThoughtIntent.ASK_IDENTITY:
                    if "я протос" in text:
                        results.append(event)

                case ThoughtIntent.ASK_CREATOR:
                    if "создател" in text:
                        results.append(event)

                case ThoughtIntent.ASK_DIRECTIVE:
                    if "директив" in text:
                        results.append(event)

        return results