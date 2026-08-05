from dataclasses import dataclass
from enum import Enum, auto


class ThoughtIntent(Enum):
    """
    Намерение, распознанное в сообщении пользователя.
    """

    UNKNOWN = auto()

    GREETING = auto()

    ASK_NAME = auto()
    ASK_IDENTITY = auto()
    ASK_CREATOR = auto()
    ASK_DIRECTIVE = auto()

    QUESTION = auto()
    COMMAND = auto()


@dataclass(slots=True, frozen=True)
class Thought:
    """
    Внутреннее представление мысли Протоса.
    """

    original_text: str

    intent: ThoughtIntent = ThoughtIntent.UNKNOWN

    subject: str | None = None

    confidence: float = 0.0