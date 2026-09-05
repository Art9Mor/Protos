from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class QAPair:
    question: str
    answer: str
    source: str