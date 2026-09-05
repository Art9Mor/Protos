from difflib import SequenceMatcher


class QAMatcher:
    """
    Поиск ответа среди известных пар вопрос-ответ.
    """

    def __init__(
        self,
        qa_pairs: list[dict],
        similarity_threshold: float = 0.85,
    ) -> None:

        self.qa_pairs = qa_pairs
        self.similarity_threshold = similarity_threshold

    def find_answer(self, question: str) -> str | None:

        question = question.lower().strip()

        best_answer = None
        best_similarity = 0.0

        for pair in self.qa_pairs:

            known_question = pair.get("question", "")
            answer = pair.get("answer", "")

            similarity = SequenceMatcher(
                None,
                question,
                known_question.lower(),
            ).ratio()

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = answer

        if best_similarity >= self.similarity_threshold:
            return best_answer

        return None