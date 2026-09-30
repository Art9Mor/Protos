import re
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


    @staticmethod
    def _normalize(text: str) -> str:
        t = (text or '').lower().strip()
        t = re.sub(r'[?.!,;:…]+', ' ', t)
        for w in (' твой ', ' твоя ', ' твоё ', ' твое ', ' ваш ', ' ваша '):
            t = t.replace(w, ' ')
        t = re.sub(r'\s+', ' ', t).strip()
        return t


    def find_answer(self, question: str) -> str | None:
        q = self._normalize(question)
        if not q:
            return None

        best_answer = None
        best_similarity = 0.0

        for pair in self.qa_pairs:
            known = self._normalize(pair.get('question', ''))
            answer = pair.get('answer', '')
            if not known or not answer:
                continue

            similarity = SequenceMatcher(None, q, known).ratio()

            wq = set(q.split())
            wk = set(known.split())
            if wq and wk:
                overlap = len(wq & wk) / max(len(wq | wk), 1)
                similarity = max(similarity, overlap)

            if similarity > best_similarity:
                best_similarity = similarity
                best_answer = answer

        if best_similarity >= self.similarity_threshold:
            return best_answer
        return None