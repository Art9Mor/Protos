class ResponseGenerator:
    """
    Генератор ответов Протоса.

    Отвечает только за формирование текста ответа.
    """

    def __init__(self, assistant):
        self.assistant = assistant

    def generate(
        self,
        user_input: str,
        speaker: str,
        context: list[str],
    ) -> str:
        """
        Сгенерировать ответ пользователю.
        """

        return self.assistant._route_request(
            user_input=user_input,
            speaker=speaker,
            context=context,
        )