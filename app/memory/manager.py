from pathlib import Path
import json


class MemoryManager:
    """
    Управляет долговременной памятью Протоса.
    """

    def __init__(self, data_dir: str = "data"):

        self.data_dir = Path(data_dir)

    def load_json(self, filename: str):

        path = self.data_dir / filename

        if not path.exists():
            return None

        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def save_json(self, filename: str, data) -> None:

        path = self.data_dir / filename

        with path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=4,
            )