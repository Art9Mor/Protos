from app.memory.context import ContextMemory


class ProtosCore:
    """
    Центральное ядро Протоса.
    Объединение всех подсистем.
    """

    def __init__(self):
        self.memory = ContextMemory
        self.text = None
        self.training = None
        self.network = None