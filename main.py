import json
import os
from app.interface.assistant import Assistant


def load_config() -> dict:
    """
    Загрузка конфигурации.
    """
    config_path = 'config/config.json'

    default_config = {
        'memory_size': 50,
        'embedding_dim': 10,
        'vocab_size': 1000,
        'input_size': 10,
        'hidden_size': 64,
        'output_size': 10
    }

    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            # Объединяем с дефолтными настройками
            return {**default_config, **config}

    # Если файла нет, создаем его
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(default_config, f, ensure_ascii=False, indent=2)

    return default_config


def main():
    """
    Главная функция.
    """
    config = load_config()

    print("=" * 50)
    print("🤖 Искин Протос")
    print("=" * 50)
    print("Введите 'help' для списка команд")
    print("=" * 50)

    # Создаем ассистента
    assistant = Assistant(config)

    # Основной цикл
    while True:
        try:
            user_input = input("\n👤 Вы: ").strip()

            if not user_input:
                continue

            # Обработка команд
            if user_input.lower() in ['exit', 'quit', 'выход', 'стоп']:
                print("👋 До свидания!")
                break

            elif user_input.lower() == 'help':
                print("\n📚 Доступные команды:")
                print("  help        - показать это сообщение")
                print("  info        - информация о системе")
                print("  stats       - статистика Протоса")
                print("  ls          - список файлов в текущей папке")
                print("  cd <путь>   - перейти в папку")
                print("  learn       - включить/выключить режим обучения")
                print("  clear       - очистить память")
                print("  exit        - выйти")

            elif user_input.lower() == 'learn':
                response = assistant.toggle_learning()
                print(f"🤖 {response}")

            elif user_input.lower() == 'clear':
                assistant.memory.clear()
                print("🧹 Память очищена")

            elif user_input.lower() in ['info', 'stats']:
                response = assistant.process_input(user_input)
                print(f"🤖 {response['response']}")

            elif user_input.startswith('ls') or user_input.startswith('cd'):
                response = assistant.process_input(user_input)
                print(f"🤖 {response['response']}")

            else:
                # Обычный диалог
                response = assistant.process_input(user_input)
                print(f"🤖 {response['response']}")

                # Показываем статус в режиме обучения
                if response['is_learning']:
                    print(f"   🧠 (Учусь... память: {response['memory_size']} сообщений)")

        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"❌ Ошибка: {e}")


if __name__ == "__main__":
    main()