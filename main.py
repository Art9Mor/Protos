import getpass
import json
import os
import sys
from app.interface.assistant import Assistant
from app.utils.logger import main_logger, error_logger, debug_logger


def load_config() -> dict:
    """
    Загрузка конфигурации.
    """
    config_path = 'config/config.json'
    debug_logger.debug(f'Загрузка конфигурации из {config_path}')

    default_config = {
        'memory_size': 50,
        'embedding_dim': 10,
        'vocab_size': 1000,
        'input_size': 10,
        'hidden_size': 64,
        'output_size': 10,
    }

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                debug_logger.debug('Конфигурация загружена из файла')
                return {**default_config, **config}
        except (json.JSONDecodeError, IOError) as e:
            error_logger.error(f'Ошибка загрузки конфигурации: {e}')
            debug_logger.debug('Использую конфигурацию по умолчанию')
            return default_config

    debug_logger.debug('Файл конфигурации не найден, создаю новый')
    os.makedirs('config', exist_ok=True)
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        debug_logger.debug('Создан файл конфигурации по умолчанию')
    except IOError as e:
        error_logger.error(f'Ошибка сохранения конфигурации: {e}')

    return default_config


def read_user_line(prompt: str = '\n👤 Вы: ', secret: bool = False) -> tuple[str, bool]:
    """
    Чтение строки. secret=True — пароль без отображения (getpass).
    Второе значение: True, если строку пришлось чинить после битых байт.
    """
    if secret:
        try:
            text = getpass.getpass(prompt='👤 Пароль (не отображается): ')
            return (text or '').strip(), False
        except (EOFError, KeyboardInterrupt):
            raise
        except Exception as e:
            debug_logger.debug(f'getpass failed: {e}, fallback input')
            # в некоторых IDE getpass ломается — тогда обычный ввод
            text = input('👤 Пароль: ')
            return (text or '').strip(), False

    try:
        return input(prompt).strip(), False
    except UnicodeDecodeError as e:
        debug_logger.debug(f'UnicodeDecodeError на input: {e}')
    except (EOFError, KeyboardInterrupt):
        raise

    sys.stdout.write(prompt)
    sys.stdout.flush()
    raw = b''
    try:
        raw = sys.stdin.buffer.readline()
        if not raw:
            return '', False
        text = raw.decode('utf-8', errors='strict').rstrip('\r\n').strip()
        return text, False
    except UnicodeDecodeError as e:
        debug_logger.debug(f'Битые байты при вводе: {e}; raw={raw!r}')
        text = raw.decode('utf-8', errors='replace').rstrip('\r\n').strip()
        return text, True


def show_bot_response(response: dict) -> bool:
    """
    Печать ответа. Возвращает True, если нужно завершить программу.
    """

    text = (response.get('response') or '').strip()
    shutdown = bool(response.get('shutdown'))
    if text.startswith('__SHUTDOWN__'):
        text = text[len('__SHUTDOWN__'):].lstrip()
        shutdown = True
    print(f'🤖 {text}')
    if response.get('is_learning', False) and not shutdown:
        print(
            f'   🧠 (Учусь... память: {response.get("memory_size", 0)} сообщений)'
        )
    return shutdown


def main():
    """
    Главная функция.
    """

    main_logger.info('🚀 Запуск Протоса')
    debug_logger.debug('Начало выполнения main()')

    try:
        config = load_config()
        debug_logger.debug(f'Конфигурация: {config}')

        print('=' * 50)
        print('🤖 Искин Протос')
        print('=' * 50)
        print("Введите 'help' для списка команд")
        print('=' * 50)

        debug_logger.debug('Создание ассистента')
        assistant = Assistant(config)
        debug_logger.debug('Ассистент создан')

        main_logger.info('Протос готов к работе')

        if assistant.is_logged_in():
            hello = assistant.router.greeting('ru')
            print(f'🤖 {hello}')
        else:
            print('🤖 Привет! Я Протос.')
            print(f'🤖 {assistant.auth.start_login_flow("ru")}')

        while True:
            try:
                pending = getattr(assistant, 'pending_login', None) or {}
                step = pending.get('step', '')
                need_secret = step in (
                    'ask_password',
                    'register_password',
                    'register_password2',
                    'register_owner_password',
                    'register_owner_password2',
                )

                user_input, corrupted = read_user_line(secret=need_secret)

                if not user_input:
                    print('🤖 Напиши что-нибудь.')
                    continue

                if corrupted:
                    print('⚠️ Часть сообщения повреждена при передаче, попробуй отправить его снова.')
                    continue

                if user_input.lower() in ['exit', 'quit', 'выход', 'стоп']:
                    debug_logger.debug('Получена команда выхода')
                    main_logger.info('Завершение работы Протоса')
                    break

                elif user_input.lower() == 'help':
                    debug_logger.debug('Вывод справки')
                    print(assistant.commands.help_text())

                elif user_input.lower() == 'learn':
                    debug_logger.debug('Переключение режима обучения')
                    response = assistant.toggle_learning()
                    print(f'🤖 {response}')

                elif user_input.lower() == 'clear':
                    response = assistant.process_input('/clear_memory')
                    if show_bot_response(response):
                        break

                elif user_input.lower() in ['info', 'stats']:
                    debug_logger.debug(f'Команда: {user_input}')
                    response = assistant.process_input(user_input)
                    if show_bot_response(response):
                        break

                elif user_input.startswith('ls') or user_input.startswith('cd'):
                    debug_logger.debug(f'Системная команда: {user_input}')
                    response = assistant.process_input(user_input)
                    if show_bot_response(response):
                        break

                else:
                    debug_logger.debug(f'Обработка ввода: {user_input[:50]}...')
                    response = assistant.process_input(user_input)
                    if show_bot_response(response):
                        debug_logger.debug('Выход по прощанию')
                        main_logger.info('Завершение работы по прощанию')
                        break

            except KeyboardInterrupt:
                debug_logger.debug('Прерывание по Ctrl+C')
                print('\n👋 До свидания!')
                main_logger.info('Завершение работы по Ctrl+C')
                break
            except Exception as e:
                import traceback
                traceback.print_exc()
                error_logger.exception(f'Ошибка в главном цикле: {e}')
                print(f'❌ Ошибка: {e}')
                debug_logger.debug('Продолжение работы после ошибки')

    except Exception as e:
        error_logger.critical(f'Критическая ошибка при запуске: {e}')
        print(f'❌ Критическая ошибка: {e}')
        sys.exit(1)

    main_logger.info('Протос завершил работу')
    debug_logger.debug('Завершение main()')


if __name__ == '__main__':
    main()
