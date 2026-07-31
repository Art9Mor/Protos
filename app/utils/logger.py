import os
import sys
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

DEBUG = os.getenv('DEBUG', 'True').lower() == 'true'
LOG_LEVELS = os.getenv('LOG_LEVELS', '1,5')
LOG_OUTPUT = os.getenv('LOG_OUTPUT', 'both').lower()
DIALOG_OUTPUT = os.getenv('DIALOG_OUTPUT', 'console').lower()
SHOW_LOGS_IN_CONSOLE = os.getenv('SHOW_LOGS_IN_CONSOLE', 'True').lower() == 'true'
LOG_DIR = os.getenv('LOG_DIR', 'logs')

os.makedirs(LOG_DIR, exist_ok=True)

logger.remove()

LEVEL_MAP = {
    0: 'NONE',
    1: 'INFO',
    2: 'SUCCESS',
    3: 'WARNING',
    4: 'ERROR',
    5: 'DEBUG',
    6: 'ALL'
}

LEVEL_VALUES = {
    'NONE': 0,
    'INFO': 1,
    'SUCCESS': 2,
    'WARNING': 3,
    'ERROR': 4,
    'DEBUG': 5,
    'ALL': 6
}


def parse_log_levels(levels_str: str) -> set:
    """
    Парсинг строки с уровнями логирования.
    """

    if not levels_str or levels_str == '0':
        return set()

    levels = set()
    for part in levels_str.split(','):
        part = part.strip()
        if part == '6' or part.upper() == 'ALL':
            return {1, 2, 3, 4, 5}
        try:
            level_num = int(part)
            if 1 <= level_num <= 5:
                levels.add(level_num)
        except ValueError:
            for num, name in LEVEL_MAP.items():
                if name.lower() == part.lower():
                    if num == 6:
                        return {1, 2, 3, 4, 5}
                    if num != 0:
                        levels.add(num)
                    break

    return levels


ACTIVE_LEVELS = parse_log_levels(LOG_LEVELS)


def is_level_active(level_name: str) -> bool:
    """
    Проверка, активен ли уровень логирования.
    """

    if not ACTIVE_LEVELS:
        return False

    level_num = LEVEL_VALUES.get(level_name, 0)
    return level_num in ACTIVE_LEVELS


def log_filter(record):
    level_name = record['level'].name
    return is_level_active(level_name)


def add_logger_handler(
    filename: str,
    level: str = 'DEBUG',
    rotation: str = '10 MB',
    retention: str = '30 days',
    compression: str = 'zip',
    filter_key: str = None,
    console: bool = SHOW_LOGS_IN_CONSOLE
):
    """
    Добавление обработчика для логов.
    """

    if filter_key:
        file_filter = lambda record: record['extra'].get(filter_key, False) and log_filter(record)
    else:
        file_filter = log_filter

    if LOG_OUTPUT in ['file', 'both']:
        logger.add(
            os.path.join(LOG_DIR, filename),
            rotation=rotation,
            retention=retention,
            compression=compression,
            level=level,
            format='{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}',
            filter=file_filter,
            enqueue=True
        )

    if LOG_OUTPUT in ['console', 'both'] and console:
        logger.add(
            sys.stdout,
            format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>',
            level=level,
            colorize=True,
            backtrace=True,
            diagnose=True,
            filter=file_filter
        )


add_logger_handler(
    'protos_main.log',
    level='DEBUG' if DEBUG else 'INFO',
    filter_key='main',
    rotation='10 MB',
    retention='30 days'
)

add_logger_handler(
    'protos_assistant.log',
    level='DEBUG' if DEBUG else 'INFO',
    filter_key='assistant',
    rotation='10 MB',
    retention='30 days'
)

add_logger_handler(
    'protos_training.log',
    level='DEBUG' if DEBUG else 'INFO',
    filter_key='training',
    rotation='10 MB',
    retention='30 days'
)

add_logger_handler(
    'protos_neural.log',
    level='DEBUG' if DEBUG else 'INFO',
    filter_key='neural',
    rotation='10 MB',
    retention='14 days'
)

add_logger_handler(
    'protos_memory.log',
    level='DEBUG' if DEBUG else 'INFO',
    filter_key='memory',
    rotation='10 MB',
    retention='14 days'
)

add_logger_handler(
    'protos_fs.log',
    level='INFO',
    filter_key='fs',
    rotation='10 MB',
    retention='14 days'
)

add_logger_handler(
    'protos_errors.log',
    level='ERROR',
    filter_key='error',
    rotation='10 MB',
    retention='60 days'
)

if DIALOG_OUTPUT in ['console', 'both']:
    logger.add(
        sys.stdout,
        format='<level>{message}</level>',
        level='INFO',
        colorize=True,
        filter=lambda record: record['extra'].get('dialog', False)
    )

logger.add(
    sys.stdout,
    format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> - <level>{message}</level>',
    level='INFO',
    colorize=True,
    filter=lambda record: record['level'].name == 'INFO'
)

active_levels_names = [LEVEL_MAP.get(l, 'UNKNOWN') for l in sorted(ACTIVE_LEVELS)]
if not active_levels_names:
    active_levels_names = ['NONE']

logger.info(f'📋 Настройки логирования:')
logger.info(f'  Активные уровни: {", ".join(active_levels_names)}')
logger.info(f'  Вывод: {LOG_OUTPUT}')
logger.info(f'  Диалоги: {DIALOG_OUTPUT}')
logger.info(f'  Логи в консоли: {SHOW_LOGS_IN_CONSOLE}')
logger.info(f'  Папка логов: {LOG_DIR}')


main_logger = logger.bind(main=True)
assistant_logger = logger.bind(assistant=True)
training_logger = logger.bind(training=True)
neural_logger = logger.bind(neural=True)
memory_logger = logger.bind(memory=True)
fs_logger = logger.bind(fs=True)
error_logger = logger.bind(error=True)
dialog_logger = logger.bind(dialog=True)
debug_logger = logger.bind(debug=True)

__all__ = [
    'logger',
    'main_logger',
    'assistant_logger',
    'training_logger',
    'neural_logger',
    'memory_logger',
    'fs_logger',
    'error_logger',
    'dialog_logger',
    'debug_logger',
    'is_level_active',
    'parse_log_levels'
]
