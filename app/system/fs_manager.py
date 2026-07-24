import os
import platform
from pathlib import Path
from typing import Optional


class FileSystemManager:
    """Управление файловой системой (кроссплатформенный)."""

    def __init__(self):
        self.current_path = Path(os.getcwd())
        self.os_type = platform.system()

        self.allowed_extensions = ['.py', '.txt', '.json', '.md', '.csv', '.yaml']
        self.forbidden_paths = ['/etc', '/boot', '/sys', 'C:\\Windows', 'C:\\Program Files']

    def list_dir(self, path: Optional[str] = None) -> str:
        """Список файлов и папок в директории."""
        target = Path(path) if path else self.current_path

        try:
            items = sorted(target.iterdir())
            result = []

            for item in items:
                if item.is_dir():
                    result.append(f"📁 {item.name}/")
                else:
                    size = item.stat().st_size
                    result.append(f"📄 {item.name} ({self._format_size(size)})")

            return '\n'.join(result)
        except PermissionError:
            return "Ошибка: нет доступа к этой папке"
        except Exception as e:
            return f"Ошибка: {e}"

    def change_dir(self, path: Optional[str] = None) -> str:
        """Смена текущей директории."""
        if not path:
            self.current_path = Path.home()
            return f"Перешел в {self.current_path}"

        new_path = self.current_path / path
        new_path = new_path.resolve()

        if not self._is_allowed(new_path):
            return "Доступ запрещен"

        if new_path.exists() and new_path.is_dir():
            self.current_path = new_path
            return f"Перешел в {self.current_path}"
        else:
            return f"Папка не найдена: {path}"

    def read_file(self, filename: str) -> str:
        """Чтение содержимого файла."""
        filepath = self.current_path / filename

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            return f"Файл не найден: {filename}"
        except UnicodeDecodeError:
            return "Ошибка: файл в бинарном формате"
        except Exception as e:
            return f"Ошибка чтения файла: {e}"

    def write_file(self, filename: str, content: str) -> str:
        """Создание или перезапись файла."""
        filepath = self.current_path / filename

        if not self._is_allowed_extension(filename):
            return f"Ошибка: расширение {filepath.suffix} не разрешено"

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"✅ Файл {filename} создан/обновлен"
        except Exception as e:
            return f"Ошибка записи файла: {e}"

    def get_info(self) -> dict[str, str]:
        """Информация о системе."""
        return {
            'current_dir': str(self.current_path),
            'os_type': self.os_type,
            'os_version': platform.version(),
            'user': os.getenv('USER', os.getenv('USERNAME', 'unknown')),
            'home': str(Path.home())
        }

    @staticmethod
    def _format_size(size: int) -> str:
        """Форматирование размера файла."""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f}{unit}"
            size /= 1024
        return f"{size:.1f}TB"

    @staticmethod
    def _is_allowed(path: Path) -> bool:
        """Проверка безопасности пути."""
        path_str = str(path)
        forbidden_paths = ['/etc', '/boot', '/sys', 'C:\\Windows', 'C:\\Program Files']

        for forbidden in forbidden_paths:
            if path_str.startswith(forbidden):
                return False
        return True

    @staticmethod
    def _is_allowed_extension(filename: str) -> bool:
        """Проверка разрешенного расширения."""
        ext = Path(filename).suffix
        allowed = ['.py', '.txt', '.json', '.md', '.csv', '.yaml']
        return ext in allowed