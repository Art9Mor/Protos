# scripts/train_on_books.py

import os
import json
import re
import zipfile
from pathlib import Path
from typing import Optional
import requests



class BookDownloader:
    """
    Загрузчик книг для обучения с поддержкой разных форматов.
    """

    def __init__(self):
        self.books_dir = Path("data/books")
        self.books_dir.mkdir(parents=True, exist_ok=True)

    def load_local_books(self) -> list[str]:
        """
        Загрузка локальных книг из папки data/books/.
        """
        books = []
        for filepath in self.books_dir.iterdir():
            if filepath.is_file():
                ext = filepath.suffix.lower()
                supported = [
                    '.txt', '.epub', '.fb2', '.doc', '.docx',
                    '.pdf', '.zip', '.rar', '.json', '.md', '.html', '.htm'
                ]
                if ext in supported:
                    books.append(str(filepath))
        return books

    @staticmethod
    def _read_txt(filepath: Path) -> str:
        """Чтение .txt файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='cp1251') as f:
                    return f.read()
            except (UnicodeDecodeError, IOError):
                return ""

    @staticmethod
    def _read_epub(filepath: Path) -> str:
        """Чтение .epub файла."""
        try:
            import ebooklib
            from ebooklib import epub
            from bs4 import BeautifulSoup

            book = epub.read_epub(filepath)
            text = []

            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text.append(soup.get_text())

            return '\n'.join(text)
        except ImportError:
            print("⚠️ Для чтения .epub нужны модули: pip install EbookLib beautifulsoup4")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения .epub: {e}")
            return ""

    @staticmethod
    def _read_fb2(filepath: Path) -> str:
        """Чтение .fb2 файла."""
        try:
            from xml.etree import ElementTree

            tree = ElementTree.parse(filepath)
            root = tree.getroot()

            text = []
            for p in root.iter('p'):
                if p.text:
                    text.append(p.text)

            return '\n'.join(text)
        except Exception as e:
            print(f"⚠️ Ошибка чтения .fb2: {e}")
            return ""

    @staticmethod
    def _read_doc(filepath: Path) -> str:
        """Чтение .doc файла (устаревший формат)."""
        try:
            import textract
            return textract.process(filepath).decode('utf-8')
        except ImportError:
            print("⚠️ Для чтения .doc нужен модуль: pip install textract")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения .doc: {e}")
            return ""

    @staticmethod
    def _read_docx(filepath: Path) -> str:
        """Чтение .docx файла."""
        try:
            import docx
            doc = docx.Document(filepath)
            text = []
            for para in doc.paragraphs:
                text.append(para.text)
            return '\n'.join(text)
        except ImportError:
            print("⚠️ Для чтения .docx нужен модуль: pip install python-docx")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения .docx: {e}")
            return ""

    @staticmethod
    def _read_pdf(filepath: Path) -> str:
        """Чтение .pdf файла."""
        try:
            import PyPDF2
            text = []
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text.append(page_text)
            return '\n'.join(text)
        except ImportError:
            print("⚠️ Для чтения .pdf нужен модуль: pip install PyPDF2")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения .pdf: {e}")
            return ""

    @staticmethod
    def _read_zip(filepath: Path) -> str:
        """Чтение .zip архива с книгами."""
        try:
            text = []
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if file_info.filename.endswith('.txt'):
                        try:
                            with zip_ref.open(file_info) as f:
                                text.append(f.read().decode('utf-8', errors='ignore'))
                        except (UnicodeDecodeError, IOError):
                            pass
            return '\n'.join(text)
        except (zipfile.BadZipFile, IOError) as e:
            print(f"⚠️ Ошибка чтения .zip: {e}")
            return ""

    @staticmethod
    def _read_rar(filepath: Path) -> str:
        """Чтение .rar архива с книгами."""
        try:
            import rarfile
            text = []
            with rarfile.RarFile(filepath) as rf:
                for file_info in rf.infolist():
                    if file_info.filename.endswith('.txt'):
                        try:
                            with rf.open(file_info) as f:
                                text.append(f.read().decode('utf-8', errors='ignore'))
                        except (UnicodeDecodeError, IOError):
                            pass
            return '\n'.join(text)
        except ImportError:
            print("⚠️ Для чтения .rar нужен модуль: pip install rarfile")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения .rar: {e}")
            return ""

    @staticmethod
    def _read_json(filepath: Path) -> str:
        """Чтение .json файла с текстом."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, dict):
                if 'text' in data:
                    return data['text']
                if 'content' in data:
                    return data['content']
                return json.dumps(data, ensure_ascii=False)
            if isinstance(data, list):
                return '\n'.join([str(item) for item in data])
            return str(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠️ Ошибка чтения .json: {e}")
            return ""

    @staticmethod
    def _read_md(filepath: Path) -> str:
        """Чтение .md (Markdown) файла."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()

            # Убираем маркдаун разметку
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            # Ссылки [текст](url) -> текст (исправлено)
            text = re.sub(r'\[([^\[\]]+)]\([^)]+\)', r'\1', text)
            # Картинки ![alt](url) -> alt (исправлено)
            text = re.sub(r'!\[([^\[\]]*)]\([^)]+\)', r'\1', text)
            text = re.sub(r'```[^`]*```', '', text, flags=re.DOTALL)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^>\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)

            # Убираем лишние пустые строки
            text = re.sub(r'\n\s*\n', '\n\n', text)

            return text.strip()
        except IOError as e:
            print(f"⚠️ Ошибка чтения .md: {e}")
            return ""

    @staticmethod
    def _read_html(filepath: Path) -> str:
        """Чтение .html/.htm файла."""
        try:
            from bs4 import BeautifulSoup
            with open(filepath, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
        except ImportError:
            print("⚠️ Для чтения .html нужен модуль: pip install beautifulsoup4")
            return ""
        except Exception as e:
            print(f"⚠️ Ошибка чтения .html: {e}")
            return ""

    def read_book(self, filepath: str) -> Optional[str]:
        """
        Чтение книги в зависимости от расширения.
        """
        ext = Path(filepath).suffix.lower()

        readers = {
            '.txt': self._read_txt,
            '.epub': self._read_epub,
            '.fb2': self._read_fb2,
            '.doc': self._read_doc,
            '.docx': self._read_docx,
            '.pdf': self._read_pdf,
            '.zip': self._read_zip,
            '.rar': self._read_rar,
            '.json': self._read_json,
            '.md': self._read_md,
            '.html': self._read_html,
            '.htm': self._read_html,
        }

        if ext in readers:
            return readers[ext](Path(filepath))
        print(f"⚠️ Неподдерживаемый формат: {ext}")
        return None

    def download_russian_books(self) -> list[str]:
        """
        Скачивание русских книг из открытых источников.
        """
        if requests is None:
            print("⚠️ Для скачивания нужен модуль requests: pip install requests")
            return []

        books = []
        russian_books = [
            {
                'name': 'Война и мир - Толстой',
                'url': 'https://www.gutenberg.org/cache/epub/2600/pg2600.txt',
                'filename': 'war_and_peace_tolstoy.txt'
            },
            {
                'name': 'Преступление и наказание - Достоевский',
                'url': 'https://www.gutenberg.org/cache/epub/2554/pg2554.txt',
                'filename': 'crime_and_punishment_dostoevsky.txt'
            }
        ]

        for book in russian_books:
            try:
                print(f"📥 Скачиваю: {book['name']}")
                response = requests.get(book['url'], timeout=60)
                if response.status_code == 200:
                    filepath = self.books_dir / book['filename']
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    books.append(str(filepath))
                    print(f"✅ Скачано: {book['filename']} ({len(response.text)} символов)")
            except requests.RequestException as e:
                print(f"❌ Ошибка скачивания {book['name']}: {e}")

        return books

    def download_english_books(self) -> list[str]:
        """
        Скачивание английских книг из открытых источников.
        """
        if requests is None:
            print("⚠️ Для скачивания нужен модуль requests: pip install requests")
            return []

        books = []
        english_books = [
            {
                'name': 'Pride and Prejudice - Jane Austen',
                'url': 'https://www.gutenberg.org/cache/epub/1342/pg1342.txt',
                'filename': 'pride_and_prejudice_austen.txt'
            },
            {
                'name': '1984 - George Orwell',
                'url': 'https://www.gutenberg.org/cache/epub/1400/pg1400.txt',
                'filename': '1984_orwell.txt'
            }
        ]

        for book in english_books:
            try:
                print(f"📥 Downloading: {book['name']}")
                response = requests.get(book['url'], timeout=60)
                if response.status_code == 200:
                    filepath = self.books_dir / book['filename']
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(response.text)
                    books.append(str(filepath))
                    print(f"✅ Downloaded: {book['filename']} ({len(response.text)} characters)")
            except requests.RequestException as e:
                print(f"❌ Error downloading {book['name']}: {e}")

        return books


def train_protos_on_books(assistant):
    """
    Обучение Протоса на книгах с поддержкой разных форматов.
    """
    print("=" * 60)
    print("📚 ОБУЧЕНИЕ ПРОТОСА НА КНИГАХ")
    print("=" * 60)

    downloader = BookDownloader()
    books = downloader.load_local_books()

    if not books:
        print("\n⚠️ Нет книг для обучения!")
        print("📥 Положите книги в папку data/books/ в форматах:")
        print("   .txt, .epub, .fb2, .doc, .docx, .pdf, .zip, .rar, .json, .md, .html")
        return

    print(f"\n📚 Найдено {len(books)} книг:")
    for book in books:
        size = os.path.getsize(book) / 1024
        print(f"  - {os.path.basename(book)} ({size:.1f} KB)")

    print(f"\n📚 Обучение на {len(books)} книгах...")
    print("=" * 60)

    total_text = 0

    for i, book_path in enumerate(books, 1):
        print(f"\n📖 [{i}/{len(books)}] Обработка: {os.path.basename(book_path)}")

        try:
            text = downloader.read_book(book_path)

            if not text:
                print(f"  ❌ Не удалось прочитать файл")
                continue

            if len(text) > 50000:
                text = text[:50000]
                print(f"  ⚠️ Текст сокращен до 50000 символов")

            assistant.learning_manager.learn_from_text(text, source=f"book:{os.path.basename(book_path)}")
            total_text += len(text)
            print(f"  ✅ Добавлена книга ({len(text)} символов)")

        except Exception as e:
            print(f"  ❌ Ошибка обработки книги: {e}")

    if total_text == 0:
        print("\n❌ Не удалось прочитать ни одной книги!")
        return

    print(f"\n🧠 Запуск обучения на {total_text} символах...")
    print("=" * 60)
    assistant.learning_manager.train(epochs=10, sequence_length=100)
    assistant.learning_manager.save_model('models/protos_lstm.json')

    print("\n" + "=" * 60)
    print("✅ ОБУЧЕНИЕ НА КНИГАХ ЗАВЕРШЕНО!")
    print("=" * 60)


def continue_training_from_saved(assistant):
    """
    Продолжение обучения из сохраненной модели.
    """
    print("=" * 60)
    print("📚 ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ")
    print("=" * 60)

    downloader = BookDownloader()
    books = downloader.load_local_books()

    if not books:
        print("⚠️ Нет локальных книг для обучения.")
        return

    print(f"\n📚 Найдено {len(books)} книг:")
    for book in books:
        size = os.path.getsize(book) / 1024
        print(f"  - {os.path.basename(book)} ({size:.1f} KB)")

    for i, book_path in enumerate(books, 1):
        print(f"\n📖 [{i}/{len(books)}] Обработка: {os.path.basename(book_path)}")

        try:
            text = downloader.read_book(book_path)

            if not text:
                print(f"  ❌ Не удалось прочитать файл")
                continue

            if len(text) > 50000:
                text = text[:50000]
                print(f"  ⚠️ Текст сокращен до 50000 символов")

            assistant.learning_manager.learn_from_text(text, source=f"book:{os.path.basename(book_path)}")
            print(f"  ✅ Добавлена книга ({len(text)} символов)")

        except Exception as e:
            print(f"  ❌ Ошибка обработки книги: {e}")

    print("\n🧠 Запуск обучения...")
    assistant.learning_manager.train(epochs=5, sequence_length=100)
    assistant.learning_manager.save_model('models/protos_lstm.json')

    print("\n✅ Обучение завершено!")