# WebLogAnalyzer

Приложение для автоматического выявления и LLM-интерпретации паттернов
активности пользователей веб-сайтов на основе анализа веб-логов.

## Быстрый старт

```bash
# 1. Клонировать / распаковать проект
cd web_log_analyzer

# 2. Установить зависимости
pip install -r requirements.txt

# 3. Задать API-ключ
cp .env.example .env
# отредактировать .env, вписать OPENAI_API_KEY=sk-...

# 4. Запустить
python app.py
# Открыть браузер: http://127.0.0.1:7860
```

## Структура проекта

```
web_log_analyzer/
├── app.py                   # Точка входа (Gradio)
├── pipeline/
│   ├── preprocessor.py      # Парсинг, фильтрация, сессионизация
│   ├── clustering.py        # K-Means + метод локтя
│   ├── pattern_mining.py    # PrefixSpan + FP-Growth
│   ├── anomaly.py           # Isolation Forest + z-score + IQR
│   └── interpreter.py       # Интеграция с OpenAI API
├── utils/
│   ├── session_model.py     # Датакласс Session
│   └── url_normalizer.py    # Нормализация URL
├── data/                    # Результаты анализа (CSV, JSON)
├── tests/                   # 138 тестов
├── requirements.txt
└── .env.example
```

## Конвейер

```
Лог-файл → Предобработка → Кластеризация (K-Means)
                         → Паттерны (PrefixSpan + FP-Growth)
                         → Аномалии (Isolation Forest + z-score + IQR)
                         → LLM-интерпретация (GPT-4o-mini)
                         → Рекомендации для тестировщиков
```

## Поддерживаемые форматы логов

- **Combined Log Format** (Apache / Nginx) — основной
- **NASA-HTTP** (нестандартная временна́я метка)
- **W3C Extended Log Format** — частичная поддержка

## Параметры запуска

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--port` | 7860 | Порт веб-сервера |
| `--host` | 127.0.0.1 | Хост |
| `--share` | False | Публичная ссылка (Gradio tunnel) |

## Тесты

```bash
python -m pytest tests/ -v
# 138 passed
```

## Результаты анализа

Сохраняются в директории `data/`:

| Файл | Содержимое |
|------|-----------|
| `sessions.csv` | Таблица сессий с признаками, кластерами и флагами аномалий |
| `patterns.json` | Последовательные паттерны (PrefixSpan) |
| `rules.json` | Ассоциативные правила (FP-Growth) |
| `interpretations.json` | LLM-интерпретации и рекомендации |
