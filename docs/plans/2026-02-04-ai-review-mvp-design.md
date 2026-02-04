# AI Review Manager — MVP Design

## Overview

Open-source AI code reviewer для GitLab. Self-hosted FastAPI сервис, который автоматически ревьюит Merge Requests или запускается по команде `/review`.

**Ключевое отличие:** в будущем — проверка логики реализации против требований из Jira/Confluence, а не только синтаксис и безопасность.

## Архитектура

```
┌─────────────────────────────────────────────────────┐
│                    GitLab                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────────────┐  │
│  │   MR    │───▶│ Webhook │───▶│ Comment /review │  │
│  └─────────┘    └────┬────┘    └────────┬────────┘  │
└──────────────────────┼──────────────────┼───────────┘
                       │                  │
                       ▼                  ▼
              ┌────────────────────────────┐
              │   ai-review-manager        │
              │   (FastAPI + Docker)       │
              ├────────────────────────────┤
              │ • POST /webhook/gitlab     │
              │ • GET  /health             │
              └─────────────┬──────────────┘
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
         ┌────────┐   ┌──────────┐   ┌────────┐
         │ Gemini │   │  Yandex  │   │ Claude │
         └────────┘   └──────────┘   └────────┘
```

### Поток данных

1. GitLab отправляет webhook при создании MR или комментарии `/review`
2. Сервис получает event, загружает `.ai-review.yaml` из репо
3. Получает diff и контекст изменённых файлов через GitLab API
4. Формирует промпт, отправляет в выбранный LLM
5. Парсит ответ, постит inline comments + summary

## Технологии

- Python 3.11+
- FastAPI
- httpx (async HTTP клиент)
- Pydantic (валидация)
- unidiff (парсинг diff)
- Docker

**Без LangChain** — своя тонкая абстракция для LLM провайдеров.

## Структура проекта

```
ai-review-manager/
├── src/
│   └── ai_review/
│       ├── __init__.py
│       ├── main.py                 # FastAPI app, endpoints
│       ├── config.py               # Settings (env vars)
│       │
│       ├── providers/              # LLM абстракции
│       │   ├── __init__.py
│       │   ├── base.py             # LLMProvider ABC
│       │   ├── gemini.py
│       │   └── yandex.py
│       │
│       ├── platforms/              # Git платформы
│       │   ├── __init__.py
│       │   ├── base.py             # GitPlatform ABC
│       │   └── gitlab.py
│       │
│       ├── review/                 # Бизнес-логика
│       │   ├── __init__.py
│       │   ├── engine.py           # Оркестрация ревью
│       │   ├── parser.py           # Парсинг diff (unidiff)
│       │   └── prompts.py          # Шаблоны промптов
│       │
│       └── models/                 # Pydantic модели
│           ├── __init__.py
│           ├── config.py           # RepoConfig (.ai-review.yaml)
│           ├── review.py           # ReviewResult, Comment
│           └── webhook.py          # GitLab webhook events
│
├── tests/
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── README.md
```

## Конфигурация репозитория

Файл `.ai-review.yaml` в корне репо:

```yaml
# Язык комментариев
language: ru  # ru | en | etc

# LLM провайдер (переопределяет дефолт сервиса)
provider: gemini  # gemini | yandex | claude

# Какие проверки включены
checks:
  - bugs           # Логические ошибки, потенциальные баги
  - security       # Уязвимости (SQL injection, XSS, etc)
  - performance    # N+1, неоптимальные алгоритмы
  - readability    # Именование, сложность функций
  - best-practices # Идиомы языка, паттерны

# Файлы/папки игнорировать
exclude:
  - "*.generated.ts"
  - "vendor/"
  - "**/*.min.js"

# Авто-ревью или только по команде
auto_review: true  # false = только /review

# Severity threshold для комментариев
min_severity: low  # low | medium | high | critical
```

Если файла нет — сервис использует дефолты.

## API Endpoints

### POST /webhook/gitlab

Принимает все события, фильтрует нужные:

- **Merge Request Hook** (action: open, update) → если `auto_review: true` — запускает ревью
- **Note Hook** (комментарий с `/review`) → запускает ревью по команде

Обработка в `BackgroundTasks`, endpoint сразу возвращает 200.

### GET /health

```json
{"status": "ok", "version": "0.1.0"}
```

## Промпты и формат ответа LLM

### Структура промпта

```
System: Ты AI code reviewer. Анализируй код на: {enabled_checks}.
        Отвечай на языке: {language}.
        Формат ответа: JSON.

User:
  Файл: {file_path}

  Полный файл для контекста:
  ```{lang}
  {full_file_content}
  ```

  Изменения (diff):
  ```diff
  {diff_content}
  ```

  Ответь JSON:
  {
    "comments": [
      {
        "line": <номер строки в NEW файле>,
        "severity": "critical|high|medium|low",
        "category": "bugs|security|performance|readability|best-practices",
        "comment": "<текст комментария>"
      }
    ],
    "summary": "<общее резюме изменений, 2-3 предложения>"
  }
```

### Парсинг ответа

- Извлечь JSON из ответа (может быть обёрнут в markdown)
- Валидация через Pydantic модель `ReviewResult`
- Фильтрация по `min_severity` из конфига
- Маппинг `line` на позицию для GitLab inline comment API

### Edge cases

- LLM вернул невалидный JSON → retry 1 раз
- Пустой список comments → только summary comment

## GitLab API интеграция

### Используемые endpoints

```python
# 1. Получить изменения MR
GET /projects/{id}/merge_requests/{iid}/changes

# 2. Получить полный файл (для контекста)
GET /projects/{id}/repository/files/{file_path}/raw?ref={branch}

# 3. Получить .ai-review.yaml
GET /projects/{id}/repository/files/.ai-review.yaml/raw?ref={branch}

# 4. Постинг inline comment
POST /projects/{id}/merge_requests/{iid}/discussions

# 5. Постинг summary comment
POST /projects/{id}/merge_requests/{iid}/notes
```

### Аутентификация

- Project Access Token или Personal Access Token
- Scope: `api`
- Header: `PRIVATE-TOKEN: {token}`

### Формат комментариев

```markdown
💡 **Performance** | `medium`

Здесь N+1 запрос в цикле. Лучше загрузить все данные одним запросом.
```

## Конфигурация сервиса

```bash
# Обязательные
GITLAB_TOKEN=<токен с доступом к репозиториям>
GITLAB_WEBHOOK_SECRET=<секрет для верификации webhook>

# LLM провайдеры (минимум один)
GEMINI_API_KEY=<ключ Google AI>
YANDEX_API_KEY=<ключ YandexGPT>
YANDEX_FOLDER_ID=<folder id для YandexGPT>

# Опциональные
DEFAULT_PROVIDER=gemini          # дефолтный LLM
DEFAULT_LANGUAGE=en              # дефолтный язык комментариев
LOG_LEVEL=INFO
```

## Error Handling

| Ошибка | Действие |
|--------|----------|
| Невалидный webhook signature | 401, логируем, игнорируем |
| Файл `.ai-review.yaml` не найден | Используем дефолты |
| Невалидный `.ai-review.yaml` | Summary comment с ошибкой парсинга |
| LLM вернул невалидный JSON | Retry 1 раз, если опять — summary с raw ответом |
| LLM timeout / rate limit | Summary comment "Ревью временно недоступно" |
| GitLab API ошибка | Логируем, retry 1 раз |
| Файл слишком большой (>10k lines) | Пропускаем с комментарием |

### Формат ошибки в MR

```markdown
⚠️ **AI Review Error**

Не удалось выполнить ревью: {причина}

Попробуйте запустить повторно командой `/review`
```

## Что НЕ входит в MVP

- GitHub поддержка
- Интеграция с Jira/Confluence (проверка против требований)
- Очереди (Redis)
- Web UI
- Мульти-токен (разные токены для разных репо)

## Будущие улучшения

1. **Контекст кодовой базы** — подтягивать импортируемые модули, AST парсинг
2. **Интеграция с Jira/Confluence** — валидация логики против требований
3. **GitHub поддержка**
4. **Vector DB** — семантический поиск по кодовой базе
5. **Кэширование** — не ревьюить одни и те же файлы повторно
6. **Web UI** — дашборд с историей ревью, статистикой
