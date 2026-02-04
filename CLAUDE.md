# AI Review Manager

## О проекте

AI-powered code reviewer для GitLab. Self-hosted FastAPI сервис, который автоматически ревьюит Merge Requests с помощью LLM (Gemini, YandexGPT).

**Ключевые особенности:**
- Автоматический ревью при создании/обновлении MR
- Ручной запуск через команду `/review` в комментариях
- Inline комментарии + summary
- Настраиваемые категории проверок и язык комментариев
- Multi-provider LLM (Gemini 2.5, YandexGPT 5)

## Архитектура

```
GitLab webhook → FastAPI → LLM Provider → GitLab API (comments)
```

**Структура:**
```
src/ai_review/
├── main.py           # FastAPI app, endpoints
├── config.py         # Settings (env vars)
├── models/           # Pydantic models
├── providers/        # LLM абстракции (Gemini, Yandex)
├── platforms/        # Git платформы (GitLab)
└── review/           # Бизнес-логика (engine, parser, prompts)
```

## Команды

```bash
# Запуск dev сервера
uvicorn ai_review.main:app --reload

# Тесты
pytest tests/                    # Все (без e2e)
pytest tests/ -m unit            # Только unit
pytest tests/ -m integration     # Только integration
pytest tests/e2e/ -m e2e -v      # E2E с реальными API

# Docker
docker build -t ai-review-manager .
docker-compose up -d
```

## Конфигурация

**Environment variables:**
- `GITLAB_TOKEN` - токен для GitLab API
- `GITLAB_WEBHOOK_SECRET` - секрет для верификации webhook
- `GEMINI_API_KEY` - ключ Google Gemini API
- `YANDEX_API_KEY` - ключ YandexGPT API
- `YANDEX_FOLDER_ID` - folder ID для YandexGPT
- `DEFAULT_PROVIDER` - провайдер по умолчанию (gemini/yandex)
- `DEFAULT_LANGUAGE` - язык комментариев (en/ru)

**Конфиг репозитория** (`.ai-review.yaml`):
```yaml
language: ru
provider: gemini
checks: [bugs, security, performance, readability, best-practices]
exclude: ["*.generated.ts", "vendor/"]
auto_review: true
min_severity: low
```

## Правила разработки

### Code Style
- Python 3.11+
- Type hints везде
- Pydantic для валидации
- Async/await для I/O операций
- Без LangChain - своя абстракция провайдеров

### Тестирование
- TDD подход
- Unit тесты для бизнес-логики
- Integration тесты с httpx_mock для HTTP
- E2E тесты для реальных API (опционально)

### Git
- Conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`
- Не коммитить `.env` и секреты
- PR для значительных изменений

## Roadmap

**MVP (done):**
- [x] GitLab webhook integration
- [x] Gemini provider
- [x] YandexGPT provider
- [x] Inline comments + summary
- [x] Configurable via .ai-review.yaml
- [x] Docker setup

**Next:**
- [ ] GitHub support
- [ ] Jira/Confluence integration (проверка против требований)
- [ ] Redis queue для async processing
- [ ] Web UI / dashboard
- [ ] Кэширование (не ревьюить одни файлы повторно)

## API Endpoints

- `GET /health` - health check
- `POST /webhook/gitlab` - GitLab webhook (MR events, note events)

## LLM Providers

### Gemini
- Model: `gemini-2.5-flash`
- Endpoint: `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent`

### YandexGPT
- Model: `yandexgpt/latest` (YandexGPT 5)
- Endpoint: `https://llm.api.cloud.yandex.net/foundationModels/v1/completion`
