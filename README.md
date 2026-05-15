# DocuMind Backend

## Environment setup

Create your local environment file from the committed example:

```bash
cp .env.example .env
```

Then replace the placeholder values in `.env` with your real API keys.

Required variables:

- `GOOGLE_API_KEY`
- `GROQ_API_KEY`
- `HUGGINGFACE_API_KEY`

Do not commit `.env`; it is ignored by git. Commit updates to `.env.example` when the required variable structure changes.
