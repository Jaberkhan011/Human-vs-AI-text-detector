# TextGuard — AI vs Human Text Detector (plain HTML/CSS/JS)

No build step, no framework, no npm install required. Three files:

```
index.html   Markup and layout
styles.css   All styling (dark "ink" shell, paper-toned input, result card)
app.js       Live stats, fetch to the backend, state rendering
config.js    One line: the backend's base URL
```

## Running locally

Just open `index.html` in a browser, or serve the folder with any static
server, e.g.:

```bash
python3 -m http.server 5173
```

Then visit `http://localhost:5173`.

If your backend isn't running on `http://localhost:8000`, edit `config.js`:

```js
window.TEXTGUARD_API_BASE_URL = "http://your-backend-host:8000";
```

## Backend

`backend/app.py` is your original FastAPI app with one addition: a
`POST /predict` endpoint that returns exactly what this frontend expects —
`ai_probability`, `human_probability`, a High/Medium/Low `confidence`
label, and the deterministic text features. Your original `/classify`,
`/features`, and `/health` endpoints are untouched.

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
```

CORS is already open (`allow_origins=["*"]`) for local development —
tighten this before deploying anywhere public.

## Notes

- "Analyze Text" is disabled until there's text in the box.
- Passages under ~25 words show a short reliability warning.
- Network errors, timeouts, and server errors all show a friendly message
  in the report panel instead of a raw exception.
- The feature grid stays collapsed by default and re-collapses on each new
  analysis, so a long list of stats doesn't dominate the screen.
# AI-vs-Human-Text-Detector
# Human-vs-AI-text-detector
