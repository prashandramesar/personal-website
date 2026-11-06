# Personal Portfolio (prashand.nl)

Personal portfolio site with a dog breed classifier. Hosted on Render.

## Architecture
- Single FastAPI app serves everything (pages, static files, API). Entry file: `app.py`.
- Flask is only used to render Jinja templates inside an app context (`render_page`). Do not add Flask routes.
- Model: MobileNetV2 as TensorFlow Lite (`dog_breed_model_quantized.tflite`) with `label_map.json`, run with the `ai-edge-litert` interpreter and loaded in the FastAPI lifespan handler. Training code is in the `dog-breed-model` repo.
- Model input is 224x224 RGB `float32` scaled to [0, 1], which matches training. Output is 120 classes.
- Uploads are processed in memory. Never write them to disk or under `static/`.

## Endpoints
- `/`, `/kaai`, `/projects`, `/about`, `/dog_classifier`, `/android_app`, `/this_website`: HTML pages
- `/sitemap.xml`, `/robots.txt`: for search engines
- `/api/dog` (GET): API root
- `/api/dog/predict` (POST, multipart `file`): returns top 3 breeds with confidence. Errors use FastAPI's `detail` field.
- `/predict` (POST, multipart `file`): same prediction, used by the website form and the Android app. Errors use an `error` field.
- `/health` (GET): returns 503 if the model failed to load

## Rules
- Never make HTTP requests from the app to its own URL. This caused an infinite loop before. Call the model code directly.
- Released Android builds call `POST /predict` and read `breed` and `confidence`, and can't be updated by a deploy. Don't rename or remove those fields or the route. Adding fields is safe.
- `breed` holds the raw label, like `n02099601-golden_retriever`. Use `display_name` for anything shown to people.
- Check model globals for None before use (mypy requires it).
- No bare `except`. Catch specific exceptions.
- Allowed upload types: png, jpg, jpeg, checked from the file contents. Max 16MB.
- The Render instance has 0.5 CPU and 512 MB RAM. Check memory use before anything that loads large data, and don't hold extra full-size copies of uploaded images.
- Keep Flask pinned to 2.2.3 and Werkzeug to 2.2.3 in requirements.txt.
- Don't claim accuracy figures on the site unless they come from a measured result.

## Dependencies
- `requirements.txt` is what Render installs. `requirements-dev.txt` adds the dev tools.
- `pyproject.toml` mirrors both, so update it and regenerate `uv.lock` when changing a version.
- Python 3.11 (`.python-version`). Pillow 13 has no Python 3.10 wheels.

## Frontend
- Templates in `templates/`, and each page extends `base.html`. Static files in `static/`.
- Styling uses Tailwind 2.2.19 utility classes, served from a trimmed `static/css/site.css`. After adding a new class to a template, run `python scripts/build_css.py` and commit the result, or the class does nothing. A test fails if a class is missing. Match the existing look (white cards, `rounded-lg shadow-md p-8`, gray text).
- New pages go in the `PAGES` table in `app.py`, which also feeds the sitemap.

## Positioning and content
- The site presents Prashand as someone who takes AI products from business need to production. Hands-on projects are hobby work that keeps Prashand close to the technology. Keep this framing.
- KAAI is the only name used for the call center project. Don't name an employer on its case study or link it to one, and credit the cost result to senior management.
- Only state facts Prashand has supplied or the code proves. Don't invent results, team sizes or decisions.

## Commands
- Setup: `python -m venv .venv`, `.venv\Scripts\activate`, `pip install -r requirements-dev.txt`
- Run locally: `python app.py` (port 10000, or the PORT env var)
- Lint and format: `ruff check .` and `ruff format --check .`
- Types: `mypy .`
- Tests: `pytest`
- Deploy: merge a PR into main, and Render redeploys using `start.sh`

## Workflow
- Work on a branch, never commit straight to main.
- Run ruff, mypy and pytest before committing. CI runs them too.
- Keep changes small and explain what changed in the commit message.
