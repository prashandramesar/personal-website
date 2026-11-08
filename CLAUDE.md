# Personal Portfolio (prashand.nl)

Personal portfolio site with a dog breed classifier. Hosted on Render.

## Architecture
- Single FastAPI app serves everything (pages, static files, API). Entry file: `app.py`.
- Flask is only used to render Jinja templates inside an app context (`render_page`). Do not add Flask routes.
- Model: MobileNetV2 as TensorFlow Lite (`dog_breed_model_quantized.tflite`) with `label_map.json`, run with the `ai-edge-litert` interpreter and loaded in the FastAPI lifespan handler. Training code is in the `dog-breed-model` repo.
- Model input is 224x224 RGB `float32` scaled to [0, 1], which matches training. Output is 120 classes.
- Uploads are processed in memory. Never write them to disk or under `static/`.
- Recipe tool: private pages under `/recipes`, in the `recipes/` package. A link goes through `fetch.py` (safe fetching), `extract.py` (trim the page, one Claude call, then convert), a review page, and `store.py` (SQLite with FTS5 search).
- Recipe conversion: `recipes/convert.py` does all metric and Celsius arithmetic, and a language model must never calculate a quantity.
- `recipes/densities.json` holds cup weights from USDA FoodData Central, with the `fdc_id` and portion for each. Only add values with a source. If an ingredient isn't listed, show ml and flag it, and never guess grams.
- Saved recipes live in SQLite at `RECIPES_DB_PATH`, on the Render disk at `/var/data`. Anything else on the filesystem is lost on deploy. `recipes/backup.py` commits a daily JSON export to a private GitHub repo, and `scripts/restore_recipes.py` rebuilds the database from one.

## Endpoints
- `/`, `/kaai`, `/projects`, `/about`, `/dog_classifier`, `/android_app`, `/recipe_tool`, `/this_website`: HTML pages. `/recipe_tool` is the public demo of the recipe tool. It never calls the API or shows saved recipes, and only uses Prashand's own recipe.
- `/sitemap.xml`, `/robots.txt`: for search engines
- `/api/dog` (GET): API root
- `/api/dog/predict` (POST, multipart `file`): returns top 3 breeds with confidence. Errors use FastAPI's `detail` field.
- `/predict` (POST, multipart `file`): same prediction, used by the website form and the Android app. Errors use an `error` field.
- `/health` (GET): returns 503 if the model failed to load
- `/recipes`, `/recipes/login`, `/recipes/logout`, `/recipes/import`, `/recipes/paste`, `/recipes/save`, `/recipes/export`, `/recipes/{id}`, `/recipes/{id}/delete`: the private recipe tool. All return 404 until `RECIPES_PASSWORD`, `RECIPES_SECRET` and `RECIPES_DB_PATH` are set.

## Rules
- Never make HTTP requests from the app to its own URL. This caused an infinite loop before. Call the model code directly.
- Released Android builds call `POST /predict` and read `breed` and `confidence`, and can't be updated by a deploy. Don't rename or remove those fields or the route. Adding fields is safe.
- `breed` holds the raw label, like `n02099601-golden_retriever`. Use `display_name` for anything shown to people.
- Check model globals for None before use (mypy requires it).
- No bare `except`. Catch specific exceptions.
- Allowed upload types: png, jpg, jpeg, checked from the file contents. Max 16MB.
- The Render instance has 0.5 CPU and 512 MB RAM. Check memory use before anything that loads large data, and don't hold extra full-size copies of uploaded images.
- Don't add the YouTube Data API or a transcript library to the recipe tool. Its policies bar storing Non-Authorized Data past 30 days and creating derived data, with no personal-use exemption, so Shorts are pasted. Reread the policies first if this comes up again.
- Keep Flask pinned to 2.2.3 and Werkzeug to 2.2.3 in requirements.txt.
- Don't claim accuracy figures on the site unless they come from a measured result.
- Recipe tool pages stay out of `PAGES`, so they stay out of the sitemap. Every route must check the login, and every POST must also check `is_same_origin`.
- Only fetch user-supplied URLs through `fetch_page`, which checks every redirect against private addresses. If a site blocks the tool, report it and don't work around it, so no browser impersonation or proxies. `SOCIAL_DOMAINS` are never requested, because their terms bar scraping, so their posts are pasted instead.
- Text from fetched pages is untrusted. Keep it inside the `<source>` tags, and don't give the extraction call any tools.
- Secrets go in Render's Environment page, and `render.yaml` only lists their names with `sync: false`. Never put a secret in a URL, because `httpx` logs request URLs.
- Saved recipes are private and may be copyrighted. Never commit them to this public repo or show them on a public page. The backup code refuses a repo that isn't private.

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
