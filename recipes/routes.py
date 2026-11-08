"""Private pages for the recipe tool. They stay out of PAGES and the sitemap.

Every route returns 404 until the password, secret and database path are set, so
a missing setting can't expose the tool.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode, urlsplit

from fastapi import APIRouter, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from starlette.datastructures import FormData

from recipes import auth, backup, convert, extract
from recipes.fetch import BlockedBySite, FetchError, canonicalize_url, fetch_page
from recipes.store import SORT_NAME, SORT_NEWEST, SORT_ORDERS, Recipe, RecipeStore

logger = logging.getLogger(__name__)

DEFAULT_DAILY_IMPORT_LIMIT = 20
MAX_LIST_ITEMS = 200
MAX_FIELD_CHARS = 5_000
MAX_TITLE_CHARS = 200
SORT_COOKIE = "recipes_sort"
SORT_COOKIE_SECONDS = 365 * 24 * 60 * 60
FLAG_LABELS = {
    convert.FLAG_VOLUME_ONLY: "Density unknown, so this stays a volume",
    convert.FLAG_ASSUMED_US: "Assumed US measures",
    convert.FLAG_UNCLEAR_TEMPERATURE: "The temperature's unit is unclear",
    convert.FLAG_VAGUE_OVEN: "The oven temperature is vague",
    extract.FLAG_NOT_IN_SOURCE: "This line wasn't found in the source",
    extract.FLAG_SOURCE_TRUNCATED: "The source was too long, so only part of it was read",
}
PRIVATE_HEADERS = {
    "X-Robots-Tag": "noindex, nofollow",
    "Cache-Control": "no-store",
}

RenderPage = Callable[..., HTMLResponse]


@dataclass(frozen=True)
class Settings:
    password: str
    secret: str
    db_path: Path


def settings_or_404() -> Settings:
    password = os.environ.get("RECIPES_PASSWORD")
    secret = os.environ.get("RECIPES_SECRET")
    db_path = os.environ.get("RECIPES_DB_PATH")
    if not password or not secret or not db_path:
        raise HTTPException(status_code=404)
    return Settings(password, secret, Path(db_path))


def storage_warning(db_path: Path) -> str | None:
    """Warn if saved recipes would vanish on the next deploy.

    Only a mounted disk survives one. Without it, saving still appears to work.
    """
    if os.environ.get("PRODUCTION", "").lower() != "true":
        return None
    on_disk = any(
        os.path.ismount(parent) for parent in db_path.parents if parent != parent.parent
    )
    if on_disk:
        return None
    return (
        "The database isn't on a Render disk, so saved recipes "
        "will be lost on the next deploy."
    )


def daily_import_limit() -> int:
    value = os.environ.get("RECIPES_DAILY_IMPORT_LIMIT", "")
    return int(value) if value.isdigit() else DEFAULT_DAILY_IMPORT_LIMIT


def _private(response: Response) -> Response:
    response.headers.update(PRIVATE_HEADERS)
    return response


def _redirect(path: str) -> Response:
    return _private(RedirectResponse(path, status_code=303))


def _is_logged_in(request: Request, settings: Settings) -> bool:
    return auth.session_is_valid(request.cookies.get(auth.COOKIE_NAME), settings.secret)


def _field(form: FormData, name: str) -> str:
    value = form.get(name)
    return value.strip() if isinstance(value, str) else ""


def _list_field(form: FormData, name: str) -> list[str]:
    return [value if isinstance(value, str) else "" for value in form.getlist(name)]


def _flags(value: str) -> list[str]:
    return [flag for flag in value.split(",") if flag in FLAG_LABELS]


def _uses_assumed_us(recipe: Recipe) -> bool:
    """Whether any line assumed US measures, which gets one note for the recipe."""
    return any(
        convert.FLAG_ASSUMED_US in item.get("flags", [])
        for item in [*recipe.ingredients, *recipe.steps]
    )


def _safe_return_path(value: str, fallback: str) -> str:
    """Only return to the tool's own pages, so a form can't redirect elsewhere."""
    parts = urlsplit(value)
    is_tool_page = parts.path == "/recipes" or parts.path.startswith("/recipes/")
    if parts.scheme or parts.netloc or not is_tool_page or "\\" in value:
        return fallback
    return value


def _list_url(query: str, favorites_only: bool, sort: str | None = None) -> str:
    """Build a link to the list, keeping whichever settings apply."""
    params: dict[str, str] = {}
    if query:
        params["q"] = query
    if favorites_only:
        params["favorites"] = "1"
    if sort:
        params["sort"] = sort
    return f"/recipes?{urlencode(params)}" if params else "/recipes"


def _safe_source_url(value: str) -> str | None:
    return value if value.startswith(("http://", "https://")) else None


def _parse_saved_recipe(form: FormData) -> dict[str, Any] | None:
    """Read the review form. Returns None if it isn't a complete, sensible recipe."""
    amounts = _list_field(form, "amount")
    names = _list_field(form, "name")
    originals = _list_field(form, "original")
    ingredient_flags = _list_field(form, "ingredient_flags")
    steps = _list_field(form, "step")
    step_originals = _list_field(form, "step_original")
    step_flags = _list_field(form, "step_flags")

    title = _field(form, "title")
    columns = [amounts, names, originals, ingredient_flags]
    step_columns = [steps, step_originals, step_flags]
    if (
        not title
        or len(title) > 200
        or len({len(column) for column in columns}) != 1
        or len({len(column) for column in step_columns}) != 1
        or len(names) > MAX_LIST_ITEMS
        or len(steps) > MAX_LIST_ITEMS
        or any(
            len(value) > MAX_FIELD_CHARS
            for column in columns + step_columns
            for value in column
        )
    ):
        return None

    ingredients = [
        {
            "amount": amount.strip(),
            "name": name.strip(),
            "original": original,
            "flags": _flags(flags),
        }
        for amount, name, original, flags in zip(
            amounts, names, originals, ingredient_flags, strict=True
        )
        if name.strip()
    ]
    saved_steps = [
        {"text": text.strip(), "original": original, "flags": _flags(flags)}
        for text, original, flags in zip(steps, step_originals, step_flags, strict=True)
        if text.strip()
    ]
    if not ingredients or not saved_steps:
        return None
    return {
        "title": title,
        "servings": _field(form, "servings")[:200],
        "source_url": _safe_source_url(_field(form, "source_url")),
        "ingredients": ingredients,
        "steps": saved_steps,
        "flags": _flags(_field(form, "recipe_flags")),
    }


def create_router(render_page: RenderPage) -> APIRouter:
    router = APIRouter(prefix="/recipes", include_in_schema=False)

    def page(template: str, status_code: int = 200, **context: Any) -> Response:
        return _private(
            render_page(
                template, status_code=status_code, flag_labels=FLAG_LABELS, **context
            )
        )

    def recipe_page(recipe: Recipe, status_code: int = 200, **context: Any) -> Response:
        return page(
            "recipes_detail.html",
            status_code=status_code,
            recipe=recipe,
            assumed_us=_uses_assumed_us(recipe),
            assumed_us_flag=convert.FLAG_ASSUMED_US,
            **context,
        )

    def import_page(
        store: RecipeStore,
        error: str | None = None,
        url: str = "",
        status_code: int = 200,
        paste_link: str = "",
        paste_text: str = "",
        suggest_paste: bool = False,
    ) -> Response:
        return page(
            "recipes_import.html",
            status_code=status_code,
            error=error,
            url=url,
            paste_link=paste_link,
            paste_text=paste_text,
            suggest_paste=suggest_paste,
            max_paste_chars=extract.MAX_SOURCE_CHARS,
            imports_today=store.imports_today(),
            daily_limit=daily_import_limit(),
            usage=store.usage_this_month(),
            model=extract.model_name(),
        )

    @router.get("/login")
    async def login_form(request: Request) -> Response:
        settings = settings_or_404()
        if _is_logged_in(request, settings):
            return _redirect("/recipes")
        return page("recipes_login.html", error=None)

    @router.post("/login")
    async def login(request: Request) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not auth.limiter.allowed():
            return page(
                "recipes_login.html",
                status_code=429,
                error="Too many attempts. Try again in 15 minutes.",
            )
        form = await request.form()
        if not auth.password_matches(_field(form, "password"), settings.password):
            auth.limiter.record_failure()
            return page(
                "recipes_login.html",
                status_code=401,
                error="That password isn't right.",
            )
        auth.limiter.reset()
        response = _redirect("/recipes")
        response.set_cookie(
            auth.COOKIE_NAME,
            auth.make_session(settings.secret),
            max_age=auth.SESSION_SECONDS,
            path="/recipes",
            httponly=True,
            samesite="strict",
            secure=auth.secure_cookies(),
        )
        return response

    @router.post("/logout")
    async def logout(request: Request) -> Response:
        settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        response = _redirect("/recipes/login")
        response.delete_cookie(auth.COOKIE_NAME, path="/recipes")
        return response

    @router.get("")
    async def recipe_list(
        request: Request, q: str = "", favorites: str = "", sort: str = ""
    ) -> Response:
        settings = settings_or_404()
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        store = RecipeStore(settings.db_path)
        query = q.strip()[:200]
        favorites_only = favorites == "1"
        chosen_sort = sort if sort in SORT_ORDERS else None
        remembered = request.cookies.get(SORT_COOKIE)
        current_sort = chosen_sort or (
            remembered if remembered in SORT_ORDERS else SORT_NEWEST
        )
        if query:
            recipes = store.search(
                query, sort=current_sort, favorites_only=favorites_only
            )
        else:
            recipes = store.recent(sort=current_sort, favorites_only=favorites_only)
        response = page(
            "recipes_list.html",
            recipes=recipes,
            query=query,
            favorites_only=favorites_only,
            current_sort=current_sort,
            sort_newest=SORT_NEWEST,
            sort_name=SORT_NAME,
            all_url=_list_url(query, False),
            favorites_url=_list_url(query, True),
            newest_url=_list_url(query, favorites_only, SORT_NEWEST),
            name_url=_list_url(query, favorites_only, SORT_NAME),
            return_to=_list_url(query, favorites_only),
            backup_warning=backup.backup_warning(store),
            storage_warning=storage_warning(settings.db_path),
            backup_status=backup.backup_status(store),
        )
        if chosen_sort:
            response.set_cookie(
                SORT_COOKIE,
                chosen_sort,
                max_age=SORT_COOKIE_SECONDS,
                path="/recipes",
                httponly=True,
                samesite="lax",
                secure=auth.secure_cookies(),
            )
        return response

    @router.get("/export")
    async def export(request: Request) -> Response:
        settings = settings_or_404()
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        return _private(
            Response(
                RecipeStore(settings.db_path).export_json(),
                media_type="application/json",
                headers={"Content-Disposition": 'attachment; filename="recipes.json"'},
            )
        )

    @router.get("/import")
    async def import_form(request: Request) -> Response:
        settings = settings_or_404()
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        return import_page(RecipeStore(settings.db_path))

    @router.post("/import")
    async def import_url(request: Request) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")

        store = RecipeStore(settings.db_path)
        form = await request.form()
        url = _field(form, "url")
        if not url:
            return import_page(store, "Enter a link to import.", url, 400)
        if not url.lower().startswith(("http://", "https://")):
            url = f"https://{url}"

        existing = store.find_by_source(canonicalize_url(url))
        if existing:
            return _redirect(f"/recipes/{existing}?{urlencode({'already': 1})}")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return import_page(
                store, "The Anthropic API key isn't set up yet.", url, 503
            )
        if store.imports_today() >= daily_import_limit():
            return import_page(
                store, "Today's import limit has been reached.", url, 429
            )

        try:
            fetched = await run_in_threadpool(fetch_page, url)
            final_url = canonicalize_url(fetched.url)
            existing = store.find_by_source(final_url)
            if existing:
                return _redirect(f"/recipes/{existing}?{urlencode({'already': 1})}")
            source = await run_in_threadpool(
                extract.trim_page, fetched.html, fetched.url
            )
            recipe, usage = await run_in_threadpool(extract.extract_recipe, source)
            store.record_import(usage.model, usage.input_tokens, usage.output_tokens)
            draft = extract.build_draft(recipe, source, final_url)
        except BlockedBySite as e:
            # The page can't be read, so go straight to pasting its text.
            return import_page(
                store,
                f"{e} Paste the recipe text below instead.",
                url,
                400,
                paste_link=url,
                suggest_paste=True,
            )
        except (FetchError, extract.ExtractionError) as e:
            return import_page(store, str(e), url, 400, paste_link=url)
        return page("recipes_review.html", draft=draft)

    @router.post("/paste")
    async def import_pasted(request: Request) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")

        store = RecipeStore(settings.db_path)
        form = await request.form()
        text = _field(form, "text")
        link = _field(form, "source_link")
        if link and not link.lower().startswith(("http://", "https://")):
            link = f"https://{link}"
        # The link is only stored, as a reference. It's never requested.
        source_url = canonicalize_url(link) if link else None

        def failed(message: str, status_code: int) -> Response:
            return import_page(
                store,
                message,
                status_code=status_code,
                paste_link=link,
                paste_text=text,
                suggest_paste=True,
            )

        if source_url:
            existing = store.find_by_source(source_url)
            if existing:
                return _redirect(f"/recipes/{existing}?{urlencode({'already': 1})}")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            return failed("The Anthropic API key isn't set up yet.", 503)
        try:
            source = extract.pasted_source(text)
        except extract.ExtractionError as e:
            return failed(str(e), 400)
        if store.imports_today() >= daily_import_limit():
            return failed("Today's import limit has been reached.", 429)

        try:
            recipe, usage = await run_in_threadpool(extract.extract_recipe, source)
            store.record_import(usage.model, usage.input_tokens, usage.output_tokens)
            draft = extract.build_draft(recipe, source, source_url)
        except extract.ExtractionError as e:
            return failed(str(e), 400)
        return page("recipes_review.html", draft=draft)

    @router.post("/save")
    async def save(request: Request) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        recipe = _parse_saved_recipe(await request.form())
        if recipe is None:
            raise HTTPException(status_code=400, detail="That recipe is incomplete.")
        recipe_id = RecipeStore(settings.db_path).save(**recipe)
        return _redirect(f"/recipes/{recipe_id}")

    @router.get("/{recipe_id:int}")
    async def detail(request: Request, recipe_id: int, already: int = 0) -> Response:
        settings = settings_or_404()
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        recipe = RecipeStore(settings.db_path).get(recipe_id)
        if recipe is None:
            raise HTTPException(status_code=404)
        return recipe_page(recipe, already=bool(already))

    @router.post("/{recipe_id:int}/rename")
    async def rename(request: Request, recipe_id: int) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        store = RecipeStore(settings.db_path)
        recipe = store.get(recipe_id)
        if recipe is None:
            raise HTTPException(status_code=404)

        form = await request.form()
        title = _field(form, "title")
        if not title or len(title) > MAX_TITLE_CHARS:
            return recipe_page(
                recipe,
                status_code=400,
                rename_error=f"A name needs 1 to {MAX_TITLE_CHARS} characters.",
                rename_value=title[:MAX_TITLE_CHARS],
            )
        store.rename(recipe_id, title)
        return _redirect(f"/recipes/{recipe_id}")

    @router.post("/{recipe_id:int}/favorite")
    async def favorite(request: Request, recipe_id: int) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        form = await request.form()
        # Each tap sets a state, so a double tap can't undo itself.
        wanted = _field(form, "favorite")
        if wanted not in ("0", "1"):
            raise HTTPException(status_code=400)
        if not RecipeStore(settings.db_path).set_favorite(recipe_id, wanted == "1"):
            raise HTTPException(status_code=404)
        return _redirect(
            _safe_return_path(_field(form, "next"), f"/recipes/{recipe_id}")
        )

    @router.post("/{recipe_id:int}/delete")
    async def delete(request: Request, recipe_id: int) -> Response:
        settings = settings_or_404()
        if not auth.is_same_origin(request):
            raise HTTPException(status_code=403)
        if not _is_logged_in(request, settings):
            return _redirect("/recipes/login")
        RecipeStore(settings.db_path).delete(recipe_id)
        return _redirect("/recipes")

    return router
