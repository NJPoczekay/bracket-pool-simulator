"""Local on-disk persistence helpers for saved Bracket Lab drafts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from bracket_sim.domain.product_models import SaveBracketRequest, SavedBracket, SavedBracketSummary
from bracket_sim.infrastructure.storage.path_defaults import safe_path_token
from bracket_sim.infrastructure.storage.run_artifacts import write_json_atomic

_JSON_GLOB = "*.json"


def list_saved_brackets(
    *,
    storage_dir: Path,
    dataset_hash: str | None = None,
) -> list[SavedBracketSummary]:
    """Return saved bracket summaries sorted by newest first."""

    records = [
        record
        for record in _read_saved_bracket_records(storage_dir)
        if dataset_hash is None or record.dataset_hash == dataset_hash
    ]
    records.sort(key=lambda record: record.updated_at, reverse=True)
    return [record.to_summary() for record in records]


def load_saved_bracket(
    *,
    storage_dir: Path,
    bracket_id: str,
) -> SavedBracket:
    """Load one saved bracket by id."""

    normalized_id = _normalize_bracket_id(bracket_id)
    if not normalized_id:
        msg = "Saved bracket id is required"
        raise ValueError(msg)

    path = storage_dir / f"{normalized_id}.json"
    if not path.exists():
        msg = f"Saved bracket {normalized_id!r} was not found"
        raise ValueError(msg)

    return _load_saved_bracket_file(path)


def save_bracket(
    *,
    storage_dir: Path,
    dataset_hash: str,
    request: SaveBracketRequest,
) -> SavedBracket:
    """Persist one bracket draft and return the saved record."""

    existing = _read_saved_bracket_records(storage_dir)
    bracket_id = _resolve_bracket_id(
        request=request,
        dataset_hash=dataset_hash,
        existing=existing,
    )
    saved = SavedBracket(
        bracket_id=bracket_id,
        name=request.name,
        bracket=request.bracket,
        pool_settings=request.pool_settings,
        completion_mode=request.completion_mode,
        dataset_hash=dataset_hash,
        updated_at=datetime.now(UTC),
    )
    write_json_atomic(
        path=storage_dir / f"{bracket_id}.json",
        payload=saved.model_dump(mode="json"),
    )
    return saved


def _resolve_bracket_id(
    *,
    request: SaveBracketRequest,
    dataset_hash: str,
    existing: list[SavedBracket],
) -> str:
    explicit_id = _normalize_bracket_id(request.bracket_id)
    if explicit_id:
        return explicit_id

    for record in existing:
        if (
            record.dataset_hash == dataset_hash
            and record.name.casefold() == request.name.casefold()
        ):
            return record.bracket_id

    base_id = safe_path_token(request.name, default="bracket")
    existing_ids = {record.bracket_id for record in existing}
    if base_id not in existing_ids:
        return base_id

    suffix = 2
    while True:
        candidate_id = f"{base_id}-{suffix}"
        if candidate_id not in existing_ids:
            return candidate_id
        suffix += 1


def _normalize_bracket_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = safe_path_token(value, default="")
    return normalized or None


def _read_saved_bracket_records(storage_dir: Path) -> list[SavedBracket]:
    if not storage_dir.exists():
        return []

    records: list[SavedBracket] = []
    for path in sorted(storage_dir.glob(_JSON_GLOB)):
        try:
            records.append(_load_saved_bracket_file(path))
        except ValueError:
            continue
    return records


def _load_saved_bracket_file(path: Path) -> SavedBracket:
    try:
        payload = path.read_text(encoding="utf-8")
    except OSError as exc:
        msg = f"Unable to read saved bracket {path.name}: {exc}"
        raise ValueError(msg) from exc

    try:
        return SavedBracket.model_validate_json(payload)
    except ValidationError as exc:
        msg = f"Invalid saved bracket payload in {path.name}"
        raise ValueError(msg) from exc
