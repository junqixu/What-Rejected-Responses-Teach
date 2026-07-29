from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


GENERATOR_VERSION = "rebuttal-v1"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {value!r}")


def add_common_cli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=414)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", type=parse_bool, default=False)
    parser.add_argument(
        "--log_level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(levelname)s %(name)s: %(message)s",
    )


def _iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[dict[str, Any]]:
    """Stream a top-level JSON array without loading a large dataset into memory."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    finished = False
    with path.open("r", encoding="utf-8") as handle:
        while not finished:
            chunk = handle.read(chunk_size)
            if not chunk:
                finished = True
            buffer += chunk
            cursor = 0
            while True:
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if not started:
                    if cursor >= len(buffer):
                        break
                    if buffer[cursor] != "[":
                        raise ValueError(f"{path} is not a top-level JSON array")
                    started = True
                    cursor += 1
                    continue
                while cursor < len(buffer) and (buffer[cursor].isspace() or buffer[cursor] == ","):
                    cursor += 1
                if cursor < len(buffer) and buffer[cursor] == "]":
                    return
                if cursor >= len(buffer):
                    break
                try:
                    value, end = decoder.raw_decode(buffer, cursor)
                except json.JSONDecodeError:
                    if finished:
                        raise
                    break
                if not isinstance(value, dict):
                    raise ValueError(f"{path} contains a non-object array item")
                yield value
                cursor = end
            buffer = buffer[cursor:]
    if started:
        raise ValueError(f"unterminated JSON array in {path}")


def iter_records(paths: str | Path | Sequence[str | Path]) -> Iterator[dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for raw_path in paths:
        path = Path(raw_path)
        if path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    yield value
            continue
        with path.open("r", encoding="utf-8") as handle:
            first_character = ""
            while not first_character:
                character = handle.read(1)
                if not character:
                    break
                if not character.isspace():
                    first_character = character
        if first_character == "[":
            yield from _iter_json_array(path)
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except json.JSONDecodeError as error:
            # Several legacy datasets use JSONL content with a .json suffix.
            if "Extra data" not in str(error):
                raise
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{line_number} is not a JSON object")
                    yield row
            continue
        if isinstance(value, dict):
            yield value
        elif isinstance(value, list):
            for index, row in enumerate(value):
                if not isinstance(row, dict):
                    raise ValueError(f"{path}[{index}] is not a JSON object")
                yield row
        else:
            raise ValueError(f"{path} must contain an object, an array, or JSONL objects")


def load_records(
    paths: str | Path | Sequence[str | Path], max_samples: int | None = None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in iter_records(paths):
        rows.append(row)
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_prompt_id(row: Mapping[str, Any]) -> str:
    for key in ("prompt_id", "id", "item_id"):
        if row.get(key) is not None:
            return str(row[key])
    return sha256_text(str(row.get("prompt", "")))[:20]


def prompt_split_leakage(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return prompt IDs assigned to more than one split."""
    assignments: dict[str, set[str]] = {}
    for row in rows:
        prompt_id = stable_prompt_id(row)
        assignments.setdefault(prompt_id, set()).add(str(row.get("split", "unspecified")))
    return {
        prompt_id: sorted(splits)
        for prompt_id, splits in assignments.items()
        if len(splits) > 1
    }


def ensure_output_dir(path: str | Path, overwrite: bool) -> Path:
    target = Path(path)
    if target.exists() and any(target.iterdir()) and not overwrite:
        raise FileExistsError(
            f"output directory is not empty: {target}; pass --overwrite true to replace files"
        )
    target.mkdir(parents=True, exist_ok=True)
    return target
