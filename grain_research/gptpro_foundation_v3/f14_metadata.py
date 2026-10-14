#!/usr/bin/env python3
"""Read explicitly selected metadata and audit observation support without text.

Author/role identifies a source, not the subject or factual status of a sentence.
Calendar time, within-document order and experimental occasion remain distinct.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

FORBIDDEN = {
    "text",
    "body",
    "author_flair_text",
    "answer",
    "response",
    "content",
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "mbti",
    "score",
    "percentile",
    "email",
    "name",
}


@dataclass(frozen=True)
class MetadataSpec:
    """An explicit, versionable field projection, not automatic data discovery."""

    dataset: str
    author: str
    event: tuple[str, ...]
    condition: tuple[str, ...] = ()
    time: str | None = None
    time_kind: str = "absent"
    language: str | None = None
    role: str | None = None
    default_role: str = "author"
    extras: tuple[str, ...] = ()
    order: str | None = None
    grain: str = "record"

    def columns(self) -> list[str]:
        """Return only the requested metadata columns, rejecting content/labels."""
        fields = [self.author, *self.event, *self.condition, *self.extras]
        fields += [x for x in (self.time, self.language, self.role, self.order) if x]
        columns = list(dict.fromkeys(fields))
        if any(x.casefold() in FORBIDDEN for x in columns):
            raise ValueError("content/questionnaire field requested as metadata")
        if self.time_kind not in {"absent", "unix_seconds", "iso8601_tz"}:
            raise ValueError("undeclared time unit")
        if bool(self.time) != (self.time_kind != "absent"):
            raise ValueError("time field/unit mismatch")
        return columns

    def contract(self) -> dict:
        """Expose field names only, with explicit missingness and target limits."""
        return asdict(self) | {
            "projection": self.columns(),
            "null_time": "unknown; not epoch zero or last-record imputation",
            "source_role": "not semantic self-qualification",
            "opportunity": "not observed by these columns",
        }


def sha256_file(path: Path) -> str:
    """Hash bytes incrementally; no values are interpreted or exported."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def inspect_schema(path: Path, allowed_columns: list[str]) -> dict:
    """Read a CSV header or Parquet footer; never deserialize text rows here."""
    path = Path(path)
    if any(x.casefold() in FORBIDDEN for x in allowed_columns):
        raise ValueError("content/questionnaire field requested as metadata")
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle))
        types = None
    elif path.suffix == ".parquet":
        import pyarrow.parquet as pq

        schema = pq.read_schema(path)
        columns = schema.names
        types = {field.name: str(field.type) for field in schema}
    else:
        raise ValueError("only explicit CSV/Parquet schema inspection is supported")
    if len(set(columns)) != len(columns):
        raise ValueError("duplicate source column names")
    return {
        "path": str(path),
        "columns": columns,
        "types": types,
        "selected_columns": allowed_columns,
        "missing_columns": sorted(set(allowed_columns) - set(columns)),
    }


def read_projected(path: Path, spec: MetadataSpec, chunksize: int = 100_000):
    """Yield metadata-only frames; label/text columns are never materialized."""
    columns = spec.columns()
    schema = inspect_schema(path, columns)
    if schema["missing_columns"]:
        raise ValueError(f"missing metadata columns: {schema['missing_columns']}")
    if Path(path).suffix == ".csv":
        yield from pd.read_csv(
            path,
            usecols=columns,
            dtype="string",
            encoding="utf-8-sig",
            keep_default_na=False,
            chunksize=chunksize,
        )
    else:
        import pyarrow.parquet as pq

        for batch in pq.ParquetFile(path).iter_batches(
            columns=columns,
            batch_size=chunksize,
        ):
            yield batch.to_pandas().astype("string")


def _clean(series: pd.Series) -> pd.Series:
    """Preserve literal IDs such as NA while treating blank fields as missing."""
    out = series.astype("string").str.strip()
    return out.mask(out.eq(""))


def composite(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """Construct collision-free compound keys; a missing part stays missing."""
    if not columns:
        return pd.Series(pd.NA, index=frame.index, dtype="string")
    cleaned = frame[list(columns)].apply(_clean)
    if len(columns) == 1:
        return cleaned.iloc[:, 0]
    result = (
        cleaned.fillna("")
        .apply(
            lambda row: json.dumps(row.tolist(), ensure_ascii=True),
            axis=1,
        )
        .astype("string")
    )
    return result.mask(cleaned.isna().any(axis=1))


def normalize_metadata(frame: pd.DataFrame, spec: MetadataSpec) -> pd.DataFrame:
    """Map one declared schema into event metadata without inferring semantics."""
    missing = set(spec.columns()) - set(frame.columns)
    if missing:
        raise ValueError(f"missing metadata columns: {sorted(missing)}")
    out = pd.DataFrame(index=frame.index)
    out["author"] = _clean(frame[spec.author])
    out["event"] = composite(frame, spec.event)
    out["condition"] = composite(frame, spec.condition)
    out["language"] = _clean(frame[spec.language]) if spec.language else pd.NA
    out["role"] = _clean(frame[spec.role]) if spec.role else spec.default_role
    out["order"] = _clean(frame[spec.order]) if spec.order else pd.NA
    if spec.time:
        raw = _clean(frame[spec.time])
        if spec.time_kind == "unix_seconds":
            numeric = pd.to_numeric(raw, errors="coerce")
            # These corpora are modern. Explicit seconds, not guessed ms/ns.
            allowed = numeric.between(946684800, 4102444800, inclusive="left")
            parsed = pd.to_datetime(numeric.where(allowed), unit="s", utc=True)
        else:
            allowed = raw.str.contains(
                r"^\d{4}-\d\d-\d\d[T ].*(?:Z|[+-]\d\d:\d\d)$",
                na=False,
            )
            parsed = pd.to_datetime(
                raw.where(allowed), format="mixed", utc=True, errors="coerce"
            )
        out["time"] = parsed.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        out["day"] = parsed.dt.strftime("%Y-%m-%d")
        out["time_invalid"] = (raw.notna() & parsed.isna()).astype(int)
        out["time_missing"] = raw.isna().astype(int)
    else:
        out["time"] = pd.NA
        out["day"] = pd.NA
        out["time_invalid"] = 0
        out["time_missing"] = 1
    return out.astype(object).where(out.notna(), None)


def audit_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    keys: list[str],
    *,
    require: str | None = None,
    complete_left: bool = False,
    agree: tuple[str, ...] = (),
) -> dict:
    """Audit joins before expansion; reject null keys and contextual conflicts."""
    if left[keys].isna().any().any() or right[keys].isna().any().any():
        raise ValueError("null join key")
    lc, rc = (
        left.groupby(keys, dropna=False).size(),
        right.groupby(keys, dropna=False).size(),
    )
    lu, ru = bool(lc.le(1).all()), bool(rc.le(1).all())
    cardinality = ("one" if lu else "many") + "_to_" + ("one" if ru else "many")
    if require is not None and cardinality != require:
        raise ValueError(f"join cardinality {cardinality}, expected {require}")
    intersection = lc.index.intersection(rc.index)
    result = {
        "left_rows": len(left),
        "right_rows": len(right),
        "left_unique_keys": len(lc),
        "right_unique_keys": len(rc),
        "intersection_keys": len(intersection),
        "cardinality": cardinality,
        "matched_left_rows": int(lc.reindex(intersection).sum()),
        "joined_rows": int((lc.reindex(intersection) * rc.reindex(intersection)).sum()),
        "left_unmatched_keys": len(lc.index.difference(rc.index)),
        "right_unmatched_keys": len(rc.index.difference(lc.index)),
    }
    if complete_left and result["left_unmatched_keys"]:
        raise ValueError("unmatched left join keys")
    if agree:
        if not ru:
            raise ValueError("context agreement requires unique right keys")
        matched = left.merge(right, on=keys, how="inner", suffixes=("_left", "_right"))
        for col in agree:
            if not matched[col + "_left"].equals(matched[col + "_right"]):
                raise ValueError(f"join context mismatch: {col}")
    return result


class MetadataProfile:
    """Bounded-RAM metadata inventory; SQLite lives in a temporary directory."""

    columns = (
        "author",
        "event",
        "condition",
        "language",
        "role",
        "order",
        "time",
        "day",
        "time_invalid",
        "time_missing",
    )

    def __init__(self, path: Path):
        self.db = sqlite3.connect(path)
        self.db.execute("PRAGMA journal_mode=OFF")
        fields = ",".join(
            f'"{name}" {"INTEGER" if name.startswith("time_") else "TEXT"}'
            for name in self.columns
        )
        self.db.execute(f"CREATE TABLE records ({fields})")

    def add(self, frame: pd.DataFrame) -> None:
        """Retain duplicates for explicit auditing, never silently count anew."""
        clean = (
            frame[list(self.columns)]
            .astype(object)
            .where(frame[list(self.columns)].notna(), None)
        )
        self.db.executemany(
            "INSERT INTO records VALUES (?,?,?,?,?,?,?,?,?,?)",
            clean.itertuples(index=False, name=None),
        )
        self.db.commit()

    def _scalar(self, query: str) -> int:
        return int(self.db.execute(query).fetchone()[0] or 0)

    def authors(self) -> pd.DataFrame:
        """Return IDs internally for joins; callers must not export identifiers."""
        return pd.read_sql_query(
            "SELECT DISTINCT author FROM records WHERE author IS NOT NULL", self.db
        )

    def availability_report(self) -> dict:
        """Compute actual row/event/time support, explicitly before deduplication."""
        scalar = self._scalar
        self.db.execute("CREATE INDEX IF NOT EXISTS record_event ON records(event)")
        self.db.execute("CREATE INDEX IF NOT EXISTS record_author ON records(author)")
        result = {
            "rows": scalar("SELECT COUNT(*) FROM records"),
            "authors": scalar("SELECT COUNT(DISTINCT author) FROM records"),
            "events": scalar("SELECT COUNT(DISTINCT event) FROM records"),
            "missing_author_rows": scalar(
                "SELECT COUNT(*) FROM records WHERE author IS NULL"
            ),
            "missing_event_rows": scalar(
                "SELECT COUNT(*) FROM records WHERE event IS NULL"
            ),
            "missing_condition_rows": scalar(
                'SELECT COUNT(*) FROM records WHERE "condition" IS NULL'
            ),
            "conditions": scalar('SELECT COUNT(DISTINCT "condition") FROM records'),
            "calendar_rows": scalar(
                "SELECT COUNT(*) FROM records WHERE time IS NOT NULL"
            ),
            "invalid_time_rows": scalar("SELECT SUM(time_invalid) FROM records"),
            "missing_time_rows": scalar("SELECT SUM(time_missing) FROM records"),
            "authors_multiple_days": scalar(
                "SELECT COUNT(*) FROM (SELECT author FROM records WHERE author IS NOT NULL GROUP BY author HAVING COUNT(DISTINCT day)>1)"
            ),
            "authors_multiple_conditions": scalar(
                'SELECT COUNT(*) FROM (SELECT author FROM records WHERE author IS NOT NULL GROUP BY author HAVING COUNT(DISTINCT "condition")>1)'
            ),
            "authors_multiple_languages": scalar(
                "SELECT COUNT(*) FROM (SELECT author FROM records WHERE author IS NOT NULL GROUP BY author HAVING COUNT(DISTINCT language)>1)"
            ),
            "repeated_event_keys": scalar(
                "SELECT COUNT(*) FROM (SELECT event FROM records WHERE event IS NOT NULL GROUP BY event HAVING COUNT(*)>1)"
            ),
            "event_author_conflicts": scalar(
                "SELECT COUNT(*) FROM (SELECT event FROM records WHERE event IS NOT NULL GROUP BY event HAVING COUNT(DISTINCT author)>1)"
            ),
            "event_time_conflicts": scalar(
                "SELECT COUNT(*) FROM (SELECT event FROM records WHERE event IS NOT NULL GROUP BY event HAVING COUNT(DISTINCT time)>1)"
            ),
            "events_multiple_conditions": scalar(
                'SELECT COUNT(*) FROM (SELECT event FROM records WHERE event IS NOT NULL GROUP BY event HAVING COUNT(DISTINCT "condition")>1)'
            ),
            "same_author_condition_multiple_days": scalar(
                'SELECT COUNT(*) FROM (SELECT author,"condition" FROM records WHERE author IS NOT NULL AND "condition" IS NOT NULL GROUP BY author,"condition" HAVING COUNT(DISTINCT day)>1)'
            ),
            "authors_multiple_events": scalar(
                "SELECT COUNT(*) FROM (SELECT author FROM records WHERE author IS NOT NULL GROUP BY author HAVING COUNT(DISTINCT event)>1)"
            ),
            "all_opportunities_observed": False,
            "opportunity_reason": "No exposure/non-response sampling frame in this projection",
        }
        result["duplicate_event_extra_rows"] = (
            result["rows"] - result["missing_event_rows"] - result["events"]
        )
        result["language_rows"] = dict(
            self.db.execute(
                "SELECT COALESCE(language,'<missing>'),COUNT(*) FROM records GROUP BY language"
            )
        )
        result["role_rows"] = dict(
            self.db.execute(
                "SELECT COALESCE(role,'<missing>'),COUNT(*) FROM records GROUP BY role"
            )
        )
        result["calendar_range"] = list(
            self.db.execute("SELECT MIN(time),MAX(time) FROM records").fetchone()
        )
        return result

    def close(self) -> None:
        """Close before deleting the temporary metadata store."""
        self.db.close()
