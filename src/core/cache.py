"""Identity-guarded JSONL cache: the single source of truth for a run.

The cache is an append-only JSONL file. Its first line is an identity header
(the run-identity tuple + its hash); every subsequent line is one problem's full
outcome (including the math judge verdict). Resume compares the current run's
identity against the header and refuses to merge into a cache from a different
run. The results file is produced purely by aggregating the cache, so the two
can never diverge.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

from .identity import compute_run_identity, diff_identity, identity_hash
from .schemas import SCHEMA_VERSION, BenchmarkConfig, EvaluationResult, ExperimentSummary
from .scoring import summarize

logger = logging.getLogger(__name__)

_HEADER_KEY = "__ockbench_cache__"


class CacheIdentityMismatch(ValueError):
    """Raised when a cache file belongs to a different run identity."""


def _read_nonempty_lines(path: Path) -> List[str]:
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _dedupe_latest(results: List[EvaluationResult]) -> List[EvaluationResult]:
    """One record per problem: the latest appended attempt wins.

    Records are appended in attempt order, and resume only re-attempts problems
    whose last cached outcome was an error, so taking the latest line per
    problem_id naturally lets a later success supersede an earlier error.
    """
    by_id = {}
    for r in results:
        by_id[r.problem_id] = r
    return list(by_id.values())


def _parse_results(lines: List[str]) -> List[EvaluationResult]:
    results: List[EvaluationResult] = []
    for line_num, line in enumerate(lines, 1):
        try:
            results.append(EvaluationResult(**json.loads(line)))
        except Exception as e:
            logger.warning(f"Cache result line {line_num}: failed to parse, skipping: {e}")
    return results


def aggregate_cache_file(cache_path: str, duration: float = 0.0) -> Tuple[List[EvaluationResult], ExperimentSummary]:
    """Aggregate a cache file into (deduped results, summary) — config-free.

    The results file is built from exactly this, so regenerating from the cache
    always reproduces the written results.
    """
    path = Path(cache_path)
    lines = _read_nonempty_lines(path)
    body = lines[1:] if lines and _is_header(lines[0]) else lines
    results = _dedupe_latest(_parse_results(body))
    return results, summarize(results, duration)


def _is_header(line: str) -> bool:
    try:
        return bool(json.loads(line).get(_HEADER_KEY))
    except Exception:
        return False


class RunCache:
    """An open, identity-verified cache for one run."""

    def __init__(self, path: str, identity: dict, ident_hash: str):
        self.path = Path(path)
        self.identity = identity
        self.identity_hash = ident_hash
        self._completed_ids: Set = set()
        self._results: List[EvaluationResult] = []
        self._rejudgable_results: Dict[Any, EvaluationResult] = {}

    @classmethod
    def open(cls, cache_path: str, config: BenchmarkConfig) -> "RunCache":
        """Open (or create) a cache for ``config``, verifying its identity.

        Raises ``CacheIdentityMismatch`` if an existing cache belongs to a run
        whose identity differs, naming exactly what changed.
        """
        identity = compute_run_identity(config)
        ident_hash = identity_hash(identity)
        inst = cls(cache_path, identity, ident_hash)
        path = Path(cache_path)

        if path.exists() and path.stat().st_size > 0:
            lines = _read_nonempty_lines(path)
            if not lines or not _is_header(lines[0]):
                raise CacheIdentityMismatch(
                    f"cache file '{cache_path}' has no identity header (incompatible/old format); "
                    "use a fresh cache path"
                )
            header = json.loads(lines[0])
            if header.get("identity_hash") != ident_hash:
                changed = diff_identity(header.get("identity", {}), identity)
                raise CacheIdentityMismatch(
                    f"cache '{cache_path}' belongs to a different run; refusing to resume into it. "
                    f"Changed identity components: {json.dumps(changed, ensure_ascii=False, default=str)}. "
                    "Use a different --cache path for the new configuration."
                )
            inst._load(lines[1:])
            logger.info(f"Resuming: {len(inst._completed_ids)} completed problems in cache")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            header = {
                _HEADER_KEY: True,
                "schema_version": SCHEMA_VERSION,
                "identity": identity,
                "identity_hash": ident_hash,
            }
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(header, ensure_ascii=False, default=str) + "\n")

        return inst

    def _load(self, body_lines: List[str]) -> None:
        latest_by_id: Dict[Any, EvaluationResult] = {}
        for r in _parse_results(body_lines):
            latest_by_id[r.problem_id] = r

        for r in latest_by_id.values():
            if not r.error and not r.evaluator_error:
                self._completed_ids.add(r.problem_id)
                self._results.append(r)
            elif self._is_rejudgable_error(r):
                self._rejudgable_results[r.problem_id] = r

    @staticmethod
    def _is_rejudgable_error(result: EvaluationResult) -> bool:
        """True when generation succeeded and only the evaluator/judge failed.

        Anything else — a generation error, or no cached response text to
        re-score — falls through to full regeneration.
        """
        if not result.model_response:
            return False
        if result.evaluator_error and not result.error:
            return True
        # Legacy rows (written before evaluator_error existed) carry judge
        # failures in the top-level error field; extraction_method is what
        # distinguishes them from generation failures there.
        return bool(result.error) and result.extraction_method not in {"error", "exception"}

    @property
    def completed_ids(self) -> Set:
        return self._completed_ids

    @property
    def rejudgable_results(self) -> Dict[Any, EvaluationResult]:
        return self._rejudgable_results

    def append(self, result: EvaluationResult) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(result.model_dump_json() + "\n")
            f.flush()

    def aggregate(self, duration: float) -> Tuple[List[EvaluationResult], ExperimentSummary]:
        """Aggregate this cache's contents into (results, summary)."""
        return aggregate_cache_file(str(self.path), duration)
