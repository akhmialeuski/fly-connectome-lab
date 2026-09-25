"""Schema-tolerant discovery and bounded evidence access for arbitrary research attempts."""

import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from flystate.viewer.files import document

MAX_TEXT_BYTES: int = 2 * 1024 * 1024
MAX_PAGE_ROWS: int = 200
MAX_ROW_GROUP_BYTES: int = 16 * 1024 * 1024
TABLE_BATCH_ROWS: int = 256
JSON_SUFFIX: str = '.json'
LOG_SUFFIX: str = '.log'
MARKDOWN_SUFFIX: str = '.md'
PARQUET_SUFFIX: str = '.parquet'
CHECKSUM_SUFFIX: str = '.sha256'
YAML_SUFFIX: str = '.yaml'
MESSAGE_SEPARATOR: str = ': '
NOT_A_FILE: str = 'Requested evidence is not a regular file.'
CLASSES: str = 'classes'
CREATED_UTC: str = 'created_utc'
DOCUMENTS: str = 'documents'
DEFAULT_KIND: str = 'experiment'
FILES: str = 'files'
GATE: str = 'gate'
ID: str = 'id'
KIND: str = 'kind'
MANIFEST_FILE: str = 'manifest.json'
MODE: str = 'mode'
NAME: str = 'name'
PARAMETERS: str = 'parameters'
REPORT_FILE: str = 'report.json'
REVISION: str = 'revision'
SCORES: str = 'scores'
STATUS: str = 'status'
ENCODING: str = 'utf-8'
WARNINGS: str = 'warnings'
MAX_DECISIONS: int = 50
DECISION: str = 'decision'
SELECTED_ALPHA: str = 'selected_alpha'
DECISION_FIELDS: tuple[str, ...] = (NAME, 'difference_pp', 'interval_95_pp', DECISION)
DETAIL_DOCUMENTS: tuple[str, ...] = (
    MANIFEST_FILE,
    'config.json',
    REPORT_FILE,
    'summary.json',
    'provenance.json',
    'model/model.json',
    'feature-statistics.json',
    'training-subset.json',
)


def nested_resource(root: Path, relative: str) -> Path:
    """Resolve a relative artifact path without permitting traversal or escaping symlinks.

    :param root: Allowed existing storage boundary.
    :type root: Path
    :param relative: Slash-separated relative path, including optional spaces.
    :type relative: str
    :returns: Existing resolved descendant.
    :rtype: Path
    :raises ValueError: If the path is absolute, malformed, or escapes the boundary.
    :raises FileNotFoundError: If the requested resource does not exist.
    """
    if (
        '\\' in relative
        or '\x00' in relative
        or any(p in {'', '.', '..'} for p in relative.split('/'))
    ):
        raise ValueError('Invalid relative artifact path.')
    target = (root / relative).resolve()
    if not target.is_relative_to(root.resolve()):
        raise ValueError('Artifact escapes its storage directory.')
    if not target.exists():
        raise FileNotFoundError('Requested artifact is unavailable.')
    return target


def bounded_document(path: Path) -> dict[str, Any]:
    """Read finite JSON metadata within the interactive response size limit.

    :param path: Existing JSON metadata file.
    :type path: Path
    :returns: Parsed metadata object.
    :rtype: dict[str, Any]
    :raises ValueError: If the file is too large or is not a finite JSON object.
    """
    if not path.is_file():
        raise ValueError(NOT_A_FILE)
    if path.stat().st_size > MAX_TEXT_BYTES:
        raise ValueError('Metadata exceeds the interactive preview limit.')
    return document(path=path)


def recorded_outcome(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract a report's decisions, frozen selections and photograph count, if it records any.

    Contrast decisions come from ``contrasts`` items that carry a ``decision``; selections come
    from ``families`` items that carry a ``selected_alpha``. Malformed items are skipped, so an
    unfamiliar report never breaks discovery.

    :param payload: Parsed ``report.json`` object.
    :type payload: dict[str, Any]
    :returns: Any of ``decisions`` (list), ``selection`` (family to alpha) and ``episodes`` (int).
    :rtype: dict[str, Any]
    """
    outcome: dict[str, Any] = {}
    contrasts = payload.get('contrasts')
    if isinstance(contrasts, list):
        decisions = [
            {key: item[key] for key in DECISION_FIELDS if key in item}
            for item in contrasts[:MAX_DECISIONS]
            if isinstance(item, dict) and isinstance(item.get(DECISION), str)
        ]
        if decisions:
            outcome['decisions'] = decisions
    families = payload.get('families')
    if isinstance(families, dict):
        selection = {
            str(family): value[SELECTED_ALPHA]
            for family, value in families.items()
            if isinstance(value, dict) and isinstance(value.get(SELECTED_ALPHA), int | float)
        }
        if selection:
            outcome['selection'] = selection
    episodes = payload.get('episodes')
    if isinstance(episodes, int) and not isinstance(episodes, bool):
        outcome['episodes'] = episodes
    return outcome


class ExperimentStore:
    """Discover manifest-backed experiments independently of study names and directory depth."""

    def __init__(self, root: Path) -> None:
        """Bind the working runs directory.

        :param root: Existing or not-yet-created runs directory.
        :type root: Path
        """
        self.root = root

    def discover(self, legacy_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Scan current manifests, retaining unreadable entries and legacy navigation targets.

        :param legacy_runs: Valid original training-run catalog entries.
        :type legacy_runs: list[dict[str, Any]]
        :returns: All current attempts with revisions, recorded scores, and per-entry warnings.
        :rtype: list[dict[str, Any]]
        """
        legacy = {row['run_id']: row for row in legacy_runs}
        entries = []
        for source in sorted(self.root.rglob(MANIFEST_FILE), reverse=True):
            relative = source.parent.relative_to(self.root).as_posix()
            if relative == '.':
                continue
            entry: dict[str, Any] = {
                ID: relative,
                NAME: source.parent.name,
                'study': source.parent.parent.relative_to(self.root).as_posix(),
                STATUS: 'unreadable',
                KIND: DEFAULT_KIND,
                PARAMETERS: {},
                SCORES: {},
                WARNINGS: [],
                REVISION: [],
            }
            try:
                directory = nested_resource(root=self.root, relative=relative)
                manifest = bounded_document(
                    path=nested_resource(root=directory, relative=MANIFEST_FILE)
                )
                parameters = manifest.get(PARAMETERS, {})
                if not isinstance(parameters, dict):
                    raise ValueError('Experiment parameters must be an object.')
                entry.update(
                    status=str(manifest.get(STATUS, 'unknown')),
                    parameters=parameters,
                    kind=str(parameters.get(KIND) or manifest.get(KIND) or DEFAULT_KIND),
                    created_utc=str(manifest.get(CREATED_UTC) or ''),
                    error=manifest.get('error'),
                )
                entry[REVISION] = [directory.stat().st_mtime_ns, source.stat().st_mtime_ns]
                if relative in legacy:
                    old = legacy[relative]
                    entry.update(
                        name=old[NAME],
                        kind='training',
                        legacy_run_id=relative,
                        classes=old[CLASSES],
                        mode=old[MODE],
                        scores={
                            'validation': {'accuracy': old['summary'].get('final_val_accuracy')}
                        },
                    )
                for filename in ('config.json', REPORT_FILE):
                    if not (directory / filename).exists():
                        continue
                    try:
                        path = nested_resource(root=directory, relative=filename)
                        payload = bounded_document(path=path)
                        entry[REVISION].append(path.stat().st_mtime_ns)
                        if filename == REPORT_FILE:
                            scores = payload.get(SCORES, {})
                            if not isinstance(scores, dict):
                                raise ValueError('Recorded scores must be an object.')
                            entry[SCORES] = scores
                            conclusion = payload.get('conclusion')
                            entry['result_summary'] = (
                                conclusion if isinstance(conclusion, str) else None
                            )
                            gate = payload.get(GATE)
                            if isinstance(gate, str):
                                entry[GATE] = gate
                            fits = payload.get('fits')
                            if isinstance(fits, dict):
                                entry['case_count'] = len(fits)
                            entry.update(recorded_outcome(payload=payload))
                        else:
                            entry['config_name'] = payload.get(NAME)
                            entry[CLASSES] = (
                                payload.get('dataset', {}).get('subset', {}).get('n_identities')
                            )
                            entry[MODE] = str(payload.get('memory', {}).get(MODE) or '')
                    except (OSError, ValueError, TypeError, AttributeError) as error:
                        entry[WARNINGS].append(str(error))
            except (OSError, ValueError, TypeError) as error:
                entry[WARNINGS].append(str(error))
            entries.append(entry)
        return sorted(entries, key=lambda row: (row.get(CREATED_UTC) or '', row[ID]), reverse=True)

    def directory(self, relative: str) -> Path:
        """Resolve an experiment only when its directory contains a bounded manifest path.

        :param relative: Relative experiment directory from discovery.
        :type relative: str
        :returns: Checked experiment directory.
        :rtype: Path
        :raises ValueError: If the experiment or manifest escapes storage.
        :raises FileNotFoundError: If its manifest is absent.
        """
        directory = nested_resource(root=self.root, relative=relative)
        nested_resource(root=directory, relative=MANIFEST_FILE)
        return directory

    def detail(self, relative: str) -> dict[str, Any]:
        """Read available metadata and an evidence inventory without loading model arrays.

        :param relative: Relative experiment directory.
        :type relative: str
        :returns: Available documents, file sizes, and recoverable read warnings.
        :rtype: dict[str, Any]
        """
        directory = self.directory(relative=relative)
        result: dict[str, Any] = {ID: relative, DOCUMENTS: {}, FILES: [], WARNINGS: []}
        for name in DETAIL_DOCUMENTS:
            if not (directory / name).exists():
                continue
            try:
                result[DOCUMENTS][name] = bounded_document(
                    path=nested_resource(root=directory, relative=name)
                )
            except (OSError, ValueError, TypeError) as error:
                result[WARNINGS].append(f'{name}: {error}')
        for source in sorted(directory.rglob('*')):
            if not source.is_file():
                continue
            name = source.relative_to(directory).as_posix()
            try:
                path = nested_resource(root=directory, relative=name)
                result[FILES].append(
                    {
                        NAME: name,
                        'bytes': path.stat().st_size,
                        'preview': path.suffix
                        in {
                            JSON_SUFFIX,
                            MARKDOWN_SUFFIX,
                            LOG_SUFFIX,
                            YAML_SUFFIX,
                            CHECKSUM_SUFFIX,
                            PARQUET_SUFFIX,
                        },
                    }
                )
            except (OSError, ValueError) as error:
                result[WARNINGS].append(f'{name}: {error}')
        return result

    def evidence(self, relative: str, name: str, offset: int, limit: int) -> dict[str, Any]:
        """Preview a bounded text document or stream one page from a saved Parquet table.

        :param relative: Relative experiment directory.
        :type relative: str
        :param name: Relative evidence file within that experiment.
        :type name: str
        :param offset: Nonnegative table row offset.
        :type offset: int
        :param limit: Page size from one to 200.
        :type limit: int
        :returns: File content or paginated table records.
        :rtype: dict[str, Any]
        :raises ValueError: If paging, extension, or text size is unsupported.
        """
        if offset < 0 or not 1 <= limit <= MAX_PAGE_ROWS:
            raise ValueError('Invalid evidence page bounds.')
        path = nested_resource(root=self.directory(relative=relative), relative=name)
        if not path.is_file():
            raise ValueError(NOT_A_FILE)
        if path.suffix == PARQUET_SUFFIX:
            table = pq.ParquetFile(source=path)
            if any(
                table.metadata.row_group(i).total_byte_size > MAX_ROW_GROUP_BYTES
                for i in range(table.metadata.num_row_groups)
            ):
                raise ValueError('Table row groups exceed the interactive preview limit.')
            rows: list[dict[str, Any]] = []
            skipped = 0
            for batch in table.iter_batches(batch_size=TABLE_BATCH_ROWS):
                if skipped + batch.num_rows <= offset:
                    skipped += batch.num_rows
                    continue
                start = max(0, offset - skipped)
                rows.extend(
                    batch.slice(
                        offset=start, length=min(limit - len(rows), batch.num_rows - start)
                    ).to_pylist()
                )
                skipped += batch.num_rows
                if len(rows) >= limit:
                    break
            if (
                len(json.dumps(obj=rows, allow_nan=False).encode(encoding=ENCODING))
                > MAX_TEXT_BYTES
            ):
                raise ValueError('Table page exceeds the interactive preview limit.')
            return {
                NAME: name,
                'rows': rows,
                'total': table.metadata.num_rows,
                'offset': offset,
                'limit': limit,
                'columns': table.schema_arrow.names,
            }
        if path.suffix not in {
            JSON_SUFFIX,
            MARKDOWN_SUFFIX,
            LOG_SUFFIX,
            YAML_SUFFIX,
            CHECKSUM_SUFFIX,
        }:
            raise ValueError('This evidence type has no interactive preview.')
        if path.stat().st_size > MAX_TEXT_BYTES:
            raise ValueError('Text exceeds the interactive preview limit.')
        return {NAME: name, 'text': path.read_text(encoding=ENCODING)}
