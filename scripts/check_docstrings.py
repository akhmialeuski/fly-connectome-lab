"""Check reStructuredText documentation and English source text without importing code."""

import ast
import re
import sys
from pathlib import Path

SOURCE_ROOTS: tuple[str, ...] = ('src', 'tests', 'scripts')
CYRILLIC_PATTERN: re.Pattern[str] = re.compile(pattern=r'[\u0400-\u04ff]')


def violations(path: Path) -> list[str]:
    """Find missing documentation fields and non-English source characters.

    :param path: Python source file.
    :type path: Path
    :returns: Human-readable violations with file and line locations.
    :rtype: list[str]
    """
    source = path.read_text(encoding='utf-8')
    tree = ast.parse(source=source, filename=str(path))
    errors: list[str] = []
    if CYRILLIC_PATTERN.search(string=source):
        errors.append(f'{path}: non-English source text')
    for node in ast.walk(node=tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node=node) or ''
        prefix = f'{path}:{node.lineno}: {node.name}'
        if not doc:
            errors.append(f'{prefix}: missing docstring')
        if isinstance(node, ast.ClassDef):
            continue
        parameters = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        for parameter in parameters:
            if parameter.arg in {'self', 'cls'}:
                continue
            if f':param {parameter.arg}:' not in doc or f':type {parameter.arg}:' not in doc:
                errors.append(f'{prefix}: document parameter {parameter.arg} with reST fields')
        if node.returns is None:
            errors.append(f'{prefix}: missing return annotation')
        elif (not isinstance(node.returns, ast.Constant) or node.returns.value is not None) and (
            ':returns:' not in doc or ':rtype:' not in doc
        ):
            errors.append(f'{prefix}: missing return documentation')
    return errors


def main() -> int:
    """Check all project Python source files.

    :returns: Zero on success, one on documentation violations.
    :rtype: int
    """
    errors = [
        error
        for root in SOURCE_ROOTS
        for path in sorted(Path(root).rglob(pattern='*.py'))
        for error in violations(path=path)
    ]
    if errors:
        sys.stderr.write('\n'.join(errors) + '\n')
    return int(bool(errors))


if __name__ == '__main__':
    raise SystemExit(main())
