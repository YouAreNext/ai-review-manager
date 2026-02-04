# src/ai_review/review/parser.py
from dataclasses import dataclass, field
from unidiff import PatchSet


@dataclass
class DiffFile:
    path: str
    diff: str
    is_new: bool
    is_deleted: bool
    added_lines: list[int] = field(default_factory=list)


def parse_diff(diff_text: str) -> list[DiffFile]:
    """Parse unified diff and extract file information."""
    patch = PatchSet(diff_text)
    files = []

    for patched_file in patch:
        added_lines = []

        for hunk in patched_file:
            for line in hunk:
                if line.is_added and line.target_line_no is not None:
                    added_lines.append(line.target_line_no)

        files.append(DiffFile(
            path=patched_file.path,
            diff=str(patched_file),
            is_new=patched_file.is_added_file,
            is_deleted=patched_file.is_removed_file,
            added_lines=added_lines,
        ))

    return files
