from __future__ import annotations
import os
from typing import Sequence, List, Optional


def build_host_cmd(argv: Sequence[str], python_bin: str = "python3") -> List[str]:
    """
    Build the host command list: [python_bin] + argv, and normalize script path
    to basename when the second argument is a .py file.
    """
    host_cmd: List[str] = [python_bin] + list(argv)
    if len(host_cmd) > 1 and isinstance(host_cmd[1], str) and host_cmd[1].endswith(".py"):
        host_cmd[1] = os.path.basename(host_cmd[1])
    return host_cmd


def make_paths_relative(cmd: List[str], project_root: Optional[str] = None) -> List[str]:
    """Rewrite absolute output-related paths in a command to be relative to project root.

    This is shared between launchers so that any paths pointing under the
    repository root become portable (e.g., inside containers or remote envs).
    """
    if project_root is None:
        project_root = os.getcwd()

    flag_keys = {
        "output",
        "output-dir",
        "out",
        "out_dir",
        "output_dir",
        "results",
        "results-dir",
        "outdir",
    }

    new_cmd = list(cmd)
    i = 0
    while i < len(new_cmd):
        tok = new_cmd[i]

        # Handle --flag=/abs/path or --flag=rel
        if isinstance(tok, str) and "=" in tok:
            left, right = tok.split("=", 1)
            left_clean = left.lstrip("-")
            if left_clean in flag_keys and os.path.isabs(right):
                try:
                    norm = os.path.normpath(right)
                    if os.path.commonpath([project_root, norm]) == project_root:
                        rel = os.path.relpath(norm, project_root)
                        new_cmd[i] = f"{left}={rel}"
                except ValueError:
                    # Different drives or invalid common path; leave untouched
                    pass

        # Handle "--flag" "/abs/path" pairs
        elif (
            isinstance(tok, str)
            and tok.lstrip("-") in flag_keys
            and i + 1 < len(new_cmd)
        ):
            nxt = new_cmd[i + 1]
            if isinstance(nxt, str) and os.path.isabs(nxt):
                try:
                    norm = os.path.normpath(nxt)
                    if os.path.commonpath([project_root, norm]) == project_root:
                        new_cmd[i + 1] = os.path.relpath(norm, project_root)
                except ValueError:
                    pass
                i += 1

        # Generic absolute path token that lives under project_root
        elif isinstance(tok, str) and os.path.isabs(tok):
            try:
                norm = os.path.normpath(tok)
                if os.path.commonpath([project_root, norm]) == project_root:
                    new_cmd[i] = os.path.relpath(norm, project_root)
            except ValueError:
                pass

        i += 1

    return new_cmd
