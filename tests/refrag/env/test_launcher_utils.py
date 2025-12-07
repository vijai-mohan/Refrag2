import os

import pytest

from refrag.framework.envs import launcher_utils


@pytest.mark.parametrize(
    "rel_root, rel_cmd, rel_expected",
    [
        # No changes when there are no absolute paths
        (".", ["python", "run.py", "--foo", "bar"], ["python", "run.py", "--foo", "bar"]),
        # --output=/abs/path under project root becomes relative
        (
            ".",
            ["python", "run.py", "--output={root}/outputs/run"],
            ["python", "run.py", "--output=outputs/run"],
        ),
        # --output /abs/path pair becomes relative
        (
            ".",
            ["python", "run.py", "--output", "{root}/outputs/run"],
            ["python", "run.py", "--output", "outputs/run"],
        ),
        # Bare absolute path token under project_root becomes relative
        (
            ".",
            ["python", "run.py", "{root}/results/out"],
            ["python", "run.py", "results/out"],
        ),
        # Absolute path outside project_root is left untouched
        (
            ".",
            ["python", "run.py", "--output=/other/dir/out"],
            ["python", "run.py", "--output=/other/dir/out"],
        ),
    ],
)
def test_make_paths_relative_unix(rel_root, rel_cmd, rel_expected, tmp_path, monkeypatch):
    # Use a real temporary directory as project_root so chdir always succeeds
    project_root = tmp_path / rel_root
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_root)

    # Expand {root} placeholders to the absolute project_root string
    root_str = str(project_root)

    def _expand(seq):
        expanded = []
        for t in seq:
            if isinstance(t, str):
                expanded.append(t.format(root=root_str))
            else:
                expanded.append(t)
        return expanded

    cmd = _expand(rel_cmd)
    expected = _expand(rel_expected)

    result = launcher_utils.make_paths_relative(cmd)
    assert result == expected


@pytest.mark.parametrize(
    "root_suffix, cmd_tmpl, expected_tmpl",
    [
        # Absolute path under project root becomes relative (OS-specific separators)
        (
            "workspace",
            ["python", "run.py", "--output={root}/outputs/run"],
            ["python", "run.py", "--output=outputs/run"],
        ),
        # Different drive/sample absolute path should be left unchanged
        (
            "workspace",
            ["python", "run.py", "--output=/mnt/other/out"],
            ["python", "run.py", "--output=/mnt/other/out"],
        ),
    ],
)
def test_make_paths_relative_windows_like(root_suffix, cmd_tmpl, expected_tmpl, tmp_path, monkeypatch):
    # Simulate a project_root on the current filesystem, but test Windows-like behavior
    project_root = tmp_path / root_suffix
    project_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(project_root)

    root_str = str(project_root)
    cmd = [t.format(root=root_str) if isinstance(t, str) else t for t in cmd_tmpl]
    expected = [t.format(root=root_str) if isinstance(t, str) else t for t in expected_tmpl]

    result = launcher_utils.make_paths_relative(cmd, project_root=root_str)
    norm_result = [os.path.normpath(t) if isinstance(t, str) else t for t in result]
    norm_expected = [os.path.normpath(t) if isinstance(t, str) else t for t in expected]
    assert norm_result == norm_expected
