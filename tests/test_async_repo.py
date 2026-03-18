"""T4: Async tests for repo.py (startup path).

Covers:
- copy() methods are coroutines (parametrized)
- LocalRepoConfig.copy() awaits runtime.upload + .execute
- GithubRepoConfig.copy() awaits runtime.execute
- No asyncio.run() calls remain in source
"""
from __future__ import annotations

import asyncio
import inspect
import subprocess
from unittest.mock import AsyncMock, MagicMock

import pytest

from sweagent.environment.repo import GithubRepoConfig, LocalRepoConfig


# ---------------------------------------------------------------------------
# Parametrized coroutine check
# ---------------------------------------------------------------------------
_REPO_COPY_CLASSES = [
    ("LocalRepoConfig", LocalRepoConfig),
    ("GithubRepoConfig", GithubRepoConfig),
]


class TestRepoCopyMethodsAreCoroutines:
    @pytest.mark.parametrize("name,cls", _REPO_COPY_CLASSES)
    def test_copy_is_coroutine(self, name, cls):
        assert asyncio.iscoroutinefunction(cls.copy), (
            f"{name}.copy should be async def"
        )


# ---------------------------------------------------------------------------
# Behavior tests
# ---------------------------------------------------------------------------
class TestAsyncLocalRepoCopy:
    @pytest.mark.asyncio
    async def test_copy_awaits_upload_and_execute(self, tmp_path):
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        subprocess.run(["git", "init", str(repo_path)], check=True, capture_output=True)
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "user.email", "test@test.com"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "user.name", "Test"],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "commit", "--allow-empty", "-m", "init"],
            check=True, capture_output=True,
        )

        config = LocalRepoConfig(path=repo_path, base_commit="HEAD")
        mock_dep = MagicMock()
        mock_dep.runtime.upload = AsyncMock()
        mock_dep.runtime.execute = AsyncMock(
            return_value=MagicMock(exit_code=0, stdout="", stderr="")
        )

        await config.copy(mock_dep)
        mock_dep.runtime.upload.assert_awaited_once()
        mock_dep.runtime.execute.assert_awaited_once()


class TestAsyncGithubRepoCopy:
    @pytest.mark.asyncio
    async def test_copy_awaits_execute(self):
        config = GithubRepoConfig(
            github_url="https://github.com/test/repo",
            base_commit="abc123",
        )
        mock_dep = MagicMock()
        mock_dep.runtime.execute = AsyncMock(return_value=MagicMock(exit_code=0))

        await config.copy(mock_dep)
        mock_dep.runtime.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Source-level
# ---------------------------------------------------------------------------
class TestNoAsyncioRunInRepo:
    def test_no_asyncio_run_in_source(self):
        from sweagent.environment import repo

        source = inspect.getsource(repo)
        assert "asyncio.run(" not in source
