"""T4: Async tests for repo.py (startup path).

These tests verify the async target API. They will FAIL on the current sync
code and PASS after the async conversion of repo.py.

Covers:
- LocalRepoConfig.copy()   : 2 asyncio.run() -> await
- GithubRepoConfig.copy()  : 1 asyncio.run() -> await
- No asyncio.run() calls remain in source
"""
from __future__ import annotations

import asyncio
import inspect
from unittest.mock import AsyncMock, MagicMock

import pytest


# ---------------------------------------------------------------------------
# LocalRepoConfig.copy
# ---------------------------------------------------------------------------
class TestAsyncLocalRepoCopy:
    @pytest.mark.asyncio
    async def test_copy_is_coroutine(self):
        from sweagent.environment.repo import LocalRepoConfig

        assert asyncio.iscoroutinefunction(LocalRepoConfig.copy)

    @pytest.mark.asyncio
    async def test_copy_awaits_upload_and_execute(self, tmp_path):
        """copy() must await deployment.runtime.upload and .execute."""
        import subprocess

        from sweagent.environment.repo import LocalRepoConfig

        # Create a minimal git repo for check_valid_repo()
        repo_path = tmp_path / "test-repo"
        repo_path.mkdir()
        subprocess.run(
            ["git", "init", str(repo_path)],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "user.email", "test@test.com"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "config", "user.name", "Test"],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(repo_path), "commit", "--allow-empty", "-m", "init"],
            check=True,
            capture_output=True,
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


# ---------------------------------------------------------------------------
# GithubRepoConfig.copy
# ---------------------------------------------------------------------------
class TestAsyncGithubRepoCopy:
    @pytest.mark.asyncio
    async def test_copy_is_coroutine(self):
        from sweagent.environment.repo import GithubRepoConfig

        assert asyncio.iscoroutinefunction(GithubRepoConfig.copy)

    @pytest.mark.asyncio
    async def test_copy_awaits_execute(self):
        from sweagent.environment.repo import GithubRepoConfig

        config = GithubRepoConfig(
            github_url="https://github.com/test/repo",
            base_commit="abc123",
        )
        mock_dep = MagicMock()
        mock_dep.runtime.execute = AsyncMock(
            return_value=MagicMock(exit_code=0)
        )

        await config.copy(mock_dep)
        mock_dep.runtime.execute.assert_awaited_once()


# ---------------------------------------------------------------------------
# Source-level
# ---------------------------------------------------------------------------
class TestNoAsyncioRunInRepo:
    def test_no_asyncio_run_in_source(self):
        from sweagent.environment import repo

        source = inspect.getsource(repo)
        assert "asyncio.run(" not in source, (
            "repo.py should use await instead of asyncio.run()"
        )
