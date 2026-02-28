from __future__ import annotations

from functools import lru_cache

from app.config import Settings, get_settings
from app.services.fs_indexer import FilesystemIndexer
from app.services.pipeline_runner import PipelineRunner
from app.services.security import PathSecurity


@lru_cache(maxsize=1)
def get_security() -> PathSecurity:
    return PathSecurity(get_settings())


@lru_cache(maxsize=1)
def get_indexer() -> FilesystemIndexer:
    return FilesystemIndexer(get_settings(), get_security())


@lru_cache(maxsize=1)
def get_runner() -> PipelineRunner:
    return PipelineRunner(get_settings())


def settings_dependency() -> Settings:
    return get_settings()
