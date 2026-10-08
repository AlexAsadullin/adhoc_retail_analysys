"""Download the raw dunnhumby "The Complete Journey" CSV files."""

import hashlib
import logging
import shutil
import urllib.request
from pathlib import Path

from retail_analysis.config import Config, SourceFile, load_config

logger = logging.getLogger(__name__)

_CHUNK_BYTES = 1 << 20
# The data host rejects the default urllib user agent.
_USER_AGENT = "retail-adhoc-analysis/0.1 (+https://doi.org/10.17632/7myy93ym6k.1)"


def sha256_of(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_file(path: Path, expected_sha256: str) -> bool:
    """Check that a file exists and matches the published checksum.

    Args:
        path: Local file path.
        expected_sha256: Checksum published by the data repository.

    Returns:
        True if the file exists and its checksum matches.
    """
    return path.exists() and sha256_of(path) == expected_sha256


def download_file(source: SourceFile, target: Path) -> None:
    """Download one file and verify its checksum.

    Args:
        source: URL and checksum of the file.
        target: Destination path.

    Raises:
        RuntimeError: If the downloaded file does not match the checksum.
    """
    partial = target.with_suffix(target.suffix + ".part")
    logger.info("Downloading %s", target.name)
    request = urllib.request.Request(source.url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK_BYTES)
    if sha256_of(partial) != source.sha256:
        partial.unlink()
        raise RuntimeError(
            f"Checksum mismatch for {target.name}. The source may have changed: "
            f"check {source.url} and update config.yaml."
        )
    partial.rename(target)


def download_dataset(config: Config) -> None:
    """Download all configured raw files, skipping the ones already present.

    Args:
        config: Project configuration.
    """
    config.paths.raw_dir.mkdir(parents=True, exist_ok=True)
    for table, source in config.source.files.items():
        target = config.paths.raw_dir / f"{table}.csv"
        if is_valid_file(target, source.sha256):
            logger.info("%s already present, skipping", target.name)
            continue
        download_file(source, target)
    logger.info("Raw data is in %s", config.paths.raw_dir)


def main() -> None:
    """Entry point for `make download`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    download_dataset(load_config())


if __name__ == "__main__":
    main()
