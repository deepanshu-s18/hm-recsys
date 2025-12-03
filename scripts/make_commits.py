#!/usr/bin/env python3
"""Utility to verify git commit history metadata for this project."""

import subprocess
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CommitInfo:
    hash: str
    date: str
    message: str


def get_commits() -> list[CommitInfo]:
    """Return all commits in chronological order."""
    result = subprocess.run(
        ["git", "log", "--format=%H|%ad|%s", "--date=short", "--reverse"],
        capture_output=True,
        text=True,
        check=True,
    )
    commits = []
    for line in result.stdout.strip().splitlines():
        parts = line.split("|", 2)
        if len(parts) == 3:
            commits.append(CommitInfo(hash=parts[0][:8], date=parts[1], message=parts[2]))
    return commits


def main() -> None:
    commits = get_commits()
    print(f"\n{'Hash':<10} {'Date':<12} {'Message'}")
    print("-" * 80)
    for c in commits:
        print(f"{c.hash:<10} {c.date:<12} {c.message[:58]}")
    print(f"\nTotal commits: {len(commits)}")


if __name__ == "__main__":
    main()
