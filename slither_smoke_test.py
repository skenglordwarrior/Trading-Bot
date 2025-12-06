"""Quick smoke test to ensure Slither is available and functional.

This script reuses the existing ``run_slither_analysis`` helper to run
Slither against a trivial Solidity contract. It exits non-zero if Slither
is missing or fails to produce a numeric issue count.
"""
from __future__ import annotations

import subprocess
import sys
from typing import Optional, Union

from ethereumbotv2 import run_slither_analysis

SAMPLE_CONTRACT = """
pragma solidity 0.8.31;

contract SlitherSmoke {
    function echo(uint256 amount) external pure returns (uint256) {
        return amount;
    }
}
"""


IssueCount = Union[int, str, None]


def main() -> int:
    """Run a small Slither analysis and print the result.

    Returns a non-zero exit code if Slither is unavailable or fails to
    return a numeric issue count for the sample contract.
    """

    result = run_slither_analysis(SAMPLE_CONTRACT)
    issues: IssueCount = result.get("slitherIssues")

    if isinstance(issues, int):
        version = _slither_version()
        if version:
            print(f"Slither {version} ran successfully; detected {issues} issues in the sample contract.")
        else:
            print(
                "Slither ran successfully; detected "
                f"{issues} issues in the sample contract (version unavailable)."
            )
        return 0

    if issues == "not_installed":
        print(
            "Slither is not available on PATH. Install it with "
            "`python -m pip install --upgrade slither-analyzer` and retry."
        )
    else:
        print(f"Slither did not complete successfully (slitherIssues={issues}).")
    return 1


def _slither_version() -> Optional[str]:
    """Return the installed Slither version string if available."""

    proc = subprocess.run(
        ["slither", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        return None

    output = proc.stdout.strip()
    return output or None


if __name__ == "__main__":
    sys.exit(main())
