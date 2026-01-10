from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional
from urllib.request import urlopen


LOG_TIME_RE = re.compile(
    r"\\[(?:INFO|WARNING|ERROR)\\] (\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2},\\d{3})"
)


@dataclass
class PairSnapshot:
    address: str
    trace_id: Optional[str] = None
    first_seen: Optional[str] = None
    recheck_pass: Optional[str] = None
    recheck_score: Optional[str] = None
    refresh_pass: Optional[str] = None
    refresh_score: Optional[str] = None
    refresh_removed: Optional[str] = None
    auto_refresh_enabled: Optional[str] = None


def _extract_timestamp(line: str) -> Optional[str]:
    match = LOG_TIME_RE.search(line)
    if not match:
        return None
    return match.group(1)


def _first_match(lines: Iterable[str], needle: str) -> Optional[str]:
    for line in lines:
        if needle.lower() in line.lower():
            return line
    return None


def _find_trace_id(lines: Iterable[str]) -> Optional[str]:
    for line in lines:
        if "trace_id=" in line:
            match = re.search(r"trace_id=([0-9a-f]+)", line)
            if match:
                return match.group(1)
    return None


def _format_score(line: Optional[str]) -> Optional[str]:
    if not line:
        return None
    match = re.search(r"=>\\s+[^\\s]+\\s+=>\\s+(\\d+/\\d+)\\s+passes", line)
    return match.group(1) if match else None


def _collect_pair_snapshot(log_lines: list[str], address: str) -> PairSnapshot:
    addr_lower = address.lower()
    pair_lines = [line for line in log_lines if addr_lower in line.lower()]
    snapshot = PairSnapshot(address=address, trace_id=_find_trace_id(pair_lines))

    first_seen = _first_match(pair_lines, "missing DexScreener data")
    if first_seen:
        snapshot.first_seen = _extract_timestamp(first_seen)

    recheck_pass = _first_match(pair_lines, "[Recheck] =>")
    if recheck_pass:
        snapshot.recheck_pass = _extract_timestamp(recheck_pass)
        snapshot.recheck_score = _format_score(recheck_pass)

    refresh_pass = _first_match(pair_lines, "[Refresh] =>")
    if refresh_pass:
        snapshot.refresh_pass = _extract_timestamp(refresh_pass)
        snapshot.refresh_score = _format_score(refresh_pass)

    refresh_removed = _first_match(pair_lines, "removing" + " " + address)
    if refresh_removed and "[Refresh]" in refresh_removed:
        snapshot.refresh_removed = _extract_timestamp(refresh_removed)

    auto_refresh = _first_match(pair_lines, "AutoRefresh] Enabled")
    if auto_refresh:
        snapshot.auto_refresh_enabled = _extract_timestamp(auto_refresh)

    return snapshot


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\\\", "\\\\\\\\").replace("(", "\\\\(").replace(")", "\\\\)")


def _render_pdf(lines: list[str], output_path: Path) -> None:
    width, height = 612, 792
    font_size = 11
    line_height = 14
    start_x, start_y = 50, 750

    content_lines = []
    content_lines.append("BT")
    content_lines.append(f"/F1 {font_size} Tf")
    content_lines.append(f"{start_x} {start_y} Td")
    for idx, line in enumerate(lines):
        if idx > 0:
            content_lines.append(f"0 -{line_height} Td")
        content_lines.append(f"({_escape_pdf_text(line)}) Tj")
    content_lines.append("ET")
    content_stream = "\n".join(content_lines).encode("utf-8")

    objects = []
    objects.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append(
        f"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 {width} {height}] "
        f"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj".encode(
            "utf-8"
        )
    )
    objects.append(b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    objects.append(
        f"5 0 obj << /Length {len(content_stream)} >> stream\n".encode("utf-8")
        + content_stream
        + b"\nendstream\nendobj"
    )

    xref_positions = []
    result = bytearray()
    result.extend(b"%PDF-1.4\n")
    for obj in objects:
        xref_positions.append(len(result))
        result.extend(obj + b"\n")

    xref_start = len(result)
    result.extend(f"xref\n0 {len(objects) + 1}\n".encode("utf-8"))
    result.extend(b"0000000000 65535 f \n")
    for pos in xref_positions:
        result.extend(f"{pos:010d} 00000 n \n".encode("utf-8"))
    result.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n".encode(
            "utf-8"
        )
    )

    output_path.write_bytes(result)


def build_report(log_path: Path, output_path: Path, pairs: list[str]) -> None:
    log_lines: list[str] = []
    if log_path.exists():
        log_lines = log_path.read_text(errors="ignore").splitlines()
    else:
        try:
            with urlopen(
                "https://raw.githubusercontent.com/skenglordwarrior/Trading-Bot/main/bot.log"
            ) as resp:
                log_lines = resp.read().decode("utf-8", errors="ignore").splitlines()
        except Exception:
            log_lines = []
    snapshots = [_collect_pair_snapshot(log_lines, pair) for pair in pairs]

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        "Institutional Scan Review",
        f"Generated: {now}",
        "",
    ]
    if not log_lines:
        lines.append("Log source unavailable; timestamps may be missing.")
        lines.append("")

    for snap in snapshots:
        lines.extend(
            [
                f"Pair: {snap.address}",
                f"Trace ID: {snap.trace_id or 'unknown'}",
                f"First seen (DexScreener not listed): {snap.first_seen or 'unknown'}",
                f"Recheck pass time: {snap.recheck_pass or 'unknown'}",
                f"Recheck score: {snap.recheck_score or 'unknown'}",
                f"Auto-refresh enabled: {snap.auto_refresh_enabled or 'unknown'}",
                f"Refresh pass time: {snap.refresh_pass or 'unknown'}",
                f"Refresh score: {snap.refresh_score or 'unknown'}",
                f"Refresh removed: {snap.refresh_removed or 'not removed'}",
                "",
            ]
        )

    lines.extend(
        [
            "Key recommendations:",
            "1) Require multiple refresh passes before trade-ready flag.",
            "2) Alert immediately on refresh downgrade/removal events.",
            "3) Enforce minimum DexScreener availability window before passing.",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _render_pdf(lines, output_path)


if __name__ == "__main__":
    report_output = Path("reports/institutional_scan_report.pdf")
    build_report(
        Path("bot.log"),
        report_output,
        [
            "0x542DB2d78047E362FA7Fc94bA744B2264275845A",
            "0x41cE04Ec71059CE20316db0BdDF84669A2Dd9e9A",
        ],
    )
