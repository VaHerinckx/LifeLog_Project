"""
LifeLog Processing Pipeline - Logger Module

Provides a 4-level verbosity system for controlling terminal output:
- QUIET: Section headers, milestones, errors, actionable warnings, final report
- NORMAL (default): All of quiet + scope/summary info per source (what's being processed, results)
- VERBOSE: All of normal + step-by-step progress, success confirmations, all warnings, info
- DEBUG: All of verbose + DataFrame shapes, sample data, column lists

Usage:
    from src.utils.logger import log
    log.header("PROCESSING PHASE")
    log.milestone("Processed 1,200 records")
    log.error("File not found")
    log.warning("Missing dates for 3 books")
    log.normal("Need weather data for 500 location records")
    log.normal_success("Fetched 200 new weather records")
    log.progress("Loading CSV...")
    log.success("Loaded 1,000 records")
    log.info("No new data found")
    log.debug("DataFrame shape: (1000, 25)")
"""

import sys
from enum import IntEnum


class Verbosity(IntEnum):
    QUIET = 0
    NORMAL = 1
    VERBOSE = 2
    DEBUG = 3


# ANSI color codes
BOLD = '\033[1m'
RED = '\033[31m'
YELLOW = '\033[33m'
GREEN = '\033[32m'
CYAN = '\033[36m'
DIM = '\033[2m'
RESET = '\033[0m'


class Logger:
    """Centralized logger with verbosity-aware output and ANSI formatting."""

    def __init__(self):
        self._verbosity = Verbosity.NORMAL

    def set_verbosity(self, level):
        """Set the global verbosity level."""
        if isinstance(level, int):
            level = Verbosity(level)
        self._verbosity = level

    def get_verbosity(self):
        """Get the current verbosity level."""
        return self._verbosity

    # ================================================================
    # ALWAYS SHOWN (QUIET+)
    # ================================================================

    def header(self, text):
        """Section/phase header — bold with ═══ borders. Always shown."""
        print(f"\n{BOLD}{'═' * 50}{RESET}")
        print(f"{BOLD}  {text}{RESET}")
        print(f"{BOLD}{'═' * 50}{RESET}")

    def subheader(self, text):
        """Sub-section header — with ─── borders. Always shown."""
        print(f"\n{BOLD}{'─' * 50}{RESET}")
        print(f"{BOLD}  {text}{RESET}")
        print(f"{BOLD}{'─' * 50}{RESET}")

    def source_header(self, index, total, name):
        """Source processing header — [1/8] Source Name. Always shown."""
        print(f"\n{BOLD}[{index}/{total}] {name}{RESET}")

    def milestone(self, text):
        """Key achievement — one-liner with arrow. Always shown."""
        print(f"  {CYAN}→{RESET} {text}")

    def error(self, text):
        """Error message — red. Always shown."""
        print(f"  {RED}✗ {text}{RESET}")

    def warning(self, text):
        """Actionable warning — yellow. Always shown.
        Use only for things the user should know about even in quiet mode:
        missing data, failed enrichment, skipped items, etc.
        """
        print(f"  {YELLOW}⚠ {text}{RESET}")

    def summary(self, text):
        """Summary/stats line. Always shown."""
        print(f"  {text}")

    def prompt(self, text):
        """User prompt text. Always shown (needed for interaction)."""
        print(text)

    def blank(self):
        """Empty line for spacing. Always shown."""
        print()

    # ================================================================
    # NORMAL+ ONLY
    # ================================================================

    def normal(self, text):
        """Scope/summary info — what's being processed and key results. Normal+ only."""
        if self._verbosity >= Verbosity.NORMAL:
            print(f"  {text}")

    def normal_success(self, text):
        """Summary-level success confirmation. Normal+ only."""
        if self._verbosity >= Verbosity.NORMAL:
            print(f"  {GREEN}✓ {text}{RESET}")

    # ================================================================
    # VERBOSE+ ONLY
    # ================================================================

    def progress(self, text):
        """Step-by-step progress. Verbose+ only."""
        if self._verbosity >= Verbosity.VERBOSE:
            print(f"  {text}")

    def success(self, text):
        """Non-critical success confirmation. Verbose+ only."""
        if self._verbosity >= Verbosity.VERBOSE:
            print(f"  {GREEN}✓ {text}{RESET}")

    def info(self, text):
        """Neutral informational message. Verbose+ only."""
        if self._verbosity >= Verbosity.VERBOSE:
            print(f"  {DIM}ℹ {text}{RESET}")

    # ================================================================
    # DEBUG ONLY
    # ================================================================

    def debug(self, text):
        """Raw data inspection (DataFrame shapes, samples). Debug only."""
        if self._verbosity >= Verbosity.DEBUG:
            print(f"  {DIM}[DEBUG] {text}{RESET}")

    # ================================================================
    # SPECIAL: Final report table
    # ================================================================

    def report_table(self, source_results, topic_results, maintenance_results, registry, timings=None):
        """Print a compact final report table. Always shown."""
        self.header("FINAL REPORT")

        # Source results
        all_sources = source_results.get('success', []) + [
            f.get('topic', f.get('name', '?')) for f in source_results.get('failed', [])
        ]

        if all_sources:
            print(f"  {'Source':<28} {'Status':<10} {'Records':<12} {'Time':<8}")
            print(f"  {'─' * 56}")

            for source in source_results.get('success', []):
                name = registry.get(source, {}).get('name', source)[:27]
                timing = timings.get(source, '') if timings else ''
                records = timings.get(f'{source}_records', '') if timings else ''
                print(f"  {name:<28} {GREEN}✔{RESET}{'':>8} {str(records):<12} {timing}")

            for failure in source_results.get('failed', []):
                name = failure.get('name', '?')[:27]
                print(f"  {name:<28} {RED}✗{RESET}{'':>8} {'—':<12} {'—'}")

        # Topic results
        topic_all = topic_results.get('success', []) + [
            f.get('topic', f.get('name', '?')) for f in topic_results.get('failed', [])
        ]

        if topic_all:
            print(f"\n  {'Topic':<28} {'Status':<10}")
            print(f"  {'─' * 38}")

            for topic in topic_results.get('success', []):
                name = registry.get(topic, {}).get('name', topic)[:27]
                print(f"  {name:<28} {GREEN}✔{RESET}")

            for failure in topic_results.get('failed', []):
                name = failure.get('name', '?')[:27]
                print(f"  {name:<28} {RED}✗{RESET}")

        # Maintenance
        if maintenance_results.get('failed'):
            print(f"\n  {RED}✗ Website maintenance failed{RESET}")
        elif maintenance_results.get('success'):
            print(f"\n  {GREEN}✔ Website maintenance completed{RESET}")

        # Overall
        total_failed = (
            len(source_results.get('failed', []))
            + len(topic_results.get('failed', []))
            + len(maintenance_results.get('failed', []))
        )

        print(f"\n{'═' * 50}")
        if total_failed == 0:
            print(f"  {GREEN}{BOLD}All processing completed successfully!{RESET}")
        else:
            print(f"  {YELLOW}{BOLD}Completed with {total_failed} failure(s){RESET}")
        print(f"{'═' * 50}")


# Global singleton instance
log = Logger()
