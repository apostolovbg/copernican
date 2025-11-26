"""User-facing CLI menus for the Copernican Suite."""

from __future__ import annotations

import os
import random
import time
from typing import Iterable

from copernican_lib import console_output as console
from copernican_lib import utils


def show_splash_screen(version: str) -> None:
    """Display the startup banner once at launch."""

    banner = [
        "=" * 70,
        "",
        "C O P E R N I C A N   S U I T E".center(70),
        "",
        "=" * 70,
        "",
        "A tool for rapid development, prototyping and testing of".center(70),
        (
            "alternative cosmological frameworks against observational data"
        ).center(70),
        "-" * 70,
        f"build {version}".center(70),
        "=" * 70,
        "",
    ]
    for line in banner:
        console.write(line)
    # Pause briefly so the banner is readable before the menu clears the
    # scrollback with prompts and progress updates.
    time.sleep(1)
    console.write("")


def select_seed() -> int:
    """Prompt the operator to choose a reproducible random seed."""

    console.write("")
    console.write("Random Seed Selection")
    console.write("---------------------")
    console.write(
        "This seed initialises every random number generator used by "
        "Copernican so runs can be repeated exactly."
    )
    console.write("")

    env_seed = os.environ.get("COPERNICAN_SEED")
    if env_seed is not None:
        try:
            seed = int(env_seed)
        except ValueError:
            console.write(
                "COPERNICAN_SEED is not an integer; falling back to the menu.",
                error=True,
            )
        else:
            console.write(f"Using environment-provided seed: {seed}")
            utils.set_random_seed(seed)
            return seed

    console.write("Please choose how to seed the sampler:")
    console.write("  1) Accept the default seed (0)")
    console.write("  2) Enter a custom integer seed")
    console.write("  3) Generate a random seed (uniform in [0, 2^32 - 1])")
    console.write("")

    while True:
        choice = console.ask("Select an option: ").strip().lower()
        if choice in {"1", "", "default"}:
            seed = 0
            console.write("Default seed 0 selected.")
            break
        if choice in {"2", "custom"}:
            while True:
                entry = console.ask("Enter integer seed: ").strip()
                try:
                    seed = int(entry)
                    console.write(f"Custom seed {seed} selected.")
                    break
                except ValueError:
                    console.write(
                        "Seeds must be whole numbers. Please try again.",
                        error=True,
                    )
            break
        if choice in {"3", "random"}:
            seed = random.randint(0, 2**32 - 1)
            console.write(f"Generated random seed {seed}.")
            break
        console.write("Please choose 1, 2 or 3.", error=True)

    utils.set_random_seed(seed)
    return seed


def select_from_list(options, prompt):
    """Display ``options`` and return the item chosen by the user."""

    if not options:
        return None
    header = prompt.replace("Select ", "").strip()
    if not header.endswith("s"):
        header += "s"
    console.write("")
    console.write(header)
    console.write("-" * len(header))
    for i, opt in enumerate(options, start=1):
        console.write(f"  {i}. {opt}")
    console.write(
        "Write the number of your preferred choice or 'c' to cancel:"
    )
    while True:
        choice = console.ask("> ").strip()
        if choice.lower() == "c":
            return None
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        console.write("Invalid selection. Try again.")


def normalise_failure_reasons(details: Iterable[str] | str) -> list[str]:
    """Return a list of human-readable reasons extracted from ``details``."""

    if isinstance(details, str):
        text = details.split(":", 1)[-1] if ":" in details else details
    raw_parts = text.replace(";", os.linesep).splitlines()
    else:
        raw_parts = []
        for item in details:
            raw_parts.extend(str(item).splitlines())

    reasons: list[str] = []
    for part in raw_parts:
        cleaned = part.strip()
        if cleaned:
            reasons.append(cleaned)
    return reasons or ["An unspecified error occurred during model setup."]


def prompt_stage1_retry(reasons: Iterable[str]) -> bool:
    """Return ``True`` to restart Stage 1, ``False`` to exit the workflow."""

    console.write("")
    console.write("Stage 1 cannot continue because:")
    for entry in reasons:
        console.write(f"  - {entry}")
    console.write("")
    console.write("How would you like to proceed?")
    console.write("  1) Restart Stage 1 configuration from the beginning")
    console.write("  C) Exit the Copernican Suite")
    console.write("")

    while True:
        decision = console.ask("Select an option: ").strip().lower()
        if decision in {"", "1", "restart"}:
            console.write("")
            console.write("Restarting Stage 1 configuration.")
            return True
        if decision in {"c", "cancel", "exit"}:
            return False
        console.write("Please choose 1 to restart or C to exit.", error=True)
