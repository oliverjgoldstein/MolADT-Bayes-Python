from __future__ import annotations

from dataclasses import dataclass

from ..chem.molecule import Molecule
from ..chem.pretty import pretty_text
from .diborane import diborane_pretty
from .ferrocene import ferrocene_pretty
from .morphine import morphine_pretty


@dataclass(frozen=True, slots=True)
class ManuscriptExample:
    slug: str
    title: str
    note: str
    molecule: Molecule

    def render(self) -> str:
        return "\n".join([self.title, self.note, "", pretty_text(self.molecule)])

    def __str__(self) -> str:
        return self.render()


FERROCENE_MANUSCRIPT = ManuscriptExample(
    slug="ferrocene",
    title="Ferrocene (Fe(C5H5)2)",
    note="Dietz-style ADT with two cyclopentadienyl pi systems and an Fe back-donation-style pool.",
    molecule=ferrocene_pretty,
)

DIBORANE_MANUSCRIPT = ManuscriptExample(
    slug="diborane",
    title="Diborane (B2H6)",
    note="Dietz-style ADT with two explicit 3c-2e bridging hydrogen bonding systems.",
    molecule=diborane_pretty,
)

MORPHINE_MANUSCRIPT = ManuscriptExample(
    slug="morphine",
    title="Morphine (explicit Dietz skeleton)",
    note="Dietz-style ADT that turns the five classic SMILES ring closures into sigma edges, keeps the phenyl ring as an explicit pi system, and preserves the five atom-centered stereochemistry flags from the standard boundary string.",
    molecule=morphine_pretty,
)

MANUSCRIPT_EXAMPLES: dict[str, ManuscriptExample] = {
    FERROCENE_MANUSCRIPT.slug: FERROCENE_MANUSCRIPT,
    DIBORANE_MANUSCRIPT.slug: DIBORANE_MANUSCRIPT,
    MORPHINE_MANUSCRIPT.slug: MORPHINE_MANUSCRIPT,
}


def get_manuscript_example(name: str) -> ManuscriptExample:
    key = name.strip().lower()
    if key not in MANUSCRIPT_EXAMPLES:
        known = ", ".join(sorted(MANUSCRIPT_EXAMPLES))
        raise KeyError(f"Unknown manuscript example {name!r}; choose one of: {known}")
    return MANUSCRIPT_EXAMPLES[key]
