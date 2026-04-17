from __future__ import annotations

from pathlib import Path

from moladt.chem.pretty import pretty_text
from moladt.examples import DIBORANE_MANUSCRIPT, FERROCENE_MANUSCRIPT, MORPHINE_MANUSCRIPT
from moladt.examples import diborane_pretty, ferrocene_pretty, morphine_pretty
from moladt.io.sdf import read_sdf


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_ferrocene_pretty_rendering_includes_shells_and_backdonation() -> None:
    rendered = pretty_text(ferrocene_pretty)
    assert "Molecule Report" in rendered
    assert "bonding systems  3" in rendered
    assert "Bonding Systems" in rendered
    assert "[#3] fe_backdonation" in rendered
    assert "[Fe#1]" in rendered
    assert "shells:" in rendered


def test_diborane_pretty_rendering_includes_3c2e_bridges() -> None:
    rendered = pretty_text(diborane_pretty)
    assert "atoms            8" in rendered
    assert "Sigma Network" in rendered
    assert "[#1] bridge_h3_3c2e" in rendered
    assert "[#2] bridge_h4_3c2e" in rendered
    assert "edge bonus:       +0.50 to each listed edge" in rendered


def test_morphine_pretty_rendering_includes_explicit_ring_and_pi_systems() -> None:
    rendered = pretty_text(morphine_pretty)
    assert "atoms            21" in rendered
    assert "[#1] alkene_bridge" in rendered
    assert "[#2] phenyl_pi_ring" in rendered
    assert "SMILES Stereochemistry" in rendered
    assert "center #3: TH2 from token @@" in rendered


def test_benzene_pretty_rendering_omits_empty_stereochemistry_section() -> None:
    rendered = pretty_text(read_sdf(PROJECT_ROOT / "molecules" / "benzene.sdf"))

    assert "SMILES Stereochemistry" not in rendered


def test_manuscript_examples_render_titles_and_notes() -> None:
    ferrocene_text = FERROCENE_MANUSCRIPT.render()
    diborane_text = DIBORANE_MANUSCRIPT.render()
    morphine_text = MORPHINE_MANUSCRIPT.render()
    assert "Ferrocene (Fe(C5H5)2)" in ferrocene_text
    assert "back-donation-style pool" in ferrocene_text
    assert "Diborane (B2H6)" in diborane_text
    assert "3c-2e bridging hydrogen bonding systems" in diborane_text
    assert "Morphine (explicit Dietz skeleton)" in morphine_text
    assert "atom-centered stereochemistry flags" in morphine_text
