from __future__ import annotations

from moladt.chem.pretty import pretty_text
from moladt.examples import DIBORANE_MANUSCRIPT, FERROCENE_MANUSCRIPT, MORPHINE_MANUSCRIPT
from moladt.examples import diborane_pretty, ferrocene_pretty, morphine_pretty


def test_ferrocene_pretty_rendering_includes_shells_and_backdonation() -> None:
    rendered = pretty_text(ferrocene_pretty)
    assert "Molecule with 21 atoms, 20 sigma bonds, 3 bonding systems" in rendered
    assert "electron shells:" in rendered
    assert "System 3 [fe_backdonation]: 6 shared electrons" in rendered
    assert "Fe #1" in rendered


def test_diborane_pretty_rendering_includes_3c2e_bridges() -> None:
    rendered = pretty_text(diborane_pretty)
    assert "Molecule with 8 atoms, 5 sigma bonds, 2 bonding systems" in rendered
    assert "System 1 [bridge_h3_3c2e]: 2 shared electrons" in rendered
    assert "System 2 [bridge_h4_3c2e]: 2 shared electrons" in rendered


def test_morphine_pretty_rendering_includes_explicit_ring_and_pi_systems() -> None:
    rendered = pretty_text(morphine_pretty)
    assert "Molecule with 21 atoms, 25 sigma bonds, 2 bonding systems" in rendered
    assert "System 1 [alkene_bridge]: 2 shared electrons" in rendered
    assert "System 2 [phenyl_pi_ring]: 6 shared electrons" in rendered


def test_manuscript_examples_render_titles_and_notes() -> None:
    ferrocene_text = FERROCENE_MANUSCRIPT.render()
    diborane_text = DIBORANE_MANUSCRIPT.render()
    morphine_text = MORPHINE_MANUSCRIPT.render()
    assert "Ferrocene (Fe(C5H5)2)" in ferrocene_text
    assert "back-donation-style pool" in ferrocene_text
    assert "Diborane (B2H6)" in diborane_text
    assert "3c-2e bridging hydrogen bonding systems" in diborane_text
    assert "Morphine (explicit Dietz skeleton)" in morphine_text
    assert "five classic SMILES ring closures" in morphine_text
