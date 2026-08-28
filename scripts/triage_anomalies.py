"""
JangAI — Triage des anomalies detectees par diagnose_layout.py

Repond a trois questions bloquantes :
  1. Les pages "vide" de l'Anglais : calques OCG, vectoriel, ou reellement vides ?
  2. SVT : quelle est la sequence complete de pagination interne ?
  3. Philosophie : la rotation est-elle declaree (/Rotate) ou reelle ?

Usage :
    python scripts/triage_anomalies.py
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/diagnostics")

ANGLAIS = "Programme_Anglais_Tle.pdf"
SVT = "Programmes_SVT_Tles_LS.pdf"
PHILO = "Programme_Philosophie_Tle.pdf"


def section(titre: str) -> None:
    print(f"\n{'=' * 68}\n  {titre}\n{'=' * 68}")


# ----------------------------------------------------------------------
# 1. Pages "vide" : ou est passe le contenu ?
# ----------------------------------------------------------------------


def triage_pages_vides(nom: str, pages_cibles: list[int]) -> None:
    section(f"{nom} — pages sans texte")

    chemin = RAW_DIR / nom
    if not chemin.exists():
        print(f"  Introuvable : {chemin}")
        return

    doc = pymupdf.open(chemin)

    # Calques OCG au niveau du document
    try:
        ocgs = doc.get_ocgs()
        print(f"  Calques OCG declares : {len(ocgs)}")
        for xref, info in list(ocgs.items())[:10]:
            print(f"      xref={xref}  nom={info.get('name')!r}  on={info.get('on')}")
        if ocgs:
            print("  -> contenu potentiellement masque par configuration de calques")
    except Exception as exc:
        print(f"  get_ocgs a echoue : {exc}")

    for num in pages_cibles:
        page = doc[num - 1]
        texte_brut = page.get_text("text")
        # rawdict expose les spans meme quand get_text() ne renvoie rien
        raw = page.get_text("rawdict")
        nb_spans = sum(
            len(l.get("spans", []))
            for b in raw.get("blocks", [])
            for l in b.get("lines", [])
        )
        traces = page.get_drawings()
        images = page.get_images(full=True)
        annots = list(page.annots()) if page.annots() else []
        xobjects = page.get_xobjects()

        print(
            f"\n  p.{num:>3} | rotation={page.rotation:>3} "
            f"| chars={len(texte_brut.strip()):>5} | spans={nb_spans:>4}"
        )
        print(
            f"         | traces_vect={len(traces):>4} | images={len(images):>3} "
            f"| annotations={len(annots):>3} | xobjects={len(xobjects):>3}"
        )

        if nb_spans and not texte_brut.strip():
            print("         -> spans presents mais get_text() vide : encodage casse")
        elif len(traces) > 50:
            print("         -> contenu vectoriel : texte converti en courbes, OCR requis")
        elif images:
            print("         -> images presentes : page scannee, OCR requis")
        elif xobjects:
            print("         -> XObjects presents : contenu imbrique non rendu")
        else:
            print("         -> page reellement vide")

    # Rendu de controle : verifier visuellement si la page est blanche
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if pages_cibles:
        num = pages_cibles[0]
        pix = doc[num - 1].get_pixmap(dpi=110)
        sortie = OUT_DIR / f"{Path(nom).stem}_p{num}.png"
        pix.save(sortie)
        print(f"\n  Rendu de controle : {sortie}")
        print("  -> ouvre cette image. Blanche = contenu inaccessible.")
        print("     Lisible = probleme d'extraction, pas de contenu.")

    doc.close()


# ----------------------------------------------------------------------
# 2. SVT : sequence complete de pagination interne
# ----------------------------------------------------------------------


def sequence_pagination(nom: str) -> None:
    section(f"{nom} — sequence de pagination interne")

    import re

    RE_PAG = re.compile(r"Page\s+(\d+)\s+sur\s+(\d+)", re.IGNORECASE)

    chemin = RAW_DIR / nom
    if not chemin.exists():
        print(f"  Introuvable : {chemin}")
        return

    doc = pymupdf.open(chemin)
    lignes = []
    for page in doc:
        m = RE_PAG.search(page.get_text("text"))
        if m:
            lignes.append((page.number + 1, int(m.group(1)), int(m.group(2))))
        else:
            lignes.append((page.number + 1, None, None))

    couverture = sum(1 for _, p, _ in lignes if p is not None)
    print(f"  Pagination lisible sur {couverture}/{len(lignes)} pages\n")

    for pdf_p, interne, total in lignes:
        if interne is None:
            print(f"    p.{pdf_p:>3} | -")
        else:
            print(f"    p.{pdf_p:>3} | {interne:>3} / {total}")

    totaux = sorted({t for _, _, t in lignes if t})
    print(f"\n  Sous-documents distincts (par total de pages) : {totaux}")
    for t in totaux:
        pages = [p for p, _, tt in lignes if tt == t]
        print(f"      total={t:>4} -> {len(pages)} pages PDF : {pages}")

    doc.close()


# ----------------------------------------------------------------------
# 3. Philosophie : rotation declaree ou dimensions reelles ?
# ----------------------------------------------------------------------


def triage_rotation(nom: str) -> None:
    section(f"{nom} — rotation des pages")

    chemin = RAW_DIR / nom
    if not chemin.exists():
        print(f"  Introuvable : {chemin}")
        return

    doc = pymupdf.open(chemin)
    from collections import Counter

    rotations = Counter()
    for page in doc:
        r = page.rect
        mb = page.mediabox
        rotations[
            (
                page.rotation,
                "paysage" if r.width > r.height else "portrait",
                "paysage" if mb.width > mb.height else "portrait",
            )
        ] += 1

    print("  (rotation, orientation rect, orientation mediabox) -> nb pages")
    for cle, n in rotations.most_common():
        print(f"      {cle} -> {n}")

    if any(rot != 0 for rot, _, _ in rotations):
        print("\n  -> rotation declaree : PyMuPDF applique deja la transformation.")
        print("     L'extraction respectera l'ordre de lecture.")
    else:
        print("\n  -> aucune rotation declaree : pages nativement en paysage.")

    doc.close()


def main() -> int:
    triage_pages_vides(ANGLAIS, [2, 3, 6])
    triage_pages_vides(PHILO, [14, 21])
    sequence_pagination(SVT)
    triage_rotation(PHILO)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())