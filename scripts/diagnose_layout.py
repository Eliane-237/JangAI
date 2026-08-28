"""
JangAI — Diagnostic de layout, page par page.

Script d'analyse EXPLORATOIRE. Il ne produit aucune donnee pour la base :
il produit un rapport JSON qui sert a decider de l'architecture du pipeline
d'extraction (branche OCR ou non, strategie par matiere, frontieres de
sous-documents).

Usage :
    python scripts/diagnose_layout.py
    python scripts/diagnose_layout.py data/raw/Programmes_SVT_Tles_LS.pdf
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import fitz  # PyMuPDF

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/diagnostics")

# Une page avec moins de caracteres que ce seuil est consideree comme
# n'ayant pas de couche texte exploitable.
MIN_CHARS_TEXTE = 60

# Part de la surface de page couverte par des tableaux au-dela de laquelle
# on considere la page comme "tableau" plutot que "mixte".
SEUIL_SURFACE_TABLEAU = 0.35

# Pagination interne ("Page 88 sur 156") : revele les concatenations
# de plusieurs sous-programmes dans un meme fichier.
RE_PAGINATION = re.compile(r"Page\s+(\d+)\s+sur\s+(\d+)", re.IGNORECASE)

# Titres de section de haut niveau, potentielles frontieres de sous-documents.
RE_TITRE_PROGRAMME = re.compile(
    r"PROGRAMME\s+D[EÉ]TAILL[EÉ]E?.{0,60}?"
    r"(?:CLASSE\s+DE\s+)?TERMINALE\s*[«\"'\u201c]?\s*([A-Z0-9]{1,4})",
    re.IGNORECASE | re.DOTALL,
)

# Mentions de serie isolees.
RE_SERIE = re.compile(r"[«\"'\u201c]\s*(S1|S2A|S2|S3|L1|L2|L1a|LA|G)\s*[»\"'\u201d]")

# Pied de page identifiant le programme (cas Maths).
RE_PIED_PROGRAMME = re.compile(
    r"Programmes?\s+de\s+([a-zàâäéèêëîïôöùûüç]+).{0,40}?S[eé]ries?\s+([A-Z0-9]+)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------
# Analyse d'une page
# ----------------------------------------------------------------------


def analyser_page(page: fitz.Page) -> dict:
    """Retourne les indicateurs bruts d'une page, sans interpretation."""

    texte = page.get_text("text")
    nb_chars = len(texte.strip())

    largeur, hauteur = page.rect.width, page.rect.height
    surface_page = largeur * hauteur

    # --- Tableaux -----------------------------------------------------
    nb_tableaux = 0
    colonnes = []
    lignes = []
    surface_tableaux = 0.0
    tableau_touche_bas = False
    tableau_touche_haut = False

    try:
        tables = page.find_tables()
        for t in tables:
            nb_tableaux += 1
            colonnes.append(t.col_count)
            lignes.append(t.row_count)
            x0, y0, x1, y1 = t.bbox
            surface_tableaux += abs((x1 - x0) * (y1 - y0))
            # Un tableau qui atteint le bas ou le haut de la zone utile
            # signale une ligne potentiellement coupee entre deux pages.
            if y1 > hauteur * 0.88:
                tableau_touche_bas = True
            if y0 < hauteur * 0.12:
                tableau_touche_haut = True
    except Exception as exc:  # find_tables peut echouer sur pages exotiques
        print(f"    [!] find_tables a echoue p.{page.number + 1} : {exc}")

    ratio_tableau = surface_tableaux / surface_page if surface_page else 0.0

    # --- Images -------------------------------------------------------
    images = page.get_images(full=True)
    surface_images = 0.0
    for img in images:
        for rect in page.get_image_rects(img[0]):
            surface_images += rect.width * rect.height
    ratio_image = surface_images / surface_page if surface_page else 0.0

    # --- Polices ------------------------------------------------------
    polices = page.get_fonts(full=True)
    noms_polices = sorted({f[3] for f in polices})
    # f[1] == "" signale une police non embarquee
    polices_non_embarquees = [f[3] for f in polices if not f[1]]

    # --- Vectoriel ----------------------------------------------------
    nb_traces = len(page.get_drawings())

    # --- Classification ----------------------------------------------
    if nb_chars < MIN_CHARS_TEXTE:
        layout = "scanne" if ratio_image > 0.5 else "vide"
    elif nb_tableaux and ratio_tableau >= SEUIL_SURFACE_TABLEAU:
        layout = "tableau"
    elif nb_tableaux:
        layout = "mixte"
    else:
        layout = "prose"

    # --- Marqueurs structurels ---------------------------------------
    marqueurs = {}

    if m := RE_PAGINATION.search(texte):
        marqueurs["pagination_interne"] = {
            "page": int(m.group(1)),
            "total": int(m.group(2)),
        }
    if m := RE_TITRE_PROGRAMME.search(texte):
        marqueurs["titre_programme"] = m.group(0)[:120].replace("\n", " ")
        marqueurs["serie_detectee"] = m.group(1)
    if "serie_detectee" not in marqueurs and (m := RE_SERIE.search(texte)):
        marqueurs["serie_detectee"] = m.group(1)
    if m := RE_PIED_PROGRAMME.search(texte):
        marqueurs["pied_programme"] = {
            "matiere": m.group(1),
            "serie": m.group(2),
        }

    return {
        "page": page.number + 1,
        "layout": layout,
        "orientation": "paysage" if largeur > hauteur else "portrait",
        "dimensions": [round(largeur), round(hauteur)],
        "nb_chars": nb_chars,
        "nb_tableaux": nb_tableaux,
        "colonnes": colonnes,
        "lignes": lignes,
        "ratio_tableau": round(ratio_tableau, 3),
        "tableau_touche_bas": tableau_touche_bas,
        "tableau_touche_haut": tableau_touche_haut,
        "nb_images": len(images),
        "ratio_image": round(ratio_image, 3),
        "nb_traces_vectorielles": nb_traces,
        "polices": noms_polices,
        "polices_non_embarquees": sorted(set(polices_non_embarquees)),
        "marqueurs": marqueurs,
        "extrait": " ".join(texte.split())[:200],
    }


# ----------------------------------------------------------------------
# Analyse d'un fichier
# ----------------------------------------------------------------------


def analyser_fichier(chemin: Path) -> dict:
    doc = fitz.open(chemin)
    pages = [analyser_page(p) for p in doc]

    layouts = Counter(p["layout"] for p in pages)
    orientations = Counter(p["orientation"] for p in pages)

    # Repartition des schemas de colonnes (revele les schemas heterogenes)
    schemas_colonnes = Counter()
    for p in pages:
        for c in p["colonnes"]:
            schemas_colonnes[c] += 1

    # Lignes de tableau potentiellement coupees entre deux pages :
    # page N se termine par un tableau ET page N+1 commence par un tableau.
    debordements = [
        pages[i]["page"]
        for i in range(len(pages) - 1)
        if pages[i]["tableau_touche_bas"] and pages[i + 1]["tableau_touche_haut"]
    ]

    # Ruptures de pagination interne : revele les concatenations.
    paginations = [
        (p["page"], p["marqueurs"]["pagination_interne"])
        for p in pages
        if "pagination_interne" in p["marqueurs"]
    ]
    ruptures = []
    for i in range(1, len(paginations)):
        prec = paginations[i - 1][1]
        cour = paginations[i][1]
        if cour["total"] != prec["total"] or cour["page"] < prec["page"]:
            ruptures.append(
                {
                    "page_pdf": paginations[i][0],
                    "avant": prec,
                    "apres": cour,
                }
            )

    series = sorted(
        {
            p["marqueurs"]["serie_detectee"]
            for p in pages
            if "serie_detectee" in p["marqueurs"]
        }
    )

    frontieres = [
        {
            "page_pdf": p["page"],
            "titre": p["marqueurs"].get("titre_programme"),
            "serie": p["marqueurs"].get("serie_detectee"),
        }
        for p in pages
        if "titre_programme" in p["marqueurs"]
    ]

    pages_sans_texte = [
        p["page"] for p in pages if p["layout"] in ("scanne", "vide")
    ]
    polices_non_emb = sorted(
        {f for p in pages for f in p["polices_non_embarquees"]}
    )

    return {
        "fichier": chemin.name,
        "nb_pages": len(pages),
        "resume": {
            "layouts": dict(layouts),
            "orientations": dict(orientations),
            "schemas_colonnes": dict(sorted(schemas_colonnes.items())),
            "ocr_requis": len(pages_sans_texte) > 0,
            "pages_sans_texte": pages_sans_texte,
            "polices_non_embarquees": polices_non_emb,
            "nb_debordements_tableau": len(debordements),
            "pages_debordement": debordements[:40],
            "series_detectees": series,
            "frontieres_sous_documents": frontieres,
            "ruptures_pagination": ruptures,
        },
        "pages": pages,
    }


# ----------------------------------------------------------------------
# Affichage
# ----------------------------------------------------------------------


def afficher(rapport: dict) -> None:
    r = rapport["resume"]
    print(f"\n{'=' * 68}")
    print(f"  {rapport['fichier']}  —  {rapport['nb_pages']} pages")
    print(f"{'=' * 68}")

    print("  Layouts        :", dict(r["layouts"]))
    print("  Orientation    :", dict(r["orientations"]))
    print("  Colonnes       :", r["schemas_colonnes"] or "aucun tableau")

    if r["ocr_requis"]:
        print(f"  /!\\ OCR REQUIS : {len(r['pages_sans_texte'])} page(s) sans texte")
        print(f"      pages      : {r['pages_sans_texte'][:20]}")
    else:
        print("  OCR            : non requis (couche texte presente partout)")

    if r["polices_non_embarquees"]:
        print(f"  /!\\ Polices non embarquees : {r['polices_non_embarquees'][:5]}")

    print(f"  Debordements   : {r['nb_debordements_tableau']} ligne(s) coupee(s)")

    if r["series_detectees"]:
        print("  Series         :", r["series_detectees"])
    if r["frontieres_sous_documents"]:
        print(f"  Sous-documents : {len(r['frontieres_sous_documents'])}")
        for f in r["frontieres_sous_documents"][:10]:
            print(f"      p.{f['page_pdf']:>4}  serie={f['serie']}")
    if r["ruptures_pagination"]:
        print(f"  Ruptures pagin.: {len(r['ruptures_pagination'])}")
        for rup in r["ruptures_pagination"][:10]:
            a, b = rup["avant"], rup["apres"]
            print(
                f"      p.{rup['page_pdf']:>4}  "
                f"{a['page']}/{a['total']} -> {b['page']}/{b['total']}"
            )


# ----------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if len(sys.argv) > 1:
        fichiers = [Path(a) for a in sys.argv[1:]]
    else:
        if not RAW_DIR.exists():
            print(f"Dossier introuvable : {RAW_DIR.resolve()}")
            return 1
        fichiers = sorted(RAW_DIR.glob("*.pdf"))

    if not fichiers:
        print(f"Aucun PDF dans {RAW_DIR.resolve()}")
        return 1

    global_resume = []

    for chemin in fichiers:
        if not chemin.exists():
            print(f"Introuvable : {chemin}")
            continue
        try:
            rapport = analyser_fichier(chemin)
        except Exception as exc:
            print(f"\n[ECHEC] {chemin.name} : {exc}")
            continue

        afficher(rapport)

        sortie = OUT_DIR / f"{chemin.stem}.layout.json"
        sortie.write_text(
            json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        global_resume.append(
            {"fichier": chemin.name, "nb_pages": rapport["nb_pages"], **rapport["resume"]}
        )

    (OUT_DIR / "_synthese.json").write_text(
        json.dumps(global_resume, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n{'=' * 68}")
    print(f"  Rapports ecrits dans {OUT_DIR.resolve()}")
    print(f"{'=' * 68}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())