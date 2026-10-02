"""
Transcription vocale (speech-to-text) locale via faster-whisper.

Role unique : audio -> texte. Le texte repart ensuite vers l'agent `/chat` ; la
voix n'est qu'une PORTE D'ENTREE, elle ne modifie pas le pipeline de reponse.
"""

from __future__ import annotations

import io
import os

from loguru import logger

from app.config import get_settings

# Instance unique du modele, gardee chaude pour le process. On n'utilise pas
# lru_cache car on peut avoir besoin de RECONSTRUIRE le modele (repli CPU).
_MODEL = None


def _register_cuda_dlls() -> None:
    """Rend visibles les DLL CUDA 12 (cuBLAS/cuDNN) fournies par pip (Windows).

    CTranslate2 (moteur de faster-whisper) cherche `cublas64_12.dll` et les
    libs cuDNN. Les wheels `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` les
    installent dans site-packages/nvidia/*/bin ; on ajoute ces dossiers au
    chemin de recherche des DLL pour qu'elles soient trouvees.
    """
    if os.name != "nt":
        return
    try:
        import nvidia

        # `nvidia` est un namespace package : pas de __file__, on lit __path__.
        base = list(nvidia.__path__)[0]
    except Exception:
        return
    for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
        path = os.path.join(base, sub, "bin")
        if os.path.isdir(path):
            try:
                os.add_dll_directory(path)
            except Exception:  # pragma: no cover
                pass
            # Certains chargeurs natifs lisent PATH plutot que les dossiers DLL
            # ajoutes : on l'alimente aussi.
            if path not in os.environ.get("PATH", ""):
                os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _build_model(force_cpu: bool = False):
    """Construit le modele faster-whisper (sur GPU, ou CPU si force/sans GPU)."""
    from faster_whisper import WhisperModel

    settings = get_settings()
    _register_cuda_dlls()

    device = "cpu" if force_cpu else (settings.stt_device or "auto")
    if device == "auto":
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # pragma: no cover
            device = "cpu"

    compute_type = settings.stt_compute_type
    if device == "cpu" and compute_type.startswith("float16"):
        # float16 n'est pas supporte sur CPU : on retombe sur int8.
        compute_type = "int8"

    logger.info(
        "Chargement du modele STT {} sur {} ({})",
        settings.stt_model, device, compute_type,
    )
    model = WhisperModel(settings.stt_model, device=device, compute_type=compute_type)
    logger.info("Modele STT charge.")
    return model


def get_model():
    """Instance unique du modele (chargee a la demande, gardee chaude)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _build_model()
    return _MODEL


def warmup() -> None:
    """Charge le modele ET chauffe le GPU (kernels/cuBLAS) sur du silence.

    Sans ce passage a vide, la toute premiere transcription paie encore
    l'initialisation CUDA. On decode 1 s de silence pour tout amorcer.
    """
    try:
        import numpy as np

        model = get_model()
        segments, _ = model.transcribe(
            np.zeros(16000, dtype="float32"),
            language=get_settings().stt_language or None,
        )
        list(segments)  # force l'execution (le generateur est paresseux)
        logger.info("Modele STT prechauffe.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Prechauffage STT incomplet ({}) : 1re requete plus lente.", exc)


def _run(model, audio: str | bytes) -> dict:
    """Execute la transcription et assemble le resultat."""
    settings = get_settings()
    source: object = io.BytesIO(audio) if isinstance(audio, bytes) else audio
    segments, info = model.transcribe(
        source,
        language=settings.stt_language or None,
        vad_filter=settings.stt_vad_filter,
    )
    # `segments` est un generateur paresseux : la transcription s'execute en le
    # parcourant.
    text = "".join(segment.text for segment in segments).strip()
    logger.info(
        "Transcription : {} caracteres | langue {} | {:.1f}s d'audio",
        len(text), info.language, info.duration,
    )
    return {"text": text, "language": info.language, "duration": info.duration}


def transcribe(audio: str | bytes) -> dict:
    """Transcrit un audio en texte.

    Args:
        audio: Chemin d'un fichier OU contenu binaire (faster-whisper decode via
            PyAV integre : wav, mp3, m4a, ogg... sans ffmpeg externe).

    Returns:
        Dictionnaire {text, language, duration}.
    """
    try:
        return _run(get_model(), audio)
    except RuntimeError as exc:
        message = str(exc).lower()
        if any(k in message for k in ("cuda", "cublas", "cudnn", "cu12")):
            # GPU indisponible a l'execution malgre le chargement : on reconstruit le modele sur CPU et on reessaie une fois, pour ne jamais echouer faute de GPU.
            logger.warning("STT GPU indisponible ({}) : repli sur CPU.", exc)
            global _MODEL
            _MODEL = _build_model(force_cpu=True)
            return _run(_MODEL, audio)
        raise
