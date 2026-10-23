"""Integración con Whisper y escritura de la transcripción."""

from __future__ import annotations

from importlib import import_module
import os
from pathlib import Path
import shutil
import sys
import tempfile


def transcribe_video(
    video: Path,
    *,
    output: Path | None = None,
    model_name: str = "base",
    language: str | None = None,
    device: str = "auto",
    model_dir: Path | None = None,
    overwrite: bool = False,
) -> Path:
    """Transcribe el audio completo y devuelve la ruta del TXT generado.

    Whisper decodifica el video mediante FFmpeg. No se recorta el audio ni
    se crea un archivo intermedio. Las dependencias se cargan bajo demanda.
    """
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise ValueError(f"El archivo de entrada no existe o no es un archivo: {video}")
    output = Path(output).expanduser().resolve() if output else video.with_suffix(".txt")
    if output == video or (output.exists() and output.samefile(video)):
        raise ValueError("La salida no puede ser el mismo archivo que la entrada.")
    if output.exists():
        if not output.is_file():
            raise ValueError(f"La ruta de salida no es un archivo: {output}")
        if not overwrite:
            raise FileExistsError(f"La salida ya existe: {output}. Usá --overwrite para reemplazarla.")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("No se encontró FFmpeg. Instalalo y agregalo al PATH (macOS: brew install ffmpeg).")
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("El dispositivo debe ser auto, cpu o cuda.")

    try:
        whisper = import_module("whisper")
        torch = import_module("torch")
        tokenizer = import_module("whisper.tokenizer")
    except ImportError as exc:
        raise RuntimeError("No se pudieron cargar Whisper y sus dependencias. Ejecutá: python -m pip install -e .") from exc

    if model_name not in whisper.available_models():
        raise ValueError(f"Modelo desconocido: {model_name}. Disponibles: {', '.join(whisper.available_models())}")
    if language is not None:
        language = language.lower()
        if language not in tokenizer.LANGUAGES:
            raise ValueError(f"Código de idioma no admitido: {language}. Por ejemplo: es, en, pt.")
        if model_name.endswith(".en") and language != "en":
            raise ValueError("Los modelos .en solo admiten inglés. Elegí un modelo multilingüe.")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible. Usá --device cpu o --device auto.")

    output.parent.mkdir(parents=True, exist_ok=True)
    options = {"device": device}
    if model_dir is not None:
        options["download_root"] = str(Path(model_dir).expanduser())
    print(f"Cargando modelo {model_name} en {device} (se descarga si es necesario)…", file=sys.stderr)
    model = whisper.load_model(model_name, **options)
    print(f"Transcribiendo: {video.name}", file=sys.stderr)
    result = model.transcribe(
        str(video), language=language, task="transcribe", fp16=device == "cuda", verbose=False
    )
    text = result["text"].strip()
    _write_text(output, text + "\n" if text else "", overwrite=overwrite)
    if not text:
        print("No se detectó voz; se generó una transcripción vacía.", file=sys.stderr)
    return output


def _write_text(output: Path, text: str, *, overwrite: bool) -> None:
    # Publicar solo un archivo completo; si falla, conservar la salida anterior.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output.parent, prefix=f".{output.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        if overwrite:
            os.replace(temporary, output)
        else:
            # El enlace falla si otro proceso creó el destino durante la transcripción.
            os.link(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
