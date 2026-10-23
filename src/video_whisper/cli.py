"""Interfaz de línea de comandos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .transcriber import transcribe_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribir-video",
        description="Transcribe todo el audio de un video a un archivo TXT usando Whisper local.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("video", type=Path, help="Ruta al archivo de video")
    parser.add_argument("-o", "--output", type=Path, help="Archivo TXT de salida; por defecto, junto al video")
    parser.add_argument("--model", default="base", help="Modelo Whisper: tiny, base, small, medium, large, turbo, etc.")
    parser.add_argument("--language", help="Código del idioma (es, en, pt, etc.); por defecto se detecta")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="Dispositivo de procesamiento")
    parser.add_argument("--model-dir", type=Path, help="Carpeta donde descargar y buscar los modelos")
    parser.add_argument("--overwrite", action="store_true", help="Reemplazar una transcripción existente")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = transcribe_video(
            args.video,
            output=args.output,
            model_name=args.model,
            language=args.language,
            device=args.device,
            model_dir=args.model_dir,
            overwrite=args.overwrite,
        )
    except KeyboardInterrupt:
        print("\nTranscripción cancelada.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Transcripción guardada en: {output}")
    return 0
