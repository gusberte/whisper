from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from video_whisper.cli import main
from video_whisper.transcriber import _write_text, transcribe_video


class TranscriptionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.video = self.root / "reunión con espacios.mp4"
        self.video.write_bytes(b"video de prueba")
        self.model = Mock()
        self.model.transcribe.return_value = {"text": " Hola, ¿cómo estás? Fin del video. "}
        self.whisper = SimpleNamespace(
            available_models=lambda: ["base", "tiny", "tiny.en"],
            load_model=Mock(return_value=self.model),
        )
        self.cuda = Mock(return_value=False)
        modules = {
            "whisper": self.whisper,
            "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=self.cuda)),
            "whisper.tokenizer": SimpleNamespace(LANGUAGES={"es": "spanish", "en": "english"}),
        }
        importer = patch("video_whisper.transcriber.import_module", side_effect=modules.__getitem__)
        importer.start()
        self.addCleanup(importer.stop)
        ffmpeg = patch("video_whisper.transcriber.shutil.which", return_value="/bin/ffmpeg")
        ffmpeg.start()
        self.addCleanup(ffmpeg.stop)
        stderr = redirect_stderr(io.StringIO())
        stderr.__enter__()
        self.addCleanup(stderr.__exit__, None, None, None)

    def test_full_input_is_passed_to_whisper_and_utf8_text_is_saved(self):
        output = transcribe_video(self.video)
        self.assertEqual(output, self.video.with_suffix(".txt"))
        self.assertEqual(output.read_text(encoding="utf-8"), "Hola, ¿cómo estás? Fin del video.\n")
        self.model.transcribe.assert_called_once_with(
            str(self.video), language=None, task="transcribe", fp16=False, verbose=False
        )

    def test_custom_destination_language_and_model_cache(self):
        output = self.root / "subcarpeta" / "resultado.txt"
        transcribe_video(self.video, output=output, language="ES", model_dir=self.root / "modelos")
        self.assertTrue(output.is_file())
        self.assertEqual(self.model.transcribe.call_args.kwargs["language"], "es")
        self.whisper.load_model.assert_called_once_with(
            "base", device="cpu", download_root=str(self.root / "modelos")
        )

    def test_existing_transcription_is_preserved_without_overwrite(self):
        output = self.video.with_suffix(".txt")
        output.write_text("anterior", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            transcribe_video(self.video)
        self.assertEqual(output.read_text(), "anterior")
        self.whisper.load_model.assert_not_called()

    def test_overwrite_replaces_complete_transcription(self):
        output = self.video.with_suffix(".txt")
        output.write_text("anterior", encoding="utf-8")
        transcribe_video(self.video, overwrite=True)
        self.assertIn("Fin del video.", output.read_text())

    def test_input_cannot_be_overwritten_even_via_an_alias(self):
        hardlink = self.root / "alias.txt"
        hardlink.hardlink_to(self.video)
        symlink = self.root / "enlace.txt"
        symlink.symlink_to(self.video)
        for output in [self.video, hardlink, symlink]:
            with self.subTest(output=output), self.assertRaises(ValueError):
                transcribe_video(self.video, output=output, overwrite=True)
        self.assertEqual(self.video.read_bytes(), b"video de prueba")

    def test_missing_input_and_directory_are_rejected_before_loading_model(self):
        for video in [self.root / "missing.mp4", self.root]:
            with self.subTest(video=video), self.assertRaises(ValueError):
                transcribe_video(video)
        self.whisper.load_model.assert_not_called()

    def test_missing_ffmpeg_has_actionable_error(self):
        with patch("video_whisper.transcriber.shutil.which", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "FFmpeg"):
                transcribe_video(self.video)

    def test_missing_dependencies_have_actionable_error(self):
        with patch("video_whisper.transcriber.import_module", side_effect=ImportError):
            with self.assertRaisesRegex(RuntimeError, "pip install"):
                transcribe_video(self.video)

    def test_invalid_language_and_model_fail_before_download(self):
        for options in [{"language": "invalid"}, {"model_name": "unknown"}, {"model_name": "tiny.en", "language": "es"}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                transcribe_video(self.video, **options)
        self.whisper.load_model.assert_not_called()

    def test_cuda_is_selected_automatically_with_fp16(self):
        self.cuda.return_value = True
        transcribe_video(self.video)
        self.whisper.load_model.assert_called_once_with("base", device="cuda")
        self.assertTrue(self.model.transcribe.call_args.kwargs["fp16"])

    def test_explicit_cpu_is_respected_when_cuda_is_available(self):
        self.cuda.return_value = True
        transcribe_video(self.video, device="cpu")
        self.whisper.load_model.assert_called_once_with("base", device="cpu")
        self.assertFalse(self.model.transcribe.call_args.kwargs["fp16"])

    def test_unavailable_cuda_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "CUDA"):
            transcribe_video(self.video, device="cuda")
        self.whisper.load_model.assert_not_called()

    def test_transcription_failure_preserves_previous_output(self):
        output = self.video.with_suffix(".txt")
        output.write_text("anterior", encoding="utf-8")
        self.model.transcribe.side_effect = RuntimeError("El video no contiene audio")
        with self.assertRaisesRegex(RuntimeError, "audio"):
            transcribe_video(self.video, overwrite=True)
        self.assertEqual(output.read_text(), "anterior")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), sorted([self.video.name, output.name]))

    def test_concurrent_destination_creation_is_not_overwritten(self):
        output = self.video.with_suffix(".txt")
        output.write_text("otro proceso", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            _write_text(output, "nuevo", overwrite=False)
        self.assertEqual(output.read_text(), "otro proceso")
        self.assertFalse(list(self.root.glob(".*.txt.*")))

    def test_empty_transcription_creates_empty_file(self):
        self.model.transcribe.return_value = {"text": "  "}
        output = transcribe_video(self.video)
        self.assertEqual(output.read_bytes(), b"")

    def test_publish_failure_preserves_previous_output_and_cleans_temporary_file(self):
        output = self.video.with_suffix(".txt")
        output.write_text("anterior", encoding="utf-8")
        with patch("video_whisper.transcriber.os.replace", side_effect=PermissionError("Sin permiso")):
            with self.assertRaises(PermissionError):
                _write_text(output, "nuevo", overwrite=True)
        self.assertEqual(output.read_text(), "anterior")
        self.assertFalse(list(self.root.glob(".*.txt.*")))

    def test_cli_success_reports_destination(self):
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            code = main([str(self.video), "--language", "es"])
        self.assertEqual(code, 0)
        self.assertIn(str(self.video.with_suffix(".txt")), stdout.getvalue())

    def test_cli_failure_has_error_code_and_no_traceback(self):
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            code = main([str(self.root / "missing.mp4")])
        self.assertEqual(code, 1)
        self.assertIn("Error:", stderr.getvalue())
        self.assertNotIn("Traceback", stderr.getvalue())

    def test_cli_cancellation_preserves_previous_output(self):
        output = self.video.with_suffix(".txt")
        output.write_text("anterior", encoding="utf-8")
        self.model.transcribe.side_effect = KeyboardInterrupt
        self.assertEqual(main([str(self.video), "--overwrite"]), 130)
        self.assertEqual(output.read_text(), "anterior")


if __name__ == "__main__":
    unittest.main()
