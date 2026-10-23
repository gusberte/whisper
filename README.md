# Video Whisper CLI

Aplicación Python para transcribir el audio completo de un archivo de video con
[OpenAI Whisper](https://github.com/openai/whisper). El procesamiento es local y
no requiere una clave de API. La primera ejecución descarga el modelo elegido;
las siguientes reutilizan la copia local.

## Instalación

Se recomienda Python 3.11. FFmpeg debe estar instalado y disponible en el PATH.
En macOS:

```sh
brew install ffmpeg
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

En Ubuntu/Debian, FFmpeg se instala con `sudo apt install ffmpeg`.
En Windows, instalalo con `winget install Gyan.FFmpeg` y activá el entorno
con `.venv\Scripts\Activate.ps1` en PowerShell.

La dependencia se llama **openai-whisper**; `whisper` es el nombre de su módulo
Python. No es necesario clonar ni copiar el código del repositorio oficial.

## Uso

```sh
transcribir-video "/ruta/mi video.mp4"
```

Genera `/ruta/mi video.txt` en UTF-8. Acepta rutas relativas o absolutas y los
formatos de video/audio que FFmpeg pueda decodificar. Los mensajes y el progreso
se muestran en la terminal. Se transcribe en el idioma original.

```sh
# Indicar español, modelo y destino (se crean las carpetas necesarias).
transcribir-video video.mp4 --language es --model small -o transcripciones/video.txt

# Reemplazar un TXT existente.
transcribir-video video.mp4 --overwrite

# Elegir dónde guardar los modelos y forzar CPU.
transcribir-video video.mp4 --model-dir .cache/whisper --device cpu

# También se puede ejecutar como módulo.
python -m video_whisper video.mp4

transcribir-video --help
```

El modelo predeterminado es `base`, multilingüe. `tiny` es más liviano; `small`,
`medium`, `large` y `turbo` ofrecen otras relaciones entre precisión, memoria y
velocidad. Los modelos terminados en `.en` son para inglés. Se puede omitir
`--language` para detectar el idioma al comienzo del audio.

Se usa CUDA automáticamente cuando está disponible y CPU en los demás casos,
incluidos los equipos Mac. `--device cuda` exige una GPU CUDA disponible.

La aplicación evita sobrescribir archivos salvo que se use `--overwrite`, y
siempre rechaza usar el video de entrada como destino. Los errores terminan con
código 1, los argumentos inválidos con código 2 y una cancelación con código 130.

Whisper carga el audio completo en memoria y lo procesa en ventanas internas de
30 segundos: no hay un límite de 30 segundos para el video. Un video muy largo
puede requerir bastante memoria y tiempo. La transcripción automática puede
contener errores; en videos con varias pistas de audio, se procesa la pista
seleccionada automáticamente por FFmpeg.

## Pruebas

```sh
python -m unittest discover -s tests -v
```

Las pruebas usan un motor simulado para verificar la CLI, la integración y la
protección de archivos sin descargar modelos. Para probar el reconocimiento
real, ejecutá la CLI con un video que contenga voz.
