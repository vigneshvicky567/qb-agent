from fastapi import UploadFile
import whisper
import tempfile
import os

# Load the Whisper model once at module level to avoid reloading on every request.
# Model size tradeoffs — larger = more accurate but slower:
#   tiny, base (fast, less accurate) → small, medium, large, turbo (slow, more accurate)
model = whisper.load_model("base")

class AudioService:
    async def transcribe_audio(self, file: UploadFile) -> dict:
        """
        Transcribe an uploaded audio file using OpenAI Whisper.

        Saves the incoming UploadFile to a temporary file on disk (required by
        Whisper's file-path API), runs transcription, then deletes the temp file.

        Returns a dict with:
            - transcript (str): The recognised text.
            - language (str): The detected language code (e.g. "en"), or "unknown".
        """
        print(f"SERVICE: Executing Whisper transcription")

        print(file.filename)

        # Whisper requires a file path rather than an in-memory buffer.
        # Preserve the original file extension so Whisper can infer the audio format.
        suffix = os.path.splitext(file.filename or "audio.mp3")[-1] or ".mp3"

        # Write the uploaded bytes to a named temp file; delete=False keeps it
        # alive after the `with` block so Whisper can open it by path.
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            contents = await file.read()
            tmp.write(contents)
            tmp_path = tmp.name

        try:
            print(f"Temp file created at: {tmp_path} (size: {len(contents)} bytes)")
            # Transcribe — result["text"] contains the full transcript string.
            result = model.transcribe(tmp_path)

            # TODO: Save the transcript to a database, or return additional metadata like word-level timestamps.
            # saved transcription is candidate's audio. 
            # compare this with the passage you have in db and calculate the score with the help of ai model

            return {
                "transcript": result["text"],
                "language": result.get("language", "unknown")
            }
            
        finally:
            # Always remove the temp file, even if transcription raises an error.
            os.unlink(tmp_path)
