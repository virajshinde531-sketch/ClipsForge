# ClipForge — AI highlight clip maker

A local starter web app that:
1. Accepts an uploaded video file.
2. Transcribes speech with Faster-Whisper.
3. Scores short windows for likely highlight moments.
4. Exports MP4 clips with burned-in captions and separate SRT caption files.

## YouTube link workflow
Paste the YouTube URL as a reference, then upload the corresponding video file that you own or have permission to edit. This app intentionally does not download videos from YouTube. Use YouTube's own download/export options where available or your original source file.

## Requirements
- Python 3.10+
- FFmpeg installed and available on PATH
- Internet for the first model download

## Run on Windows
1. Install Python from https://www.python.org/downloads/
2. Install FFmpeg and add its `bin` folder to PATH.
3. Extract this ZIP.
4. Open Command Prompt in the extracted folder and run:

   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   streamlit run app.py

5. Open the local address Streamlit prints (usually http://localhost:8501).

## Notes
- This is a local prototype, not a hosted public website.
- CPU transcription can take a while. Start with the `tiny` or `base` model.
- Automatic highlight scoring is heuristic, not a guarantee that every selected clip is the best moment.
- Review captions, clip boundaries, and rights/permissions before publishing.
