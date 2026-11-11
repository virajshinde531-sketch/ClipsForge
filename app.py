import os, re, subprocess, tempfile, zipfile
from pathlib import Path
import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(page_title="ClipForge", page_icon="✂️", layout="wide")
st.title("✂️ ClipForge — AI highlight clips")
st.write("Turn a video you own or have permission to edit into short, captioned clips.")

st.info("YouTube links can be used as a reference, but this starter app does not download videos from YouTube. Paste the link for context, then upload the matching video file you own or are authorized to edit.")
yt_url = st.text_input("YouTube video link (optional)", placeholder="https://www.youtube.com/watch?v=...")
video = st.file_uploader("Upload the video file", type=["mp4", "mov", "mkv", "webm"])
col1, col2, col3 = st.columns(3)
with col1:
    clip_len = st.slider("Clip length (seconds)", 15, 90, 35)
with col2:
    number = st.slider("Number of clips", 1, 10, 5)
with col3:
    model_size = st.selectbox("Transcription model", ["tiny", "base", "small"], index=1)

st.caption("The first run downloads the selected speech-recognition model. Processing speed depends on your computer.")
go = st.button("Find highlights & make clips", type="primary", disabled=video is None)

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2500:])
    return p.stdout

def srt_time(seconds):
    ms = int(max(0, seconds) * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def score_segment(seg):
    text = seg.text.strip()
    words = re.findall(r"\b[\w']+\b", text.lower())
    if not words:
        return 0
    hooks = {"important","secret","mistake","reason","because","best","worst","never","always",
             "tip","trick","remember","surprising","actually","problem","solution","lesson",
             "first","finally","instead","however","discover","learn","why","how"}
    score = min(len(words), 35) / 10
    score += sum(w in hooks for w in words) * 1.4
    score += min(sum(1 for w in words if w.isupper()), 3) * .2
    if len(words) < 5:
        score -= 2
    return score

def make_srt(segments, start, end, path):
    lines, idx = [], 1
    for seg in segments:
        a, b = max(seg.start, start), min(seg.end, end)
        if b <= a:
            continue
        txt = seg.text.strip().replace("-->", "→")
        lines += [str(idx), f"{srt_time(a-start)} --> {srt_time(b-start)}", txt, ""]
        idx += 1
    Path(path).write_text("\n".join(lines), encoding="utf-8")

if go:
    work = Path(tempfile.mkdtemp(prefix="clipforge_"))
    source = work / video.name
    source.write_bytes(video.getbuffer())
    outdir = work / "clips"
    outdir.mkdir()
    try:
        with st.spinner("Checking video and transcribing speech..."):
            info = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1", str(source)])
            duration = float(info.strip())
            model = WhisperModel(model_size, device="cpu", compute_type="int8")
            segments_iter, _ = model.transcribe(str(source), vad_filter=True, word_timestamps=False)
            segments = list(segments_iter)
        if not segments:
            st.error("No clear speech was detected in this video.")
            st.stop()

        # Generate candidate windows around transcript segments and score the speech inside each.
        step = max(8, clip_len // 3)
        candidates = []
        start = 0.0
        while start < duration:
            end = min(duration, start + clip_len)
            contained = [s for s in segments if s.end > start and s.start < end]
            if contained:
                score = sum(score_segment(s) for s in contained)
                speech = sum(max(0, min(s.end, end)-max(s.start, start)) for s in contained)
                score += min(speech / max(clip_len, 1), 1) * 2
                candidates.append((score, start, end))
            start += step

        chosen = []
        for item in sorted(candidates, reverse=True):
            _, a, b = item
            if all(abs(a-x[1]) >= clip_len * .65 for x in chosen):
                chosen.append(item)
            if len(chosen) >= number:
                break
        chosen.sort(key=lambda x: x[1])

        st.success(f"Found {len(chosen)} highlight candidates.")
        generated = []
        for i, (score, a, b) in enumerate(chosen, 1):
            srt = outdir / f"clip_{i}.srt"
            mp4 = outdir / f"clip_{i}.mp4"
            make_srt(segments, a, b, srt)
            # Use a relative subtitle path and escape it for ffmpeg's filter syntax.
            subpath = str(srt).replace("\\", "/").replace(":", r"\:")
            cmd = ["ffmpeg", "-y", "-ss", str(a), "-i", str(source), "-t", str(b-a),
                   "-vf", f"subtitles='{subpath}':force_style='FontSize=20,Outline=2,Shadow=1,Alignment=2,MarginV=35'",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                   "-c:a", "aac", "-b:a", "128k", str(mp4)]
            with st.spinner(f"Rendering clip {i}/{len(chosen)}..."):
                run(cmd)
            generated.append((mp4, srt, a, b))
            st.write(f"**Clip {i}** · {a:.1f}s–{b:.1f}s")
            st.video(str(mp4))
            with open(mp4, "rb") as f:
                st.download_button(f"Download clip {i}", f, file_name=mp4.name, mime="video/mp4", key=f"mp4_{i}")
            with open(srt, "rb") as f:
                st.download_button(f"Download captions {i} (.srt)", f, file_name=srt.name, key=f"srt_{i}")

        bundle = work / "ClipForge_exports.zip"
        with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
            for mp4, srt, _, _ in generated:
                z.write(mp4, mp4.name)
                z.write(srt, srt.name)
        st.download_button("Download all clips + captions (ZIP)", bundle.read_bytes(),
                           file_name="ClipForge_exports.zip", mime="application/zip")
        st.caption("Highlight selection is an automatic first pass. Review clips before publishing; it may miss context or choose imperfect moments.")
    except Exception as e:
        st.error("Could not process this video.")
        st.code(str(e))
        st.write("Check that FFmpeg is installed and available on PATH, and try a smaller transcription model.")
