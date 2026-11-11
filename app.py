import re, subprocess, tempfile, zipfile
from pathlib import Path
from collections import Counter
import streamlit as st
import cv2
from faster_whisper import WhisperModel

st.set_page_config(page_title="ClipForge", page_icon="✂️", layout="wide")
st.title("✂️ ClipForge")
st.caption("Turn authorized videos into captioned vertical highlight clips.")

st.info("You may paste a YouTube URL as a reference, but this app does not download YouTube videos. Upload the original video or an authorized export.")
url = st.text_input("YouTube URL (optional)", placeholder="https://www.youtube.com/watch?v=...")
video = st.file_uploader("Upload your video", type=["mp4", "mov", "mkv", "webm"])

with st.sidebar:
    st.header("Clip settings")
    clip_seconds = st.slider("Clip length", 15, 90, 35, 5)
    clip_count = st.slider("Number of clips", 1, 10, 5)
    model_size = st.selectbox("Speech model", ["tiny", "base", "small"], index=1)
    face_track = st.checkbox("Auto face framing", True)
    music_on = st.checkbox("Add background music", False)
    music_volume = st.slider("Music volume", 0.0, 0.35, 0.10, 0.01) if music_on else 0.0
    sfx_on = st.checkbox("Add sound effect", False)

music = st.file_uploader("Upload royalty-free/authorized background music", type=["mp3","wav","m4a"]) if music_on else None
sfx = st.file_uploader("Upload authorized sound effect", type=["mp3","wav","m4a"]) if sfx_on else None
rights = st.checkbox("I own this video/audio or have permission to edit and publish it.")
go = st.button("Generate clips", type="primary", disabled=not (video and rights))

HOOKS = {"secret":2.0,"mistake":1.8,"reason":1.4,"because":1.0,"best":1.0,"worst":1.0,
         "never":1.2,"important":1.5,"surprising":2.0,"actually":1.2,"problem":1.2,
         "solution":1.5,"lesson":1.3,"why":0.8,"how":0.8,"stop":1.0,"truth":1.5,
         "avoid":1.0,"discover":1.2,"learn":0.8,"finally":0.8}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return p.stdout

def duration(path):
    return float(run(["ffprobe","-v","error","-show_entries","format=duration",
                      "-of","default=noprint_wrappers=1:nokey=1",str(path)]).strip())

def ass_time(t):
    t=max(0,t); h=int(t//3600); m=int(t%3600//60); s=t%60
    return f"{h}:{m:02}:{s:05.2f}"

def choose_windows(segments, total, length, count):
    step=max(8,length//3); candidates=[]; start=0
    while start < total:
        end=min(total,start+length)
        inside=[s for s in segments if s.end>start and s.start<end]
        words=re.findall(r"[a-z']+"," ".join(s.text for s in inside).lower())
        if words:
            hook=sum(HOOKS.get(w,0) for w in words)
            density=min(len(words)/max(length/60*140,1),1.5)
            questions=sum("?" in s.text for s in inside)*.7
            score=hook+density*4+questions
            if len(words)<10: score-=2
            candidates.append((score,start,end))
        start+=step
    selected=[]
    for item in sorted(candidates,reverse=True):
        if all(abs(item[1]-x[1]) >= length*.7 for x in selected):
            selected.append(item)
        if len(selected)>=count: break
    return sorted(selected,key=lambda x:x[1])

def make_ass(segments, start, end, path):
    header="""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,64,&H00FFFFFF,&H0000FFFF,&H00101010,&H80000000,1,0,0,0,100,100,0,0,1,5,2,2,65,65,210,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    rows=[header]
    for seg in segments:
        a=max(start,seg.start); b=min(end,seg.end)
        if b<=a: continue
        tokens=[]
        for w in (getattr(seg,"words",None) or []):
            wa=max(a,w.start); wb=min(b,w.end)
            if wb>wa:
                txt=w.word.strip().replace("{","").replace("}","")
                tokens.append(r"{\k"+str(max(1,round((wb-wa)*100)))+"}"+txt)
        txt=" ".join(tokens) if tokens else seg.text.strip()
        txt=txt.replace("{","").replace("}","").replace("\n"," ")
        rows.append(f"Dialogue: 0,{ass_time(a-start)},{ass_time(b-start)},Default,,0,0,0,,{txt}\n")
    Path(path).write_text("".join(rows),encoding="utf-8-sig")

def face_center(path, start, end):
    cap=cv2.VideoCapture(str(path))
    if not cap.isOpened(): return None
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detector=cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    centers=[]
    for t in [start+(end-start)*i/6 for i in range(7)]:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,frame=cap.read()
        if not ok: continue
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=detector.detectMultiScale(gray,1.1,4,minSize=(35,35))
        if len(faces):
            x,y,w,h=max(faces,key=lambda f:f[2]*f[3]); centers.append(x+w/2)
    cap.release()
    return (sum(centers)/len(centers),width,height) if centers else None

def metadata(text, n):
    words=re.findall(r"[A-Za-z0-9']+",text)
    stop={"the","and","that","this","with","you","your","are","for","was","have","has","but","not","from","they","then","what","when","will","into","about","just","like"}
    useful=[w for w in words if w.lower() not in stop and len(w)>3]
    title=" ".join(dict.fromkeys(w.capitalize() for w in useful[:7])) or f"Highlight {n}"
    tags=["#shorts","#reels","#highlights"]
    for w,_ in Counter(w.lower() for w in useful if len(w)>4).most_common(3):
        tags.append("#"+re.sub(r"[^a-z0-9]","",w))
    return title, " ".join(dict.fromkeys(tags))

if go:
    work=Path(tempfile.mkdtemp(prefix="clipforge_"))
    src=work/video.name
    src.write_bytes(video.getbuffer())
    outdir=work/"exports"; outdir.mkdir()
    music_path=None; sfx_path=None
    if music:
        music_path=work/("music_"+music.name); music_path.write_bytes(music.getbuffer())
    if sfx:
        sfx_path=work/("sfx_"+sfx.name); sfx_path.write_bytes(sfx.getbuffer())
    try:
        with st.spinner("Transcribing speech and finding highlight moments..."):
            total=duration(src)
            model=WhisperModel(model_size,device="cpu",compute_type="int8")
            seg_iter,_=model.transcribe(str(src),vad_filter=True,word_timestamps=True)
            segs=list(seg_iter)
        if not segs:
            st.error("No speech was detected. Try another video.")
            st.stop()
        chosen=choose_windows(segs,total,clip_seconds,clip_count)
        if not chosen:
            st.error("Could not find suitable speech segments.")
            st.stop()
        st.success(f"Selected {len(chosen)} candidate highlights. Review them before publishing.")
        exported=[]
        for i,(_,a,b) in enumerate(chosen,1):
            ass=outdir/f"clip_{i}.ass"; mp4=outdir/f"clip_{i}.mp4"
            make_ass(segs,a,b,ass)
            center=face_center(src,a,b) if face_track else None
            if center:
                cx,w,h=center
                crop_w=int(h*9/16)
                x=max(0,min(w-crop_w,int(cx-crop_w/2)))
                vf=f"crop={crop_w}:{h}:{x}:0,scale=1080:1920,ass='{str(ass).replace(chr(92),'/').replace(':',r'\:')}'"
            else:
                vf=f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,ass='{str(ass).replace(chr(92),'/').replace(':',r'\:')}'"
            cmd=["ffmpeg","-y","-ss",str(a),"-i",str(src),"-t",str(b-a)]
            if music_path:
                cmd += ["-stream_loop","-1","-i",str(music_path)]
            if sfx_path:
                cmd += ["-i",str(sfx_path)]
            if music_path and sfx_path:
                cmd += ["-filter_complex",f"[0:a]volume=1[a0];[1:a]volume={music_volume}[am];[2:a]volume=0.6[as];[a0][am][as]amix=inputs=3:duration=first:dropout_transition=2[aout]","-map","0:v","-map","[aout]"]
            elif music_path:
                cmd += ["-filter_complex",f"[0:a]volume=1[a0];[1:a]volume={music_volume}[am];[a0][am]amix=inputs=2:duration=first:dropout_transition=2[aout]","-map","0:v","-map","[aout]"]
            elif sfx_path:
                cmd += ["-filter_complex","[0:a][1:a]amix=inputs=2:duration=first:dropout_transition=2[aout]","-map","0:v","-map","[aout]"]
            cmd += ["-vf",vf,"-c:v","libx264","-preset","veryfast","-crf","23","-c:a","aac","-b:a","128k","-shortest",str(mp4)]
            with st.spinner(f"Rendering clip {i}/{len(chosen)}..."):
                run(cmd)
            transcript=" ".join(s.text.strip() for s in segs if s.end>a and s.start<b)
            title,tags=metadata(transcript,i)
            st.subheader(f"Clip {i}: {title}")
            st.caption(f"{a:.1f}s–{b:.1f}s · Suggested hashtags: {tags}")
            st.video(str(mp4))
            exported.append(mp4)
        bundle=work/"ClipForge_all_clips.zip"
        with zipfile.ZipFile(bundle,"w",zipfile.ZIP_DEFLATED) as z:
            for f in exported: z.write(f,f.name)
        st.download_button("⬇️ Download all clips (ZIP)",bundle.read_bytes(),
                           file_name="ClipForge_all_clips.zip",mime="application/zip")
        st.caption("Highlight scoring and title/hashtag suggestions are automated heuristics, not a guarantee of virality. Face framing uses basic face detection and may miss speakers. Word highlighting depends on transcription timing.")
    except Exception as e:
        st.error("Processing failed. Try a shorter/smaller video or the tiny model.")
        st.code(str(e))

