import os, re, shutil, subprocess, json, tempfile
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Video Merge / Compress (Cloud)", page_icon="🎬", layout="centered")

# ===== Session state =====
defaults = {
    "busy_single": False,
    "busy_merge": False,
    "single_payload": None,
    "merge_payload": None,
    "run_params": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🎬 Video Merge / Compress (MP4) — Cloud")
st.caption("Two modes: (1) Compress 1 video, (2) Merge 2 videos & compress. Choose Size Cap (two-pass) or CRF (quality). H.264 or HEVC/H.265.")

# ===== Sidebar =====
with st.sidebar:
    st.header("⚙️ Global Settings")
    mode = st.radio("Workflow", ["Compress 1 video", "Merge 2 videos & compress"], index=0)
    encode_mode = st.radio("Encoding mode", ["Size Cap (two-pass)", "CRF (quality target)"], index=0)
    codec = st.radio("Codec", ["H.264 (libx264)", "HEVC / H.265 (libx265)"], index=0)
    max_height = st.number_input("Max output height (px)", 144, 2160, 1080, 36)
    preset = st.selectbox("Encoder preset (slower = better compression)",
                          ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], index=2)
    audio_kbps = st.slider("Audio bitrate (kbps)", 48, 320, 128, 16)

    st.divider()
    st.subheader("🔧 Advanced")
    custom_ffmpeg = st.text_input("FFmpeg path (optional)", value="")
    custom_ffprobe = st.text_input("FFprobe path (optional)", value="")
    extra_flags = st.text_input("Extra ffmpeg flags (optional)", value="")

# ===== Helpers =====
def which_or(path_hint: str, default_name: str) -> str:
    if path_hint.strip():
        return path_hint.strip()
    found = shutil.which(default_name)
    if not found:
        raise RuntimeError(f"'{default_name}' not found. Install FFmpeg and ensure it's on PATH, or provide a full path in the sidebar.")
    return found

def ffprobe_json(ffprobe: str, filename: str) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", filename],
                                  stderr=subprocess.STDOUT)
    return json.loads(out.decode("utf-8", errors="ignore"))

def ffprobe_duration(ffprobe: str, filename: str) -> float:
    data = ffprobe_json(ffprobe, filename)
    dur = 0.0
    try:
        dur = float(data.get("format", {}).get("duration", 0.0))
    except: pass
    if dur <= 0:
        for s in data.get("streams", []):
            try:
                d = float(s.get("duration", 0.0))
                if d > dur: dur = d
            except: pass
    return dur

def has_audio_stream(ffprobe: str, filename: str) -> bool:
    data = ffprobe_json(ffprobe, filename)
    return any(s.get("codec_type") == "audio" for s in data.get("streams", []))

_time_re = re.compile(r"time=(\d+):(\d+):(\d+(?:\.\d+)?)")
def parse_ffmpeg_time(line: str):
    m = _time_re.search(line)
    if not m: return None
    hh, mm, ss = m.groups()
    try: return int(hh)*3600 + int(mm)*60 + float(ss)
    except: return None

def run_with_progress(cmd: list, step_dur: float, seg_start: float, seg_end: float, progress_bar, log_area, pct_text):
    # add -stats for periodic progress
    cmd = cmd[:2] + ["-stats"] + cmd[2:]
    log_area.write("```\n" + " ".join(cmd) + "\n```")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    max_seen = 0.0
    for line in proc.stdout:
        if line.strip(): log_area.write(line.rstrip())
        t = parse_ffmpeg_time(line)
        if t is not None and step_dur > 0:
            frac = min(max(t / step_dur, 0.0), 1.0)
            overall = seg_start + frac * (seg_end - seg_start)
            if overall > max_seen:
                max_seen = overall
                progress_bar.progress(min(max_seen, 0.999))
                pct_text.write(f"**{int(max_seen*100)}%**")
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {proc.returncode}")

def human_bitrate(bps: float) -> str:
    return f"{bps/1_000_000:.2f} Mbps" if bps >= 1_000_000 else f"{bps/1000:.0f} kbps"

def compute_bitrates(size_mb: float, total_duration: float, audio_kbps: int):
    # 5% safety margin
    target_bits = size_mb * 1024 * 1024 * 8 * 0.95
    audio_bps = audio_kbps * 1000.0
    total_bps = target_bits / max(total_duration, 0.001)
    video_bps = max(total_bps - audio_bps, 200_000)
    return total_bps, video_bps, audio_bps

def codec_name(label: str):
    return "libx264" if "H.264" in label else "libx265"

# ---- Single-video filters: pass1 video-only (avoids aformat error), pass2/CRF full
def single_filters(height: int, dur: float, has_audio: bool):
    pass1 = f"[0:v]scale=-2:{height}:flags=lanczos[v]"  # VIDEO ONLY
    if has_audio:
        a = "[0:a]aresample=async=1:first_pts=0,aformat=sample_rates=48000:channel_layouts=stereo[a]"
    else:
        a = f"anullsrc=cl=stereo:r=48000,atrim=0:{dur:.3f},asetpts=N/SR/TB[a]"
    full = f"[0:v]scale=-2:{height}:flags=lanczos[v];{a}"
    return pass1, full

# ===== Encoders for single-video path =====
def encode_size_cap(ffmpeg: str, vcodec: str, audio_kbps: int, preset: str, video_bps: float,
                    input_specs, pass1_filter: str, full_filter: str, out_path: str,
                    total_dur: float, extra: list, progress_bar, log_area, pct_text):
    passlog = str(Path(out_path).with_suffix("")) + "_2passlog"
    # PASS 1: VIDEO ONLY
    cmd1 = [
        ffmpeg, "-y",
        *input_specs,
        "-filter_complex", pass1_filter,
        "-map", "[v]",
        "-c:v", vcodec,
        "-preset", preset,
        "-b:v", str(int(video_bps)),
        "-maxrate", str(int(video_bps)),
        "-bufsize", str(int(video_bps*2)),
        "-pass", "1",
        "-passlogfile", passlog,
        "-an",
        "-f", "mp4",
        os.devnull
    ] + (extra or [])
    # PASS 2: VIDEO + AUDIO
    cmd2 = [
        ffmpeg, "-y",
        *input_specs,
        "-filter_complex", full_filter,
        "-map", "[v]", "-map", "[a]",
        "-c:v", vcodec,
        "-preset", preset,
        "-b:v", str(int(video_bps)),
        "-maxrate", str(int(video_bps)),
        "-bufsize", str(int(video_bps*2)),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", f"{audio_kbps}k",
        "-pass", "2",
        "-passlogfile", passlog,
        out_path
    ] + (extra or [])
    run_with_progress(cmd1, total_dur, 0.0, 0.5, progress_bar, log_area, pct_text)
    run_with_progress(cmd2, total_dur, 0.5, 1.0, progress_bar, log_area, pct_text)
    for ext in (".log", "-0.log", "-0.log.mbtree", ".mbtree"):
        f = passlog + ext
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

def encode_crf(ffmpeg: str, vcodec: str, audio_kbps: int, preset: str, crf: int,
               input_specs, full_filter: str, out_path: str,
               total_dur: float, extra: list, progress_bar, log_area, pct_text):
    cmd = [
        ffmpeg, "-y",
        *input_specs,
        "-filter_complex", full_filter,
        "-map", "[v]", "-map", "[a]",
        "-c:v", vcodec,
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", f"{audio_kbps}k",
        out_path
    ] + (extra or [])
    run_with_progress(cmd, total_dur, 0.0, 1.0, progress_bar, log_area, pct_text)

# ===== Robust merge helpers: normalize -> concat demuxer -> final compress =====
def make_intermediate(ffmpeg: str, in_path: str, out_path: str, height: int, fps: int, audio_kbps: int, has_audio: bool, dur: float, extra: list, progress_bar, log_area, pct_text, seg_start: float, seg_end: float):
    """Re-encode one clip to uniform size/FPS/audio so concat is safe."""
    if has_audio:
        fc = f"[0:v]scale=-2:{height}:flags=lanczos[v];[0:a]aresample=async=1:first_pts=0,aformat=sample_rates=48000:channel_layouts=stereo[a]"
    else:
        fc = f"[0:v]scale=-2:{height}:flags=lanczos[v];anullsrc=cl=stereo:r=48000,atrim=0:{dur:.3f},asetpts=N/SR/TB[a]"
    cmd = [
        ffmpeg, "-y",
        "-i", in_path,
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-ar", "48000",
        out_path
    ] + (extra or [])
    run_with_progress(cmd, dur, seg_start, seg_end, progress_bar, log_area, pct_text)

def concat_intermediates(ffmpeg: str, list_file: str, out_path: str, progress_bar, log_area, pct_text):
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", "-movflags", "+faststart", out_path]
    # Concat is fast; map to a small progress slice
    run_with_progress(cmd, 1.0, 0.60, 0.68, progress_bar, log_area, pct_text)

def final_encode_size_cap(ffmpeg: str, in_path: str, out_path: str, vcodec: str, preset: str, video_bps: int, audio_kbps: int, total_dur: float, extra: list, progress_bar, log_area, pct_text):
    passlog = str(Path(out_path).with_suffix("")) + "_2passlog"
    # pass 1: video only
    cmd1 = [ffmpeg, "-y", "-i", in_path,
            "-map", "0:v:0",
            "-c:v", vcodec, "-preset", preset,
            "-b:v", str(int(video_bps)), "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
            "-pass", "1", "-passlogfile", passlog, "-an", "-f", "mp4", os.devnull] + (extra or [])
    # pass 2: video + audio
    cmd2 = [ffmpeg, "-y", "-i", in_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", vcodec, "-preset", preset,
            "-b:v", str(int(video_bps)), "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-c:a", "aac", "-b:a", f"{audio_kbps}k",
            "-pass", "2", "-passlogfile", passlog, out_path] + (extra or [])
    run_with_progress(cmd1, total_dur, 0.68, 0.84, progress_bar, log_area, pct_text)
    run_with_progress(cmd2, total_dur, 0.84, 1.00, progress_bar, log_area, pct_text)
    for ext in (".log", "-0.log", "-0.log.mbtree", ".mbtree"):
        f = passlog + ext
        if os.path.exists(f):
            try: os.remove(f)
            except: pass

def final_encode_crf(ffmpeg: str, in_path: str, out_path: str, vcodec: str, preset: str, crf: int, audio_kbps: int, total_dur: float, extra: list, progress_bar, log_area, pct_text):
    cmd = [ffmpeg, "-y", "-i", in_path,
           "-map", "0:v:0", "-m
