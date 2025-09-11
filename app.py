import os, re, shutil, subprocess, json, tempfile
from pathlib import Path
from uuid import uuid4
import streamlit as st

st.set_page_config(page_title="Video Merge / Compress (Cloud)", page_icon="🎬", layout="centered")

# ============= Big-file friendly workspace =============
WORKDIR = Path("workdir")
WORKDIR.mkdir(exist_ok=True)

# ============= Session state =============
defaults = {
    "busy_single": False,
    "busy_merge": False,
    "merge_payload": None,   # [{"name":..., "data":...}, {"name":..., "data":...}]
    "run_params": None,      # snapshot of UI & paths at click time
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

st.title("🎬 Video Merge / Compress (MP4) — Cloud")
st.caption("Two modes: (1) Compress 1 video, (2) Merge 2 videos & compress. Choose Size Cap (two-pass) or CRF (quality). H.264 or HEVC/H.265.")

# ============= Sidebar =============
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

# ============= Helpers =============
def which_or(path_hint: str, default_name: str) -> str:
    if path_hint.strip():
        return path_hint.strip()
    found = shutil.which(default_name)
    if not found:
        raise RuntimeError(f"'{default_name}' not found. Install FFmpeg or provide a full path in the sidebar.")
    return found

def ffprobe_json(ffprobe: str, filename: str) -> dict:
    out = subprocess.check_output([ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", filename],
                                  stderr=subprocess.STDOUT)
    return json.loads(out.decode("utf-8", errors="ignore"))

def ffprobe_duration(ffprobe: str, filename: str) -> float:
    data = ffprobe_json(ffprobe, filename)
    dur = 0.0
    try: dur = float(data.get("format", {}).get("duration", 0.0))
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
    # add -stats so ffmpeg prints periodic time= lines
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
    target_bits = size_mb * 1024 * 1024 * 8 * 0.95  # 5% safety
    audio_bps = audio_kbps * 1000.0
    total_bps = target_bits / max(total_duration, 0.001)
    video_bps = max(total_bps - audio_bps, 200_000)
    return total_bps, video_bps, audio_bps

def codec_name(label: str):
    return "libx264" if "H.264" in label else "libx265"

# ---- Single-video filters: pass1 video-only (fixes aformat), pass2/CRF full
def single_filters(height: int, dur: float, has_audio: bool):
    p1 = f"[0:v]scale=-2:{height}:flags=lanczos[v]"  # PASS 1: VIDEO ONLY
    if has_audio:
        a = "[0:a]aresample=async=1:first_pts=0,aformat=sample_rates=48000:channel_layouts=stereo[a]"
    else:
        a = f"anullsrc=cl=stereo:r=48000,atrim=0:{dur:.3f},asetpts=N/SR/TB[a]"
    full = f"[0:v]scale=-2:{height}:flags=lanczos[v];{a}"
    return p1, full

# ---- Merge: normalize -> concat demuxer -> final compress (rock-solid)
def make_intermediate(ffmpeg: str, in_path: str, out_path: str, height: int, fps: int, audio_kbps: int, has_audio: bool, dur: float, extra: list, progress_bar, log_area, pct_text, seg_start: float, seg_end: float):
    if has_audio:
        fc = f"[0:v]scale=-2:{height}:flags=lanczos[v];[0:a]aresample=async=1:first_pts=0,aformat=sample_rates=48000:channel_layouts=stereo[a]"
    else:
        fc = f"[0:v]scale=-2:{height}:flags=lanczos[v];anullsrc=cl=stereo:r=48000,atrim=0:{dur:.3f},asetpts=N/SR/TB[a]"
    cmd = [
        ffmpeg, "-y", "-i", in_path,
        "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(fps), "-movflags", "+faststart",
        "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-ar", "48000",
        out_path
    ] + (extra or [])
    run_with_progress(cmd, dur, seg_start, seg_end, progress_bar, log_area, pct_text)

def concat_intermediates(ffmpeg: str, list_file: str, out_path: str, progress_bar, log_area, pct_text):
    cmd = [ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", "-movflags", "+faststart", out_path]
    run_with_progress(cmd, 1.0, 0.60, 0.68, progress_bar, log_area, pct_text)  # tiny slice for UX

def final_encode_size_cap(ffmpeg: str, in_path: str, out_path: str, vcodec: str, preset: str, video_bps: int, audio_kbps: int, total_dur: float, extra: list, progress_bar, log_area, pct_text):
    passlog = str(Path(out_path).with_suffix("")) + "_2passlog"
    # Pass 1: video only
    cmd1 = [ffmpeg, "-y", "-i", in_path, "-map", "0:v:0",
            "-c:v", vcodec, "-preset", preset, "-b:v", str(int(video_bps)),
            "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
            "-pass", "1", "-passlogfile", passlog, "-an", "-f", "mp4", os.devnull] + (extra or [])
    # Pass 2: video + audio
    cmd2 = [ffmpeg, "-y", "-i", in_path,
            "-map", "0:v:0", "-map", "0:a:0?",
            "-c:v", vcodec, "-preset", preset, "-b:v", str(int(video_bps)),
            "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
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
           "-map", "0:v:0", "-map", "0:a:0?",
           "-c:v", vcodec, "-preset", preset, "-crf", str(crf),
           "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           "-c:a", "aac", "-b:a", f"{audio_kbps}k",
           out_path] + (extra or [])
    run_with_progress(cmd, total_dur, 0.68, 1.00, progress_bar, log_area, pct_text)

# ============= Single-video (big-file safe & bug-fixed) =============
def compress_single(ffmpeg: str, ffprobe: str, params: dict):
    in_path = params["in_path"]
    if not os.path.exists(in_path):
        st.error("Uploaded file is no longer available on disk.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        out_path = tmpdir / params["out_name"]

        log_area = st.empty(); pct_text = st.empty(); progress_bar = st.progress(0.0)

        dur = ffprobe_duration(ffprobe, in_path)
        has_a = has_audio_stream(ffprobe, in_path)
        vcodec = "libx264" if "H.264" in params["codec"] else "libx265"

        p1_filter, full_filter = single_filters(params["max_height"], dur, has_a)

        if "Size Cap" in params["encode_mode"]:
            target_bits = params["size_mb"] * 1024 * 1024 * 8 * 0.95
            total_bps = target_bits / max(dur, 0.001)
            video_bps = max(total_bps - params["audio_kbps"] * 1000.0, 200_000)
            st.info(f"Duration: **{dur:.1f}s** • Video: **{human_bitrate(video_bps)}** • Audio: **{params['audio_kbps']} kbps**")

            passlog = str(out_path) + "_2passlog"
            # Pass 1 — video only (avoids 'aformat unconnected')
            cmd1 = [params["ffmpeg"], "-y", "-i", in_path,
                    "-filter_complex", p1_filter, "-map", "[v]",
                    "-c:v", vcodec, "-preset", params["preset"],
                    "-b:v", str(int(video_bps)), "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
                    "-pass", "1", "-passlogfile", passlog, "-an",
                    "-f", "mp4", os.devnull] + (params["extra"] or [])
            # Pass 2 — video + audio
            cmd2 = [params["ffmpeg"], "-y", "-i", in_path,
                    "-filter_complex", full_filter, "-map", "[v]", "-map", "[a]",
                    "-c:v", vcodec, "-preset", params["preset"],
                    "-b:v", str(int(video_bps)), "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-c:a", "aac", "-b:a", f"{params['audio_kbps']}k",
                    "-pass", "2", "-passlogfile", passlog,
                    str(out_path)] + (params["extra"] or [])
            run_with_progress(cmd1, dur, 0.00, 0.50, progress_bar, log_area, pct_text)
            run_with_progress(cmd2, dur, 0.50, 1.00, progress_bar, log_area, pct_text)
            for ext in (".log", "-0.log", "-0.log.mbtree", ".mbtree"):
                f = passlog + ext
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass
        else:
            st.info(f"CRF: **{params['crf']}** (lower = higher quality, larger file)")
            cmd = [params["ffmpeg"], "-y", "-i", in_path,
                   "-filter_complex", full_filter, "-map", "[v]", "-map", "[a]",
                   "-c:v", vcodec, "-preset", params["preset"], "-crf", str(params["crf"]),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                   "-c:a", "aac", "-b:a", f"{params['audio_kbps']}k",
                   str(out_path)] + (params["extra"] or [])
            run_with_progress(cmd, dur, 0.00, 1.00, progress_bar, log_area, pct_text)

        progress_bar.progress(1.0); pct_text.write("**100%**")
        with open(out_path, "rb") as f:
            st.success(f"Done! Output size: {out_path.stat().st_size/(1024*1024):.1f} MB")
            st.download_button("⬇️ Download MP4", f, file_name=params["out_name"], mime="video/mp4")

    # Try to clean the original uploaded file from disk
    try: os.remove(in_path)
    except: pass

# ============= Merge pipeline (normalize → concat → final compress) =============
def merge_and_compress(ffmpeg: str, ffprobe: str, payloads: list, params: dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        # Write inputs
        in1 = tmpdir / payloads[0]["name"]; in1.write_bytes(payloads[0]["data"])
        in2 = tmpdir / payloads[1]["name"]; in2.write_bytes(payloads[1]["data"])

        log_area = st.empty(); pct_text = st.empty(); progress_bar = st.progress(0.0)

        dur1 = ffprobe_duration(ffprobe, str(in1))
        dur2 = ffprobe_duration(ffprobe, str(in2))
        total_dur = dur1 + dur2
        has_a1 = has_audio_stream(ffprobe, str(in1))
        has_a2 = has_audio_stream(ffprobe, str(in2))

        # 1) Make intermediates (uniform size/FPS/audio)
        inter1 = tmpdir / "inter1.mp4"
        inter2 = tmpdir / "inter2.mp4"
        fps = 30
        make_intermediate(params["ffmpeg"], str(in1), str(inter1), params["max_height"], fps, params["audio_kbps"], has_a1, dur1, params["extra"], progress_bar, log_area, pct_text, 0.00, 0.30)
        make_intermediate(params["ffmpeg"], str(in2), str(inter2), params["max_height"], fps, params["audio_kbps"], has_a2, dur2, params["extra"], progress_bar, log_area, pct_text, 0.30, 0.60)

        # 2) Concatenate without re-encoding
        listfile = tmpdir / "list.txt"
        listfile.write_text(f"file '{inter1.as_posix()}'\nfile '{inter2.as_posix()}'\n", encoding="utf-8")
        merged_pre = tmpdir / "merged_pre.mp4"
        concat_intermediates(params["ffmpeg"], str(listfile), str(merged_pre), progress_bar, log_area, pct_text)

        # 3) Final compress
        final_path = tmpdir / params["out_name"]
        vcodec = "libx264" if "H.264" in params["codec"] else "libx265"
        if "Size Cap" in params["encode_mode"]:
            _, video_bps, _ = compute_bitrates(params["size_mb"], total_dur, params["audio_kbps"])
            st.info(f"Total duration: **{total_dur:.1f}s** • Video: **{human_bitrate(video_bps)}** • Audio: **{params['audio_kbps']} kbps**")
            final_encode_size_cap(params["ffmpeg"], str(merged_pre), str(final_path), vcodec, params["preset"], int(video_bps), params["audio_kbps"], total_dur, params["extra"], progress_bar, log_area, pct_text)
        else:
            st.info(f"CRF: **{params['crf']}** (lower = higher quality, bigger file)")
            final_encode_crf(params["ffmpeg"], str(merged_pre), str(final_path), vcodec, params["preset"], params["crf"], params["audio_kbps"], total_dur, params["extra"], progress_bar, log_area, pct_text)

        progress_bar.progress(1.0); pct_text.write("**100%**")
        st.success(f"Done! Output size: {final_path.stat().st_size/(1024*1024):.1f} MB")
        st.download_button("⬇️ Download MP4", final_path.read_bytes(), file_name=params["out_name"], mime="video/mp4")

# ============= UI =============
if mode == "Compress 1 video":
    st.markdown("**Step 1.** Upload a single video file.")
    up = st.file_uploader("Drop 1 file here", accept_multiple_files=False, type=None, key="single_uploader")
    st.markdown("**Step 2.** Choose your output strategy below.")
    size_mb = st.number_input("Target size (MB) — only for Size Cap", 10.0, 10000.0, 250.0, 10.0)
    crf = st.slider("CRF (18–28 typical) — only for CRF", 0, 51, 23)
    out_name = st.text_input("Output filename", value="compressed.mp4")

    if not st.session_state["busy_single"]:
        if st.button("▶️ Start", type="primary", disabled=(up is None)):
            if up is None:
                st.warning("Please upload a file.")
            else:
                # Save upload to disk in chunks; only store the PATH in session
                dst = WORKDIR / f"{uuid4()}_{up.name}"
                up.seek(0)
                with open(dst, "wb") as f:
                    shutil.copyfileobj(up, f, length=1024 * 1024)  # 1MB chunks
                st.session_state["run_params"] = {
                    "encode_mode": encode_mode, "codec": codec, "max_height": int(max_height),
                    "preset": preset, "audio_kbps": int(audio_kbps), "size_mb": float(size_mb),
                    "crf": int(crf), "out_name": out_name, "in_path": str(dst),
                    "ffmpeg": which_or(custom_ffmpeg, "ffmpeg"),
                    "extra": extra_flags.strip().split() if extra_flags.strip() else []
                }
                st.session_state["busy_single"] = True
                st.rerun()
    else:
        st.button("Processing…", disabled=True)
        try:
            ffprobe = which_or(custom_ffprobe, "ffprobe")
            compress_single(st.session_state["run_params"]["ffmpeg"], ffprobe, st.session_state["run_params"])
        except Exception as e:
            if "Unknown encoder 'libx265'" in str(e):
                st.error("HEVC/H.265 (libx265) not available in this FFmpeg build. Switch to H.264.")
            else:
                st.error(f"Failed: {e}")
        finally:
            st.session_state["busy_single"] = False
            st.session_state["run_params"] = None

else:
    st.markdown("**Step 1. Choose order explicitly**")
    col1, col2 = st.columns(2)
    with col1:
        up1 = st.file_uploader("First video", accept_multiple_files=False, type=None, key="merge_first")
    with col2:
        up2 = st.file_uploader("Second video", accept_multiple_files=False, type=None, key="merge_second")

    st.markdown("**Step 2.** Choose your output strategy below.")
    size_mb = st.number_input("Target size (MB) — only for Size Cap", 10.0, 10000.0, 250.0, 10.0)
    crf = st.slider("CRF (18–28 typical) — only for CRF", 0, 51, 23)
    out_name = st.text_input("Output filename", value="merged.mp4")

    if not st.session_state["busy_merge"]:
        disabled = (up1 is None or up2 is None)
        if st.button("▶️ Start", type="primary", disabled=disabled):
            if disabled:
                st.warning("Please upload both files (First + Second).")
            else:
                st.session_state["merge_payload"] = [
                    {"name": up1.name, "data": up1.getvalue()},
                    {"name": up2.name, "data": up2.getvalue()},
                ]
                st.session_state["run_params"] = {
                    "encode_mode": encode_mode, "codec": codec, "max_height": int(max_height),
                    "preset": preset, "audio_kbps": int(audio_kbps), "size_mb": float(size_mb),
                    "crf": int(crf), "out_name": out_name,
                    "ffmpeg": which_or(custom_ffmpeg, "ffmpeg"),
                    "extra": extra_flags.strip().split() if extra_flags.strip() else []
                }
                st.session_state["busy_merge"] = True
                st.rerun()
    else:
        st.button("Processing…", disabled=True)
        try:
            ffprobe = which_or(custom_ffprobe, "ffprobe")
            merge_and_compress(st.session_state["run_params"]["ffmpeg"], ffprobe, st.session_state["merge_payload"], st.session_state["run_params"])
        except Exception as e:
            if "Unknown encoder 'libx265'" in str(e):
                st.error("HEVC/H.265 (libx265) not available in this FFmpeg build. Switch to H.264.")
            else:
                st.error(f"Failed: {e}")
        finally:
            st.session_state["busy_merge"] = False
            st.session_state["merge_payload"] = None
            st.session_state["run_params"] = None
