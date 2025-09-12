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
    preset_choice = st.selectbox("Encoder preset (slower = better compression)",
                                 ["ultrafast","superfast","veryfast","faster","fast","medium","slow","slower","veryslow"], index=2)
    audio_kbps = st.slider("Audio bitrate (kbps)", 48, 320, 128, 16)

    st.divider()
    st.subheader("⚡ Speed profile")
    speed_profile = st.selectbox(
        "Prioritize speed vs quality",
        ["Balanced (use preset above)", "Fast", "Turbo (fastest, more loss)"],
        index=0
    )

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

def ffprobe_height_bitrate_codec(ffprobe: str, filename: str):
    """Return (height, total_bps, vcodec, pix_fmt) best-effort."""
    data = ffprobe_json(ffprobe, filename)
    h = None; vcodec = None; pix_fmt = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            h = s.get("height", h)
            vcodec = s.get("codec_name", vcodec)
            pix_fmt = s.get("pix_fmt", pix_fmt)
    try:
        total_bps = float(data.get("format", {}).get("bit_rate", 0.0))
    except:
        total_bps = 0.0
    return h or 0, total_bps or 0.0, (vcodec or "").lower(), (pix_fmt or "").lower()

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
    # add -stats for periodic progress (assumes cmd[0] is ffmpeg, cmd[1] is -y)
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

def speed_params(profile: str, user_preset: str):
    """Return (preset, scale_kernel, fps_cap, crf_bump)."""
    if profile.startswith("Turbo"):
        return ("ultrafast", "bilinear", 24, 3)
    if profile.startswith("Fast"):
        return ("veryfast", "bicubic", 30, 1)
    # Balanced
    return (user_preset, "lanczos", None, 0)

# ---- Single-video filters: pass1 video-only (avoids aformat error), pass2/CRF full
def single_filters(height: int, dur: float, has_audio: bool, scale_kernel: str, fps_cap: int|None):
    fps_part = f",fps={fps_cap}" if fps_cap else ""
    pass1 = f"[0:v]scale=-2:{height}:flags={scale_kernel}{fps_part}[v]"  # VIDEO ONLY
    if has_audio:
        a = "[0:a]aresample=async=1:first_pts=0,aformat=sample_rates=48000:channel_layouts=stereo[a]"
    else:
        a = f"anullsrc=cl=stereo:r=48000,atrim=0:{dur:.3f},asetpts=N/SR/TB[a]"
    full = f"[0:v]scale=-2:{height}:flags={scale_kernel}{fps_part}[v];{a}"
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

# ===== Pipelines =====
def compress_single(ffmpeg: str, ffprobe: str, payload: dict, params: dict):
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        in1 = tmpdir / payload["name"]; in1.write_bytes(payload["data"])
        out_path = tmpdir / params["out_name"]

        log_area = st.empty(); pct_text = st.empty(); progress_bar = st.progress(0.0)

        dur = ffprobe_duration(ffprobe, str(in1))
        has_a = has_audio_stream(ffprobe, str(in1))
        in_h, in_total_bps, in_vcodec, in_pix = ffprobe_height_bitrate_codec(ffprobe, str(in1))

        vcodec = codec_name(params["codec"])
        preset = params["preset"]
        scale_kernel = params["scale_kernel"]; fps_cap = params["fps_cap"]
        p1, full = single_filters(params["max_height"], dur, has_a, scale_kernel, fps_cap)
        input_specs = ["-i", str(in1)]

        # Fast path: stream copy if already small enough and no resize needed
        if "Size Cap" in params["encode_mode"]:
            target_total_bps, video_bps, _ = compute_bitrates(params["size_mb"], dur, params["audio_kbps"])
            already_small = (in_total_bps > 0 and in_total_bps <= target_total_bps)
            no_resize = (in_h <= params["max_height"])
            codec_ok = (in_vcodec in ("h264","hevc")) and (in_pix in ("yuv420p","yuvj420p"))
            if already_small and no_resize and codec_ok and not params["extra"]:
                st.info("Fast path: input already under target and no resize needed — stream copying…")
                cmd = [
                    params["ffmpeg"], "-y",
                    "-i", str(in1),
                    "-map", "0",
                    "-c", "copy",
                    "-movflags", "+faststart",
                    str(out_path)
                ]
                run_with_progress(cmd, max(dur,1.0), 0.0, 1.0, progress_bar, log_area, pct_text)
                progress_bar.progress(1.0); pct_text.write("**100%**")
                with open(out_path, "rb") as f:
                    st.success(f"Done (stream copy)! Output size: {out_path.stat().st_size/(1024*1024):.1f} MB")
                    st.download_button("⬇️ Download MP4", f, file_name=params["out_name"], mime="video/mp4")
                return

        # Encode path
        if "Size Cap" in params["encode_mode"]:
            _, video_bps, _ = compute_bitrates(params["size_mb"], dur, params["audio_kbps"])
            st.info(f"Duration: **{dur:.1f}s** • Video: **{human_bitrate(video_bps)}** • Audio: **{params['audio_kbps']} kbps**")
            encode_size_cap(params["ffmpeg"], vcodec, params["audio_kbps"], preset, video_bps,
                            input_specs, p1, full, str(out_path), dur, params["extra"], progress_bar, log_area, pct_text)
        else:
            eff_crf = max(0, min(51, params["crf"] + params["crf_bump"]))
            st.info(f"CRF: **{eff_crf}** (lower = higher quality & bigger file)")
            encode_crf(params["ffmpeg"], vcodec, params["audio_kbps"], preset, eff_crf,
                       input_specs, full, str(out_path), dur, params["extra"], progress_bar, log_area, pct_text)

        progress_bar.progress(1.0); pct_text.write("**100%**")
        with open(out_path, "rb") as f:
            st.success(f"Done! Output size: {out_path.stat().st_size/(1024*1024):.1f} MB")
            st.download_button("⬇️ Download MP4", f, file_name=params["out_name"], mime="video/mp4")

def merge_and_compress(ffmpeg: str, ffprobe: str, payloads: list, params: dict):
    """
    SINGLE-PASS MERGE: build one filter graph that normalizes both clips (scale/fps/audio),
    concat them, then encode once (two-pass for Size Cap, or single-pass CRF).
    This is much faster than pre-encoding intermediates.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        in1 = tmpdir / payloads[0]["name"]; in1.write_bytes(payloads[0]["data"])
        in2 = tmpdir / payloads[1]["name"]; in2.write_bytes(payloads[1]["data"])

        log_area = st.empty(); pct_text = st.empty(); progress_bar = st.progress(0.0)

        try:
            dur1 = ffprobe_duration(ffprobe, str(in1))
            dur2 = ffprobe_duration(ffprobe, str(in2))
            total_dur = dur1 + dur2
            has_a1 = has_audio_stream(ffprobe, str(in1))
            has_a2 = has_audio_stream(ffprobe, str(in2))
        except Exception as e:
            st.error(f"FFprobe failed: {e}")
            return

        H = int(params["max_height"])
        scale_kernel = params["scale_kernel"]; fps_cap = params["fps_cap"]
        fps_part = f",fps={fps_cap}" if fps_cap else ""

        def audio_node(idx, has_a, dur):
            return (f"[{idx}:a]aresample=async=1:first_pts=0,"
                    f"aformat=sample_rates=48000:channel_layouts=stereo,asetpts=N/SR/TB[a{idx}]") if has_a else \
                   (f"anullsrc=cl=stereo:r=48000,atrim=0:{dur:.3f},asetpts=N/SR/TB[a{idx}]")

        # Normalize both videos (scale + optional fps cap + format)
        v0 = f"[0:v]scale=-2:{H}:flags={scale_kernel}{fps_part},format=yuv420p,setpts=PTS-STARTPTS[v0]"
        v1 = f"[1:v]scale=-2:{H}:flags={scale_kernel}{fps_part},format=yuv420p,setpts=PTS-STARTPTS[v1]"
        a0 = audio_node(0, has_a1, dur1)
        a1 = audio_node(1, has_a2, dur2)

        # Two graphs: pass1 video-only (a=0) and full (a=1)
        pass1_filter = ";".join([v0, v1, "[v0][v1]concat=n=2:v=1:a=0[v]"])
        full_filter  = ";".join([v0, v1, a0, a1, "[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]"])

        final_path = tmpdir / params["out_name"]
        vcodec = codec_name(params["codec"])
        preset = params["preset"]

        if "Size Cap" in params["encode_mode"]:
            _, video_bps, _ = compute_bitrates(params["size_mb"], total_dur, params["audio_kbps"])
            st.info(f"Total duration: **{total_dur:.1f}s** • Video: **{human_bitrate(video_bps)}** • Audio: **{params['audio_kbps']} kbps**")
            passlog = str(final_path) + "_2passlog"

            # Pass 1 (video only)
            cmd1 = [
                params["ffmpeg"], "-y",
                "-i", str(in1), "-i", str(in2),
                "-filter_complex", pass1_filter,
                "-map", "[v]",
                "-c:v", vcodec, "-preset", preset,
                "-b:v", str(int(video_bps)), "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
                "-pass", "1", "-passlogfile", passlog,
                "-an", "-f", "mp4", os.devnull
            ] + (params["extra"] or [])
            # Pass 2 (video + audio)
            cmd2 = [
                params["ffmpeg"], "-y",
                "-i", str(in1), "-i", str(in2),
                "-filter_complex", full_filter,
                "-map", "[v]", "-map", "[a]",
                "-c:v", vcodec, "-preset", preset,
                "-b:v", str(int(video_bps)), "-maxrate", str(int(video_bps)), "-bufsize", str(int(video_bps*2)),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", f"{params['audio_kbps']}k",
                "-pass", "2", "-passlogfile", passlog,
                str(final_path)
            ] + (params["extra"] or [])
            run_with_progress(cmd1, total_dur, 0.00, 0.50, progress_bar, log_area, pct_text)
            run_with_progress(cmd2, total_dur, 0.50, 1.00, progress_bar, log_area, pct_text)

            for ext in (".log", "-0.log", "-0.log.mbtree", ".mbtree"):
                f = passlog + ext
                if os.path.exists(f):
                    try: os.remove(f)
                    except: pass

        else:
            eff_crf = max(0, min(51, params["crf"] + params["crf_bump"]))
            st.info(f"CRF: **{eff_crf}** (lower = higher quality & bigger file)")
            cmd = [
                params["ffmpeg"], "-y",
                "-i", str(in1), "-i", str(in2),
                "-filter_complex", full_filter,
                "-map", "[v]", "-map", "[a]",
                "-c:v", vcodec, "-preset", preset, "-crf", str(eff_crf),
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                "-c:a", "aac", "-b:a", f"{params['audio_kbps']}k",
                str(final_path)
            ] + (params["extra"] or [])
            run_with_progress(cmd, total_dur, 0.00, 1.00, progress_bar, log_area, pct_text)

        progress_bar.progress(1.0); pct_text.write("**100%**")
        with open(final_path, "rb") as f:
            st.success(f"Done! Output size: {final_path.stat().st_size/(1024*1024):.1f} MB")
            st.download_button("⬇️ Download MP4", f, file_name=params["out_name"], mime="video/mp4")

# ===== UI =====
if mode == "Compress 1 video":
    st.markdown("**Step 1.** Upload a single video file.")
    up = st.file_uploader(
        "Drop 1 file here",
        accept_multiple_files=False,
        type=["mp4","mov","mkv","webm","avi"],
        key="single_uploader"
    )
    st.markdown("**Step 2.** Choose your output strategy below.")
    size_mb = st.number_input("Target size (MB) — only for Size Cap", 10.0, 10000.0, 250.0, 10.0)
    crf = st.slider("CRF (18–28 typical) — only for CRF", 0, 51, 23)
    out_name = st.text_input("Output filename", value="compressed.mp4")

    # Map speed profile to real params
    preset_eff, scale_kernel, fps_cap, crf_bump = speed_params(speed_profile, preset_choice)

    if not st.session_state["busy_single"]:
        if st.button("▶️ Start", type="primary", disabled=(up is None)):
            if up is None:
                st.warning("Please upload a file.")
            else:
                st.session_state["single_payload"] = {"name": up.name, "data": up.getvalue()}
                st.session_state["run_params"] = {
                    "encode_mode": encode_mode, "codec": codec, "max_height": int(max_height),
                    "preset": preset_eff, "audio_kbps": int(audio_kbps), "size_mb": float(size_mb),
                    "crf": int(crf), "crf_bump": int(crf_bump),
                    "scale_kernel": scale_kernel, "fps_cap": fps_cap,
                    "out_name": out_name,
                    "ffmpeg": which_or(custom_ffmpeg, "ffmpeg"),
                    "extra": extra_flags.strip().split() if extra_flags.strip() else []
                }
                st.session_state["busy_single"] = True
                st.rerun()
    else:
        st.button("Processing…", disabled=True)
        try:
            ffprobe = which_or(custom_ffprobe, "ffprobe")
            compress_single(st.session_state["run_params"]["ffmpeg"], ffprobe,
                            st.session_state["single_payload"], st.session_state["run_params"])
        except Exception as e:
            if "Unknown encoder 'libx265'" in str(e):
                st.error("HEVC/H.265 (libx265) not available in this FFmpeg build. Switch to H.264.")
            else:
                st.error(f"Failed: {e}")
        finally:
            st.session_state["busy_single"] = False
            st.session_state["single_payload"] = None
            st.session_state["run_params"] = None

else:
    # ==== MERGE path in a FORM; submit is ALWAYS enabled ====
    with st.form("merge_form", clear_on_submit=False):
        st.markdown("**Step 1. Choose order explicitly**")
        col1, col2 = st.columns(2)
        with col1:
            up1 = st.file_uploader(
                "First video",
                accept_multiple_files=False,
                type=["mp4","mov","mkv","webm","avi"],
                key="merge_first",
            )
        with col2:
            up2 = st.file_uploader(
                "Second video",
                accept_multiple_files=False,
                type=["mp4","mov","mkv","webm","avi"],
                key="merge_second",
            )

        st.caption(f"First: {'✅' if up1 else '❌'}  |  Second: {'✅' if up2 else '❌'}")

        st.markdown("**Step 2.** Choose your output strategy below.")
        size_mb = st.number_input(
            "Target size (MB) — only for Size Cap",
            10.0, 10000.0, 250.0, 10.0,
            key="merge_size"
        )
        crf = st.slider("CRF (18–28 typical) — only for CRF", 0, 51, 23, key="merge_crf")
        out_name = st.text_input("Output filename", value="merged.mp4", key="merge_out")

        # Map speed profile here too
        preset_eff, scale_kernel, fps_cap, crf_bump = speed_params(speed_profile, preset_choice)

        submitted = st.form_submit_button("▶️ Start", type="primary")

    if submitted:
        if (up1 is None) or (up2 is None):
            st.warning("Please upload both files (First + Second) before pressing Start.")
        else:
            st.session_state["merge_payload"] = [
                {"name": up1.name, "data": up1.getvalue()},
                {"name": up2.name, "data": up2.getvalue()},
            ]
            st.session_state["run_params"] = {
                "encode_mode": encode_mode, "codec": codec, "max_height": int(max_height),
                "preset": preset_eff, "audio_kbps": int(audio_kbps), "size_mb": float(size_mb),
                "crf": int(crf), "crf_bump": int(crf_bump),
                "scale_kernel": scale_kernel, "fps_cap": fps_cap,
                "out_name": out_name,
                "ffmpeg": which_or(custom_ffmpeg, "ffmpeg"),
                "extra": extra_flags.strip().split() if extra_flags.strip() else []
            }
            st.session_state["busy_merge"] = True
            st.rerun()

    if st.session_state["busy_merge"]:
        st.button("Processing…", disabled=True)
        try:
            ffprobe = which_or(custom_ffprobe, "ffprobe")
            merge_and_compress(
                st.session_state["run_params"]["ffmpeg"],
                ffprobe,
                st.session_state["merge_payload"],
                st.session_state["run_params"]
            )
        except Exception as e:
            if "Unknown encoder 'libx265'" in str(e):
                st.error("HEVC/H.265 (libx265) not available in this FFmpeg build. Switch to H.264.")
            else:
                st.error(f"Failed: {e}")
        finally:
            st.session_state["busy_merge"] = False
            st.session_state["merge_payload"] = None
            st.session_state["run_params"] = None
