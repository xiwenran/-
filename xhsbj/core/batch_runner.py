import os
from typing import List, Optional, Tuple

from PyQt6.QtCore import QThread, pyqtSignal

from PIL import Image

from models.template_model import Template
from core.image_processor import embed_image_pil_fast, precompute_template_cache

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv"}


def get_image_files(folder: str):
    files = []
    for fn in sorted(os.listdir(folder)):
        if os.path.splitext(fn)[1].lower() in IMAGE_EXTS:
            files.append(os.path.join(folder, fn))
    return files


class BatchRunner(QThread):
    progress = pyqtSignal(int, int, str)   # done, total, status_msg
    finished = pyqtSignal(bool, str)       # success, message

    def __init__(
        self,
        tasks,               # List of (group_name: str, file_list: List[str], templates: List[Template])
        output_dir: str,
        output_format: str = "PNG",   # "PNG" or "JPEG"
        parent=None,
    ):
        super().__init__(parent)
        self.tasks = tasks
        self.output_dir = output_dir
        self.output_format = output_format
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        try:
            os.makedirs(self.output_dir, exist_ok=True)

            total = sum(len(files) * len(templates) for _, files, templates in self.tasks)
            done = 0

            for group_name, files, templates in self.tasks:
                for template in templates:
                    if self._abort:
                        self.finished.emit(False, "已取消"); return

                    out_sub = os.path.join(self.output_dir, group_name, template.name)
                    os.makedirs(out_sub, exist_ok=True)

                    # Use template's configured output size (0 = auto/background size)
                    output_size = (
                        (template.output_width, template.output_height)
                        if template.output_width > 0 else None
                    )

                    # Precompute mask + bg array once per template (shared across all files)
                    bg_img = Image.open(template.background_path)
                    cache = precompute_template_cache(bg_img, template.screen_points)

                    for i, img_path in enumerate(files, 1):
                        if self._abort:
                            self.finished.emit(False, "已取消"); return

                        ext = ".jpg" if self.output_format == "JPEG" else ".png"
                        out_path = os.path.join(out_sub, f"{i}{ext}")

                        ppt_img = Image.open(img_path)
                        result = embed_image_pil_fast(ppt_img, cache)

                        if output_size:
                            result = result.resize(output_size, Image.LANCZOS)

                        if self.output_format == "JPEG":
                            result = result.convert("RGB")
                            result.save(out_path, "JPEG", quality=95)
                        else:
                            result.save(out_path, "PNG")

                        done += 1
                        self.progress.emit(done, total, f"{group_name}/{template.name}/{i}{ext}")

            self.finished.emit(True, f"完成！共处理 {done} 张图片")

        except Exception as e:
            import traceback
            self.finished.emit(False, f"错误: {str(e)}\n{traceback.format_exc()}")


class VideoRunner(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(bool, str)

    def __init__(self, tasks, output_dir, parent=None):
        """
        tasks: List of (video_path: str, templates: List[Template])
        Each video frame is treated as PPT content; template's background is the scene.
        Audio is preserved via PyAV (no external ffmpeg needed).
        """
        super().__init__(parent)
        self.tasks = tasks
        self.output_dir = output_dir
        self._abort = False

    def abort(self):
        self._abort = True

    def run(self):
        import av

        try:
            os.makedirs(self.output_dir, exist_ok=True)

            # Pre-scan frame counts
            meta = []
            total = 0
            for video_path, templates in self.tasks:
                with av.open(video_path) as c:
                    vs = c.streams.video[0]
                    n = vs.frames if vs.frames else 0
                    fps = float(vs.average_rate or 25)
                meta.append((n, fps))
                total += max(n, 1) * len(templates)

            done = 0
            for (video_path, templates), (n_frames, fps) in zip(self.tasks, meta):
                if self._abort:
                    self.finished.emit(False, "已取消"); return

                vid_name = os.path.splitext(os.path.basename(video_path))[0]

                for template in templates:
                    if self._abort:
                        self.finished.emit(False, "已取消"); return

                    # Precompute mask + bg array once per template (reused for every frame)
                    bg_img = Image.open(template.background_path).convert("RGBA")
                    cache = precompute_template_cache(bg_img, template.screen_points)
                    bg_w, bg_h = cache["bg_size"]

                    out_dir = os.path.join(self.output_dir, vid_name, template.name)
                    os.makedirs(out_dir, exist_ok=True)
                    out_path = os.path.join(out_dir, f"{vid_name}.mp4")

                    with av.open(video_path) as inp, av.open(out_path, "w", format="mp4") as outp:
                        in_vs = inp.streams.video[0]

                        # Output video stream (H.264)
                        out_vs = outp.add_stream("libx264", rate=in_vs.average_rate)
                        out_vs.width = bg_w
                        out_vs.height = bg_h
                        out_vs.pix_fmt = "yuv420p"
                        out_vs.options = {"crf": "18", "preset": "veryfast"}

                        # Output audio streams (AAC re-encode) + resamplers for format conversion
                        out_as_list = []
                        resamplers = []
                        for in_as in inp.streams.audio:
                            sr = in_as.codec_context.sample_rate or 44100
                            layout = in_as.codec_context.layout or "stereo"
                            out_as = outp.add_stream("aac", rate=sr)
                            resampler = av.AudioResampler(
                                format="fltp", layout=layout, rate=sr
                            )
                            out_as_list.append(out_as)
                            resamplers.append(resampler)

                        streams = [in_vs] + list(inp.streams.audio)
                        in_audio_list = list(inp.streams.audio)
                        frame_i = 0
                        # Per-audio-stream sample counter for correct PTS
                        audio_pts = [0] * len(out_as_list)

                        for packet in inp.demux(*streams):
                            if self._abort:
                                self.finished.emit(False, "已取消"); return
                            if packet.dts is None:
                                continue

                            if packet.stream == in_vs:
                                for frame in packet.decode():
                                    pil = frame.to_image().convert("RGBA")
                                    result = embed_image_pil_fast(pil, cache)
                                    out_frame = av.VideoFrame.from_image(result.convert("RGB"))
                                    # Use sequential frame counter; out_vs.codec_context.time_base
                                    # is 1/fps so pts=frame_i gives correct duration.
                                    out_frame.pts = frame_i
                                    for p in out_vs.encode(out_frame):
                                        outp.mux(p)
                                    frame_i += 1
                                    done += 1
                                    if frame_i % 30 == 0 or frame_i == 1:
                                        self.progress.emit(done, total,
                                            f"{vid_name}/{template.name}  {frame_i}/{n_frames} 帧")
                            elif packet.stream in in_audio_list:
                                idx = in_audio_list.index(packet.stream)
                                if idx < len(out_as_list):
                                    for frame in packet.decode():
                                        for resampled in resamplers[idx].resample(frame):
                                            # Use sample count as PTS (audio time_base = 1/sample_rate)
                                            resampled.pts = audio_pts[idx]
                                            audio_pts[idx] += resampled.samples
                                            for p in out_as_list[idx].encode(resampled):
                                                outp.mux(p)

                        # Flush video encoder
                        for p in out_vs.encode():
                            outp.mux(p)
                        # Flush audio resamplers and encoders
                        for i, (out_as, resampler) in enumerate(zip(out_as_list, resamplers)):
                            for resampled in resampler.resample(None):
                                resampled.pts = audio_pts[i]
                                audio_pts[i] += resampled.samples
                                for p in out_as.encode(resampled):
                                    outp.mux(p)
                            for p in out_as.encode():
                                outp.mux(p)

                    self.progress.emit(done, total, f"✓ {vid_name}/{template.name}.mp4")

            self.finished.emit(True, f"完成！共处理 {done} 帧")

        except Exception as e:
            import traceback
            self.finished.emit(False, f"错误: {str(e)}\n{traceback.format_exc()}")
