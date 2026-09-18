from pathlib import Path
import queue
import shutil
import subprocess
import threading

import numpy as np
from pyglet.gl import (
    GL_DEPTH_TEST,
    GL_ENABLE_BIT,
    GL_POINTS,
    GL_RGB,
    GL_TRIANGLES,
    GL_UNSIGNED_BYTE,
    GLubyte,
    glClearColor,
    glDisable,
    glPopAttrib,
    glPushAttrib,
    glReadPixels,
)


DRAW_SCALE = 50.0
TRACK_LAYER = 5
TRAJECTORY_LAYER = 20
VEHICLE_LAYER = 100


def ordered_group(order):
    import pyglet.graphics

    return pyglet.graphics.OrderedGroup(order)


def top_overlay_group(order):
    import pyglet.graphics

    class TopOverlayGroup(pyglet.graphics.OrderedGroup):
        def set_state(self):
            glPushAttrib(GL_ENABLE_BIT)
            glDisable(GL_DEPTH_TEST)

        def unset_state(self):
            glPopAttrib()

    return TopOverlayGroup(order)


def set_background_color(e, color=(1.0, 1.0, 1.0, 1.0), label_color=(0, 0, 0, 255)):
    glClearColor(*color)
    if hasattr(e, "score_label"):
        e.score_label.color = label_color


def draw_point(e, point, colour, layer=TRAJECTORY_LAYER):
    scaled_point = DRAW_SCALE * point
    ret = e.batch.add(
        1,
        GL_POINTS,
        ordered_group(layer),
        ('v3f/stream', [scaled_point[0], scaled_point[1], 0]),
        ('c3B/stream', colour),
    )
    return ret


def draw_polyline(e, points, colour, width=16.0, closed=False, layer=TRAJECTORY_LAYER):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2:
        return None

    scaled_points = DRAW_SCALE * points[:, :2]
    starts = scaled_points
    ends = np.roll(scaled_points, -1, axis=0) if closed else scaled_points[1:]
    starts = starts if closed else starts[:-1]

    vertices = []
    for start, end in zip(starts, ends):
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-9:
            continue

        normal = np.array([-direction[1], direction[0]]) / length * (width / 2.0)
        p1 = start + normal
        p2 = start - normal
        p3 = end - normal
        p4 = end + normal
        vertices.extend([
            p1[0], p1[1], 0.0,
            p2[0], p2[1], 0.0,
            p3[0], p3[1], 0.0,
            p1[0], p1[1], 0.0,
            p3[0], p3[1], 0.0,
            p4[0], p4[1], 0.0,
        ])

    if not vertices:
        return None

    vertex_count = len(vertices) // 3
    colours = colour * vertex_count
    return e.batch.add(
        vertex_count,
        GL_TRIANGLES,
        ordered_group(layer),
        ('v3f/stream', vertices),
        ('c3B/stream', colours),
    )


class DrawDebug:
    def __init__(self):
        self.reference_traj_show = np.array([[0, 0]])
        self.predicted_traj_show = np.array([[0, 0]])
        self.dyn_obj_drawn = []
        self.static_obj_drawn = []
        self.f = 0

    def draw_track_once(self, e, waypoints):
        if self.static_obj_drawn:
            return
        track_line = draw_polyline(e, waypoints, [183, 193, 222], width=14.0, closed=True, layer=TRACK_LAYER)
        if track_line is not None:
            self.static_obj_drawn.append(track_line)

    def draw_debug(self, e):
        while len(self.dyn_obj_drawn) > 0:
            if self.dyn_obj_drawn[0] is not None:
                self.dyn_obj_drawn[0].delete()
            self.dyn_obj_drawn.pop(0)

        reference_line = draw_polyline(e, self.reference_traj_show, [255, 0, 0], width=20.0, layer=TRAJECTORY_LAYER)
        predicted_line = draw_polyline(e, self.predicted_traj_show, [0, 255, 0], width=20.0, layer=TRAJECTORY_LAYER)

        if reference_line is not None:
            self.dyn_obj_drawn.append(reference_line)
        else:
            for p in self.reference_traj_show:
                self.dyn_obj_drawn.append(draw_point(e, p, [255, 0, 0], layer=TRAJECTORY_LAYER))

        if predicted_line is not None:
            self.dyn_obj_drawn.append(predicted_line)
        else:
            for p in self.predicted_traj_show:
                self.dyn_obj_drawn.append(draw_point(e, p, [0, 255, 0], layer=TRAJECTORY_LAYER))


class VehicleImageOverlay:
    def __init__(self, image_path, rotation_offset_deg=0.0, length=5.8, width=3.1):
        self.image_path = Path(image_path) if image_path else None
        self.rotation_offset_deg = rotation_offset_deg
        self.length = length
        self.width = width
        self.sprite = None
        self._warned_missing = False

    def update(self, e, agent_idx=0):
        if self.image_path is None:
            return
        if not self.image_path.exists():
            if not self._warned_missing:
                print(f"Vehicle image not found: {self.image_path}. Drawing default box.")
                self._warned_missing = True
            return
        if not hasattr(e, "poses") or e.poses is None or len(e.poses) <= agent_idx:
            return

        if self.sprite is None:
            import pyglet
            import pyglet.graphics
            import pyglet.image
            import pyglet.sprite

            image = pyglet.image.load(str(self.image_path))
            image.anchor_x = image.width // 2
            image.anchor_y = image.height // 2
            self.sprite = pyglet.sprite.Sprite(image, batch=e.batch, group=top_overlay_group(VEHICLE_LAYER))
            self.sprite.scale_x = DRAW_SCALE * self.length / image.width
            self.sprite.scale_y = DRAW_SCALE * self.width / image.height

        pose = e.poses[agent_idx]
        self.sprite.x = DRAW_SCALE * pose[0]
        self.sprite.y = DRAW_SCALE * pose[1]
        self.sprite.rotation = -np.degrees(pose[2]) + self.rotation_offset_deg
        self._hide_default_box(e, agent_idx)

    def _hide_default_box(self, e, agent_idx):
        if not hasattr(e, "cars") or len(e.cars) <= agent_idx:
            return
        e.cars[agent_idx].vertices = [0.0] * len(e.cars[agent_idx].vertices)


class VideoRecorder:
    def __init__(self, output_path, fps=25, capture_every=1, enabled=True):
        self.output_path = Path(output_path)
        self.fps = fps
        self.capture_every = max(1, int(capture_every))
        self.enabled = enabled
        self.frame_count = 0
        self.writer = None
        self.ffmpeg_process = None
        self.frame_queue = None
        self.writer_thread = None
        self.dropped_frames = 0
        self.video_width = None
        self.video_height = None
        self.read_width = None
        self.read_height = None
        self.read_buffer = None
        self.renderer = None
        self._hook_installed = False

    def update(self, e):
        if not self.enabled or self._hook_installed:
            return
        self.renderer = e

        original_on_draw = e.on_draw

        def on_draw_with_video_capture():
            original_on_draw()
            self.capture_frame()

        e.on_draw = on_draw_with_video_capture
        self._hook_installed = True

    def capture_frame(self):
        if not self.enabled:
            return
        self.frame_count += 1
        if self.frame_count % self.capture_every != 0:
            return

        frame = self._read_frame()

        self._ensure_writer(frame)
        if self.ffmpeg_process is not None:
            frame = frame[:self.video_height, :self.video_width]
            try:
                self.frame_queue.put_nowait(frame.tobytes())
            except queue.Full:
                self.dropped_frames += 1
        else:
            self.writer.append_data(frame)

    def close(self):
        if self.ffmpeg_process is not None:
            self.frame_queue.put(None)
            self.writer_thread.join()
            self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
            self.ffmpeg_process = None
            print(f"Video saved to {self.output_path}")
            if self.dropped_frames:
                print(f"Dropped {self.dropped_frames} video frames to keep the simulation responsive.")
            return
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        print(f"Video saved to {self.output_path}")

    def _ensure_writer(self, frame):
        if self.writer is not None or self.ffmpeg_process is not None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg_path = self._find_ffmpeg()
        if self.output_path.suffix.lower() == ".mp4" and ffmpeg_path:
            self._open_ffmpeg_writer(frame, ffmpeg_path)
            return

        self._open_imageio_writer()

    def _read_frame(self):
        import pyglet

        color_buffer = pyglet.image.get_buffer_manager().get_color_buffer()
        width, height = color_buffer.width, color_buffer.height
        if self.read_buffer is None or width != self.read_width or height != self.read_height:
            self.read_width = width
            self.read_height = height
            self.read_buffer = (GLubyte * (width * height * 3))()

        glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, self.read_buffer)
        frame = np.frombuffer(self.read_buffer, dtype=np.uint8).reshape((height, width, 3))
        return np.flipud(frame)

    def _find_ffmpeg(self):
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path:
            return ffmpeg_path
        for candidate in ("/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"):
            if Path(candidate).exists():
                return candidate
        return None

    def _open_ffmpeg_writer(self, frame, ffmpeg_path):
        height, width = frame.shape[:2]
        width -= width % 2
        height -= height % 2
        self.video_width = width
        self.video_height = height

        cmd = [
            ffmpeg_path,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-preset", "ultrafast",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            str(self.output_path),
        ]
        self.ffmpeg_process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.frame_queue = queue.Queue(maxsize=128)
        self.writer_thread = threading.Thread(target=self._write_ffmpeg_frames, daemon=True)
        self.writer_thread.start()
        print(f"Recording video with ffmpeg to {self.output_path}")

    def _write_ffmpeg_frames(self):
        while True:
            frame_bytes = self.frame_queue.get()
            if frame_bytes is None:
                return
            self.ffmpeg_process.stdin.write(frame_bytes)

    def _open_imageio_writer(self):
        import imageio.v2 as imageio

        try:
            self.writer = imageio.get_writer(str(self.output_path), mode="I", fps=self.fps)
            print(f"Recording video with imageio to {self.output_path}")
        except Exception as exc:
            if self.output_path.suffix.lower() == ".gif":
                raise
            fallback_path = self.output_path.with_suffix(".gif")
            print(f"Could not open video writer for {self.output_path}: {exc}")
            print(f"Falling back to {fallback_path}")
            self.output_path = fallback_path
            self.writer = imageio.get_writer(str(self.output_path), mode="I", fps=self.fps)
