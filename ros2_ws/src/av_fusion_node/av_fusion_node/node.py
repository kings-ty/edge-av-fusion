"""rclpy node owning the edge-av-fusion pipeline (T4.4).

The node is a thin shell: it constructs an AudioSource (Mode A/B from params),
hands it to `avfusion.pipeline.Pipeline`, and converts FusionEvents into
AvDetection messages. All DSP/inference threading lives in the pipeline; rclpy
only contributes the executor thread (subscriptions, diagnostics timer).

Detections are published from the pipeline's DSP thread — rclpy publishers are
thread-safe — so a hop never waits on the executor.

Topics (names from config/pipeline.yaml `ros:` section):
  pub  /av_fusion/detections            av_fusion_interfaces/AvDetection (SensorDataQoS)
  pub  /av_fusion/diagnostics           diagnostic_msgs/DiagnosticArray (1 Hz)
  pub  /av_fusion/vision_request        std_msgs/Bool (edge-triggered by the FSM)
  sub  /av_fusion/vision_confirmation   std_msgs/Bool
  sub  /odom                            nav_msgs/Odometry (yaw for ambiguity fusion)
"""
import math
import os
import sys
import threading


def _ensure_avfusion_on_path() -> None:
    """The avfusion package lives in the repo, not the ROS install space.
    Also ensures the .venv site-packages are visible if running from system python."""
    src = os.environ.get("AVFUSION_SRC",
                         os.path.expanduser("~/edge-av-fusion/src"))
    repo_root = os.path.abspath(os.path.join(src, ".."))
    
    # Add src/ to path
    if src not in sys.path:
        sys.path.insert(0, src)
        
    # Add .venv site-packages to path if they exist
    venv_site = os.path.join(repo_root, ".venv/lib/python3.8/site-packages")
    if os.path.isdir(venv_site) and venv_site not in sys.path:
        sys.path.append(venv_site)

    try:
        import avfusion  # noqa: F401
    except ImportError:
        # Fallback if AVFUSION_SRC is not set/correct
        log_dir = os.path.dirname(os.path.abspath(__file__))
        # maybe we are in install/ or build/
        pass


_ensure_avfusion_on_path()

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

from av_fusion_interfaces.msg import AvDetection

from avfusion.config import load_config
from avfusion.pipeline import Pipeline, StageTimes
from avfusion.fusion.fsm import FusionEvent


def _yaw_deg_from_quaternion(q) -> float:
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.degrees(math.atan2(siny, cosy))


class AvFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("av_fusion")

        self.declare_parameter("mode", "alsa")          # alsa | file | pyaudio
        self.declare_parameter("media", "")             # .mp4 path for mode=file
        self.declare_parameter("config", "")            # pipeline.yaml override
        self.declare_parameter("enable_classifier", True)
        self.declare_parameter("with_video", False)     # mode=file: decode video too

        mode = self.get_parameter("mode").value
        media = self.get_parameter("media").value
        cfg_path = self.get_parameter("config").value or None
        self.cfg = load_config(cfg_path)

        source = self._make_source(mode, media)
        self._mode = mode
        self._eos_logged = False

        ros = self.cfg.ros
        self._pub_det = self.create_publisher(
            AvDetection, ros.detections_topic, qos_profile_sensor_data)
        self._pub_diag = self.create_publisher(
            DiagnosticArray, ros.diagnostics_topic, 10)
        self._pub_vision_req = self.create_publisher(
            Bool, ros.vision_topic + "_request", 10)
        self.create_subscription(
            Bool, ros.vision_topic, self._on_vision, 10)
        self.create_subscription(
            Odometry, ros.odom_topic, self._on_odom, qos_profile_sensor_data)

        self._pub_lock = threading.Lock()
        self.pipeline = Pipeline(
            self.cfg, source, event_cb=self._on_event,
            enable_classifier=bool(self.get_parameter("enable_classifier").value))
        self.pipeline.start()
        self.create_timer(1.0, self._publish_diagnostics)
        self.get_logger().info(
            "pipeline up: mode=%s degraded=%s -> %s"
            % (mode, self.pipeline.stats.degraded, ros.detections_topic))

    # ------------------------------------------------------------- sources
    def _make_source(self, mode: str, media: str):
        a = self.cfg.audio
        if mode == "alsa":
            from avfusion.audio.gst_alsa_source import GstAlsaSource
            return GstAlsaSource(a.alsa_device, a.sample_rate, a.channels,
                                 a.hop_samples, a.ring_capacity_hops)
        if mode == "file":
            if not media:
                # auto-pick from Edge-materials if present (relative to repo root)
                import os
                src_env = os.environ.get("AVFUSION_SRC", os.path.expanduser("~/edge-av-fusion/src"))
                repo_root = os.path.abspath(os.path.join(src_env, ".."))
                mat_dir = os.path.join(repo_root, "Edge-materials")
                
                if os.path.isdir(mat_dir):
                    clips = sorted([f for f in os.listdir(mat_dir) if f.endswith(".mp4")])
                    if clips:
                        media = os.path.join(mat_dir, clips[0])
                        self.get_logger().info("no media specified, defaulting to: %s" % media)

            if not media:
                raise ValueError("mode=file requires the 'media' parameter")
            from avfusion.audio.gst_file_source import GstFileSource
            return GstFileSource(
                media, a.sample_rate, a.channels, a.hop_samples,
                a.ring_capacity_hops,
                with_video=bool(self.get_parameter("with_video").value))
        if mode == "pyaudio":
            from avfusion.audio.pyaudio_source import PyAudioSource
            return PyAudioSource(a.sample_rate, a.channels, a.hop_samples,
                                 a.ring_capacity_hops)
        raise ValueError("unknown mode %r (alsa|file|pyaudio)" % mode)

    # ----------------------------------------------------------- callbacks
    def _on_vision(self, msg: Bool) -> None:
        self.pipeline.on_vision(bool(msg.data))

    def _on_odom(self, msg: Odometry) -> None:
        self.pipeline.update_yaw(
            _yaw_deg_from_quaternion(msg.pose.pose.orientation))

    def _on_event(self, event: FusionEvent, st: StageTimes) -> None:
        """Runs on the pipeline DSP thread, once per processed hop."""
        msg = AvDetection()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "mic_array"

        msg.sound_class = event.sound_class
        msg.confidence = float(event.confidence)
        msg.classifier_degraded = bool(self.pipeline.stats.degraded)

        tr = event.track
        if tr is not None:
            msg.doa_deg = float(tr.angle_deg)
            msg.doa_confidence = float(tr.confidence)
            msg.front_back_ambiguous = bool(tr.front_back_ambiguous)
            msg.front_log_odds = float(tr.front_log_odds)
            msg.track_id = int(tr.track_id)
        else:
            msg.doa_confidence = 0.0
            msg.front_back_ambiguous = True

        msg.fusion_state = event.state.value
        msg.vision_confirmed = bool(event.vision_confirmed)
        msg.alert = bool(event.alert)
        msg.pipeline_latency_ms = st.e2e_ns / 1e6

        with self._pub_lock:
            self._pub_det.publish(msg)
            if event.request_vision:
                self._pub_vision_req.publish(Bool(data=True))

    # --------------------------------------------------------- diagnostics
    def _publish_diagnostics(self) -> None:
        s = self.pipeline.stats
        status = DiagnosticStatus(
            name="av_fusion/pipeline",
            hardware_id=self._mode,
            level=DiagnosticStatus.WARN if s.degraded else DiagnosticStatus.OK,
            message="energy-trigger fallback" if s.degraded else "nominal")
        kv = [("hops", s.hops), ("dropped_hops", s.dropped_hops),
              ("classifications", s.classifications),
              ("fusion_state", self.pipeline.fsm.state.value)]
        if hasattr(self.pipeline.source, "xruns"):
            kv.append(("xruns", self.pipeline.source.xruns))
        status.values = [KeyValue(key=k, value=str(v)) for k, v in kv]

        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()
        arr.status = [status]
        self._pub_diag.publish(arr)

        if getattr(self.pipeline.source, "finished", False) and not self._eos_logged:
            self._eos_logged = True
            self.get_logger().info("Mode B media reached EOS; node stays up")

    def destroy_node(self) -> None:
        self.pipeline.stop()
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AvFusionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
