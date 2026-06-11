"""ros2 launch av_fusion_node pipeline.launch.py mode:=file media:=test.mp4

Arguments:
  mode               alsa (Mode A, ReSpeaker) | file (Mode B) | pyaudio
  media              .mp4 path, required when mode:=file
  config             pipeline.yaml override (default: repo config)
  enable_classifier  set false for DoA-only bring-up
  avfusion_src       path to the repo's src/ (exported as AVFUSION_SRC)
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="alsa"),
        DeclareLaunchArgument("media", default_value=""),
        DeclareLaunchArgument("config", default_value=""),
        DeclareLaunchArgument("enable_classifier", default_value="true"),
        DeclareLaunchArgument(
            "avfusion_src",
            default_value=os.path.expanduser("~/edge-av-fusion/src")),
        SetEnvironmentVariable("AVFUSION_SRC", LaunchConfiguration("avfusion_src")),
        Node(
            package="av_fusion_node",
            executable="av_fusion",
            name="av_fusion",
            output="screen",
            parameters=[{
                "mode": LaunchConfiguration("mode"),
                "media": LaunchConfiguration("media"),
                "config": LaunchConfiguration("config"),
                "enable_classifier": LaunchConfiguration("enable_classifier"),
            }],
        ),
    ])
