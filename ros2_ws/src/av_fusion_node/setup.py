from setuptools import setup

package_name = "av_fusion_node"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages",
         ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/pipeline.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Taeyeong Lee",
    maintainer_email="ttae.yeong.lee@gmail.com",
    description="rclpy node running the edge-av-fusion perception pipeline.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "av_fusion = av_fusion_node.node:main",
        ],
    },
)
