from glob import glob
import os
from setuptools import setup, find_packages

package_name = "robot_sim"

def get_model_data_files():
    data_files = []
    for root, dirs, files in os.walk("models"):
        if files:
            install_dir = os.path.join(f"share/{package_name}", root)
            file_paths = [os.path.join(root, f) for f in files]
            data_files.append((install_dir, file_paths))
    return data_files

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", [f for f in glob("launch/*.py") if os.path.isfile(f)]),
        (f"share/{package_name}/urdf", [f for f in glob("urdf/*") if os.path.isfile(f)]),
        (f"share/{package_name}/urdf/meshes", [f for f in glob("urdf/meshes/*") if os.path.isfile(f)]),
        (f"share/{package_name}/worlds", [f for f in glob("worlds/*") if os.path.isfile(f)]),
        (f"share/{package_name}/config", [f for f in glob("config/*") if os.path.isfile(f)]),
    ] + get_model_data_files(),
    install_requires=["setuptools"],
    entry_points={
        "console_scripts": [
            "sim_bridge = robot_sim.sim_bridge:main",
        ],
    },
)
