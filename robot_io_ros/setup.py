from catkin_pkg.python_setup import generate_distutils_setup
from distutils.core import setup

d = generate_distutils_setup(
    packages=['robot_io_ros'],
    package_dir={'': 'src'}
)

setup(**d)
