# Cameras

###Azure Kinect (Kinect 4)
- On Ubuntu 18 install azure kinect SDK with apt
- On Ubuntu 20 download libk4a*(-dev) and libk4abt*(-dev) from https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/libk/
  and k4atools from https://packages.microsoft.com/ubuntu/18.04/prod/pool/main/k/k4a-tools \
  Install with `sudo dpkg -i`

- Install Open3D in your Python env with `pip install open3d`

- For default usage, start `$ python robot_io/cams/kinect4/kinect4.py`

###RealSense SR300/SR305

First follow installation instructions for librealsense2 [here](https://github.com/IntelRealSense/librealsense)
```
pip install pyrealsense2
python robot_io/cams/realsense/realsenseSR300_librs2.py  # to test
```

###Framos D435e
- If `/usr/src/librealsense2` does not exist, download FRAMOS software package from
  https://www.framos.com/en/industrial-depth-cameras#downloads. Follow installation instructions, 
  make sure to use local admin user (e.g. xam2) to install (file system may NOT be network mounted).
  Copy `robot_io/cams/framos/setup_files/setup.py` to `/usr/src/librealsense2`.
- Get a local copy of framos librealsense2 in your Home directory.\
`$ cp -r /usr/src/librealense2 <PATH/IN/YOUR/HOME>`
- Uninstall existing installations of pyrealsense2 in your Python env.  
- Install package 
```
cd librealsense2
pip install -e .
```
- Ask Michael Keser to add your user account to the 'video' unix group. Otherwise the camera will not be recognized.
- Use Ethernet sockets on the ceiling for PoE. 


# KUKA iiwa

Clone Kuka Java Repository on aisgit
```git clone https://aisgit.informatik.uni-freiburg.de/hermannl/kuka_java_interface_no_ros```

`robot_io/robot_interface/iiwa_interface.py` is a ROS free python controller for KUKA iiwa. It sends UDP messages to the iiwa_java_controller.

Supported control modes:
- Joint position control
    - degrees / radians
- Cartesian position control
    - PTP / LIN motions
    - with / without impedance
    - absolute / relative coordinates

For more information please read README of [kuka_java_interface_no_ros](https://aisgit.informatik.uni-freiburg.de/hermannl/kuka_java_interface_no_ros)

# Install SpaceMouse
```
sudo apt install libspnav-dev spacenavd
conda activate robot
pip install spnav
```

Next test if it works, some common pitfalls are:
1. Turn on SpaceMouse in the back
2. May not work while charging.
3. Wireless range is quite limited.
4. Comment the following two lines in `site-packages/spnav/__init__.py`
```
#pythonapi.PyCObject_AsVoidPtr.restype = c_void_p
#pythonapi.PyCObject_AsVoidPtr.argtypes = [py_object]
```

To test execute the following program. When moving the mouse you should
see numbers scrolling by.
```
python robot_io/input_devices/space_mouse.py
```

# Franka Emika Panda

### IK fast
IK fast is an analytic IK solver. In order to use IK fast, first install `ikfast-pybind`:
```
git clone --recursive https://github.com/yijiangh/ikfast_pybind
cd ikfast_pybind
# copy panda IK solution .cpp and .h to ikfast_pybind
cp <PATH>/robot_io/robot_io/panda_control/src/ikfast.h ./src/franka_panda/
cp <PATH>/robot_io/robot_io/panda_control/src/ikfast0x10000049.Transform6D.0_1_2_3_4_5_f6.cpp ./src/franka_panda/ 
pip install .
```
For creating different IK solutions (e.g. in case of a different gripper) please refer to: 
`http://docs.ros.org/en/kinetic/api/framefab_irb6600_support/html/doc/ikfast_tutorial.html`

# VR Teleoperation

### Install Steam
### Install SteamVR
- In terminal run `$ steam`, it will start downloading an update and create a `.steam` folder in your home directory.
- In Steam, create user account or use existing account.
- Install SteamVR
  - If on `pickup` click `Steam -> Settings -> Downloads -> Steam Library Folders -> Add Library Folder -> /media/hdd/SteamLibrary` to add the existing installation of SteamVR to your Steam account
  - Otherwise download SteamVR
- Restart Steam
- Connect and turn on HTC VIVE
- Launch `Library -> SteamVR` (if not shown, check `[] Tools` box)
- If SteamVR throws an  `Error: setcap of vrcompositor-launcher failed`, run `$ sudo setcap CAP_SYS_NICE+ep /media/hdd/SteamLibrary/steamapps/common/SteamVR/bin/linux64/vrcompositor-launcher`
- Make sure Headset and controller are correctly detected, go through VR setup procedure
### Install Bullet
```
$ git clone https://github.com/bulletphysics/bullet3.git
$ cd bullet3

# Optional: patch bullet for selecting correct rendering device
# (only relevant when using EGL and multi-gpu training)
$ wget https://raw.githubusercontent.com/BlGene/bullet3/egl_remove_works/examples/OpenGLWindow/EGLOpenGLWindow.cpp -O examples/OpenGLWindow/EGLOpenGLWindow 

# For building Bullet for VR  add -DUSE_OPENVR=ON to line 8 of build_cmake_pybullet_double.sh
# Run
$ ./build_cmake_pybullet_double.sh

$ pip install numpy  # important to have numpy installed before installing bullet
$ pip install -e .  # effectively this is building bullet a second time, but importing is easier when installing with pip

# add alias to your bashrc
alias bullet_vr="~/.steam/steam/ubuntu12_32/steam-runtime/run.sh </PATH/TO/BULLET/>bullet3/build_cmake/examples/SharedMemory/App_PhysicsServer_SharedMemory_VR"
```

### Marker Detector

```
$ cd robot_io/marker_detection/apriltag_detection
$ python setupBatch.py build_ext --inplace
```