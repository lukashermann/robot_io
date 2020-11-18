# Cameras

###Azure Kinect (Kinect 4)

- Follow installation guide: http://www.open3d.org/docs/release/tutorial/Basic/azure_kinect.html

- For default usage, start `$ python robot_io/cams/kinect4.py`

###RealSense SR300/SR305

- Follow installation instructions for librealsense2: (librealsense 1 is outdated) https://github.com/IntelRealSense/librealsense \
```$ pip install pyrealsense2```

- For default usage, start `$ python robot_io/cams/realsenseSR300_librs2.py` 

###Framos D435e
- Get a local copy of framos librealsense2 \
`$ cp -r /usr/src/librealense2 <PATH/IN/YOUR/HOME>`
- Install package 
```
cd librealsense2
$ pip install -e .
```

# KUKA iiwa

Clone Kuka Java Repository on aisgit
```git clone https://aisgit.informatik.uni-freiburg.de/hermannl/kuka_java_interface_no_ros```

`robot_io/kuka_iiwa/iiwa_controller.py` is a ROS free python controller for KUKA iiwa. It sends UDP messages to the iiwa_java_controller.
Supported control modes:
- Joint position control
    - degrees / radians
- Cartesian position control
    - PTP / LIN motions
    - with / without impedance
    - absolute / relative coordinates
    
# Install SpaceMouse
```
sudo apt install libspnav spacenavd # don't need libspnav-dev?
conda activate bullet
pip install spnav
```

Next test if it works, some common pitfalls are:
1. Turn on SpaceMouse in the back
2. May not work while charging.
3. Wireless range is quite limited.

To test execute the following program:
```
cd robot_io/input_devices
python space_mouse.py
# move the mouse and you should see number scrolling by
```

    