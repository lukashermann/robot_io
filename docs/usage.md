# Camera Calibration
### Static Camera

### Gripper Camera
- Place Aruco Marker in front of robot
- Run `python robot_io/calibration/gripper_cam_calibration.py --config-name=[panda_calibrate_gripper_cam|kuka_calibrate_gripper_cam]`

------------------

### Teleoperation
Make sure to set workspace limits appropriately in `robot_io/conf/robot/<robot_interface.yaml>
```
$ python robot_io/control/teleop_robot.py --config-name=[panda_teleop|kuka_teleop]
```