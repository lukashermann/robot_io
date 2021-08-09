# Camera Calibration
### Static Camera
- Stick the marker to the robot end-effector
- Run `python robot_io/calibration/static_cam_calibration.py --config-name=[panda_calibrate_static_cam|iiwa_calibrate_static_cam]`
- If you set `record_traj: true`, then you should use vr controller to move the robot. Press the record button (on top) to sample poses, and hold record button to finish the pose sampling.
- If you set `record_traj: false` and `play_traj: true`, the robot will move to the previously recorded poses and captures the marker pose. This option is helpful in case the camera is moved slightly.

### Gripper Camera
- Place Aruco Marker in front of robot
- Run `python robot_io/calibration/gripper_cam_calibration.py --config-name=[panda_calibrate_gripper_cam|kuka_calibrate_gripper_cam]`

------------------

### Teleoperation
Make sure to set workspace limits appropriately in `robot_io/conf/robot/<robot_interface.yaml>
```
$ python robot_io/control/teleop_robot.py --config-name=[panda_teleop|kuka_teleop]
```