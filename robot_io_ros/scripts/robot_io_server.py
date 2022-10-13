#!/usr/bin/env python3
import rospy
import numpy as np
import pickle
import hydra
import os

from geometry_msgs.msg import Pose       as PoseMsg
from sensor_msgs.msg   import JointState as JointStateMsg
from std_msgs.msg      import UInt8MultiArray as UInt8MultiArrayMsg

from robot_io.robot_interface.base_robot_interface import BaseRobotInterface

from robot_io_ros.msg import Float64Array as Float64ArrayMsg

from robot_io_ros.srv import PoseGoal      as PoseGoalSrv, \
                             JointGoal     as JointGoalSrv, \
                             GripperAction as GripperActionSrv, \
                             PoseGoalRequest       as PoseGoalReqMsg,  \
                             PoseGoalResponse      as PoseGoalResMsg,  \
                             JointGoalRequest      as JointGoalReqMsg, \
                             JointGoalResponse     as JointGoalResMsg, \
                             GripperActionRequest  as GripperActionReqMsg,  \
                             GripperActionResponse as GripperActionResMsg

def point2np(point_msg):
    return np.asarray((point_msg.x, point_msg.y, point_msg.z))

def quat2np(quat_msg):
    return np.asarray((quat_msg.x, quat_msg.y, quat_msg.z, quat_msg.w))


class RobotServer(BaseRobotInterface):
    def __init__(self, robot, state_frequency=50):
        self.robot = robot

        self.pub_state = rospy.Publisher('~joint_state', JointStateMsg, 
                                                         queue_size=1, 
                                                         tcp_nodelay=True)

        self.pub_pickle_state = rospy.Publisher('~pickle_state', 
                                                 UInt8MultiArrayMsg, 
                                                 queue_size=1, 
                                                 tcp_nodelay=True)

        self.sub_move_async_cart_pos_abs_ptp = rospy.Subscriber('~move_async_cart_pos_abs_ptp', 
                                                                PoseMsg, 
                                                                callback=self.move_async_cart_pos_abs_ptp, 
                                                                queue_size=1)
        self.sub_move_async_cart_pos_abs_lin = rospy.Subscriber('~move_async_cart_pos_abs_lin', 
                                                                PoseMsg, 
                                                                callback=self.move_async_cart_pos_abs_lin, 
                                                                queue_size=1)
        self.sub_move_async_cart_pos_rel_ptp = rospy.Subscriber('~move_async_cart_pos_rel_ptp', 
                                                                PoseMsg, 
                                                                callback=self.move_async_cart_pos_rel_ptp, 
                                                                queue_size=1)
        self.sub_move_async_cart_pos_rel_lin = rospy.Subscriber('~move_async_cart_pos_rel_lin', 
                                                                PoseMsg, 
                                                                callback=self.move_async_cart_pos_rel_lin, 
                                                                queue_size=1)
        self.sub_move_async_joint_pos = rospy.Subscriber('~move_async_joint_pos', 
                                                         Float64ArrayMsg, 
                                                         callback=self.move_async_joint_pos, 
                                                         queue_size=1)
        self.sub_move_async_joint_vel = rospy.Subscriber('~move_async_joint_vel', 
                                                         Float64ArrayMsg, 
                                                         callback=self.move_async_joint_vel, 
                                                         queue_size=1)

        self.srv_move_cart_pos_abs_ptp = rospy.Service('~move_cart_pos_abs_ptp', 
                                                       PoseGoalSrv,
                                                       self.move_cart_pos_abs_ptp)
        self.srv_move_joint_pos = rospy.Service('~move_joint_pos', 
                                                JointGoalSrv,
                                                self.move_joint_pos)

        self.srv_move_to_neutral = rospy.Service('~move_to_neutral', 
                                                 PoseGoalSrv,
                                                 self.move_to_neutral)

        self.srv_abort_motion = rospy.Service('~abort_motion', 
                                              PoseGoalSrv,
                                              self.abort_motion)
        
        self.srv_close_gripper = rospy.Service('~close_gripper', 
                                               GripperActionSrv,
                                               self.close_gripper)
        
        self.srv_open_gripper = rospy.Service('~open_gripper', 
                                               GripperActionSrv,
                                               self.open_gripper)

        self._state_timer = rospy.Timer(rospy.Duration(1 / state_frequency), self._cb_state_tick)

    def _cb_state_tick(self, *args):
        state = self.robot.get_state()

        # print(state['gripper_opening_width'])
        
        pickle_state = UInt8MultiArrayMsg()
        pickle_state.data = pickle.dumps(state)
        self.pub_pickle_state.publish(pickle_state)

    def move_to_neutral(self, *args):
        """
        Move robot to initial position defined in robot conf. This method is blocking.
        """
        return PoseGoalResMsg(pickle.dumps(self.robot.move_to_neutral()))

    def move_cart_pos_abs_ptp(self, req : PoseGoalReqMsg):
        ret = self.robot.move_cart_pos_abs_ptp(point2np(req.tcp_goal.position),
                                               quat2np(req.tcp_goal.orientation))
        return PoseGoalResMsg(pickle.dumps(ret))

    def move_joint_pos(self, req : JointGoalReqMsg):
        ret = self.robot.move_joint_pos(req.joint_goal)

        return JointGoalResMsg(pickle.dumps(ret))

    def move_async_cart_pos_abs_ptp(self, pose_msg : PoseMsg):
        self.robot.move_async_cart_pos_abs_ptp(point2np(pose_msg.position),
                                               quat2np(pose_msg.orientation))

    def move_async_cart_pos_abs_lin(self, pose_msg : PoseMsg):
        self.robot.move_async_cart_pos_abs_lin(point2np(pose_msg.position),
                                               quat2np(pose_msg.orientation))

    def move_async_cart_pos_rel_ptp(self, pose_msg : PoseMsg):
        self.robot.move_async_cart_pos_rel_ptp(point2np(pose_msg.position),
                                               quat2np(pose_msg.orientation))

    def move_async_cart_pos_rel_lin(self, pose_msg : PoseMsg):
        self.robot.move_async_cart_pos_rel_lin(point2np(pose_msg.position),
                                               quat2np(pose_msg.orientation))

    def move_async_joint_pos(self, array_msg : Float64ArrayMsg):
        self.robot.move_async_joint_pos(array_msg.array)

    def move_async_joint_vel(self, array_msg : Float64ArrayMsg):
        self.robot.move_async_joint_vel(array_msg.array)

    def abort_motion(self, *args):
        return PoseGoalResMsg(pickle.dumps(self.robot.abort_motion()))

    def close_gripper(self, req : GripperActionReqMsg):
        print('recieved request to close gripper')
        ret = self.robot.close_gripper(req.blocking)
        pret = pickle.dumps(ret)
        return GripperActionResMsg(pret)

    def open_gripper(self, req : GripperActionReqMsg):
        print('recieved request to open gripper')
        ret = self.robot.open_gripper(req.blocking)
        return GripperActionResMsg(pickle.dumps(ret))


if __name__ == "__main__":
    rospy.init_node('robot_io_ros_server')

    config_path = rospy.get_param('~hydra_config_path')
    robot_path  = rospy.get_param('~hydra_robot_config')
    hydra_overrides = rospy.get_param('~hydra_overrides', [])

    hydra.initialize(config_path=config_path)
    cfg    = hydra.compose(robot_path, overrides=hydra_overrides)

    robot  = hydra.utils.instantiate(cfg.robot)
    print(f'TYPE OF ROBOT: {type(robot)}')

    server = RobotServer(robot, rospy.get_param('~state_frequency', 50.0))

