import rospy
import pickle
import numpy as np

from multiprocessing import RLock

from geometry_msgs.msg import Pose       as PoseMsg, \
                              Point      as PointMsg, \
                              Quaternion as QuaternionMsg
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

from robot_io.utils.utils import pos_orn_to_matrix, get_git_root, ReferenceType

def np2point(point):
    return PointMsg(point[0], point[1], point[2])

def np2quat(quat):
    return QuaternionMsg(quat[0], quat[1], quat[2], quat[3])

def np2pose(point, quat):
    return PoseMsg(np2point(point, np2quat(quat)))


class RobotClient(BaseRobotInterface):
    def __init__(self, server_name):

        # self.sub_state = rospy.Subscriber(f'{server_name}/joint_state', 
        #                                    JointStateMsg,
        #                                    callback=self._cb_state,
        #                                    queue_size=1)
        
        self.pub_pickle_state = rospy.Subscriber(f'{server_name}/pickle_state', 
                                                 UInt8MultiArrayMsg, 
                                                 callback=self._cb_pickle_state,
                                                 queue_size=1)

        self._state_lock = RLock()
        self._last_state = None

        self.pub_move_async_cart_pos_abs_ptp = rospy.Publisher(f'{server_name}/move_async_cart_pos_abs_ptp', 
                                                                PoseMsg, 
                                                                queue_size=1, 
                                                                tcp_nodelay=True)
        self.pub_move_async_cart_pos_abs_lin = rospy.Publisher(f'{server_name}/move_async_cart_pos_abs_lin', 
                                                                PoseMsg, 
                                                                queue_size=1, 
                                                                tcp_nodelay=True)
        self.pub_move_async_cart_pos_rel_ptp = rospy.Publisher(f'{server_name}/move_async_cart_pos_rel_ptp', 
                                                                PoseMsg, 
                                                                queue_size=1, 
                                                                tcp_nodelay=True)
        self.pub_move_async_cart_pos_rel_lin = rospy.Publisher(f'{server_name}/move_async_cart_pos_rel_lin', 
                                                                PoseMsg, 
                                                                queue_size=1, 
                                                                tcp_nodelay=True)
        self.pub_move_async_joint_pos = rospy.Publisher(f'{server_name}/move_async_joint_pos', 
                                                         Float64ArrayMsg, 
                                                         queue_size=1, 
                                                         tcp_nodelay=True)
        self.pub_move_async_joint_vel = rospy.Publisher(f'{server_name}/move_async_joint_vel', 
                                                         Float64ArrayMsg, 
                                                         queue_size=1, 
                                                         tcp_nodelay=True)

        self.srv_move_cart_pos_abs_ptp = rospy.ServiceProxy(f'{server_name}/move_cart_pos_abs_ptp', 
                                                            PoseGoalSrv)
        self.srv_move_joint_pos = rospy.ServiceProxy(f'{server_name}/move_joint_pos', 
                                                     JointGoalSrv)

        self.srv_move_to_neutral = rospy.ServiceProxy(f'{server_name}/move_to_neutral', 
                                                      PoseGoalSrv)

        self.srv_abort_motion = rospy.ServiceProxy(f'{server_name}/abort_motion', 
                                                   PoseGoalSrv)
        
        self.srv_close_gripper = rospy.ServiceProxy(f'{server_name}/close_gripper', 
                                                    GripperActionSrv)
        
        self.srv_open_gripper = rospy.ServiceProxy(f'{server_name}/open_gripper', 
                                                   GripperActionSrv)

    # def _cb_state(self, msg : JointStateMsg):
    #     raise NotImplementedError

    def _cb_pickle_state(self, msg : UInt8MultiArrayMsg):
        state = pickle.loads(msg.data)
        
        with self._state_lock:
            self._last_state = state

    def get_state(self):
        with self._state_lock:
            return self._last_state

    def get_tcp_pose(self):
        with self._state_lock:
            return pos_orn_to_matrix(self._last_state['tcp_pos'],
                                     self._last_state['tcp_orn'])

    def get_tcp_pos_orn(self):
        return np.array(self._last_state['tcp_pos']), np.array(self._last_state['tcp_orn'])

    def move_to_neutral(self, *args):
        """
        Move robot to initial position defined in robot conf. This method is blocking.
        """
        return pickle.loads(self.srv_move_to_neutral(PoseMsg()))

    def move_cart_pos_abs_ptp(self, target_pos, target_orn):
        return pickle.loads(self.srv_move_cart_pos_abs_ptp(np2pose(target_pos, target_orn)))

    def move_joint_pos(self, joint_positions):
        return pickle.loads(self.srv_move_joint_pos(list(joint_positions)))

    def move_async_cart_pos_abs_ptp(self, target_pos, target_orn):
        self.pub_move_async_cart_pos_abs_ptp(np2pose(target_pos, target_orn))

    def move_async_cart_pos_abs_lin(self, target_pos, target_orn):
        self.pub_move_async_cart_pos_abs_lin(np2pose(target_pos, target_orn))

    def move_async_cart_pos_rel_ptp(self, rel_target_pos, rel_target_orn):
        self.pub_move_async_cart_pos_rel_ptp(np2pose(rel_target_pos, rel_target_orn))

    def move_async_cart_pos_rel_lin(self, rel_target_pos, rel_target_orn):
        self.pub_move_async_cart_pos_rel_lin(np2pose(rel_target_pos, rel_target_orn))

    def move_async_joint_pos(self, joint_positions):
        self.pub_move_async_joint_pos.publish(Float64ArrayMsg(joint_positions))

    def move_async_joint_vel(self, joint_velocities):
        self.pub_move_async_joint_vel.publish(Float64ArrayMsg(joint_velocities))

    def abort_motion(self, *args):
        return pickle.loads(self.srv_abort_motion())

    def close_gripper(self, blocking=False):
        ret = self.srv_close_gripper(blocking)
        return pickle.loads(ret)

    def open_gripper(self, blocking=False):
        ret = self.srv_open_gripper(blocking)
        return pickle.loads(ret)

if __name__ == '__main__':
    rospy.init_node('robot_io_ros_client')

    client = RobotClient('panda_ros_server')

    while not rospy.is_shutdown():
        print(client.get_state())
        rospy.sleep(0.05)
