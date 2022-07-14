import logging
import socket
import time

from robot_io.utils.utils import timeit

commands = dict(id=("ID?", str),
				serial_number=("SN?", str),
				label=("LABEL?", str),
				firmware_version=("version",str),
				verbose=("VERBOSE?",bool),
				# device specific w/ [<port>] being 0...3
				vendor_id=("DEVVID?[<port>]", int),
				home=("HOME(<port>", "ACK"),
				grip=("GRIP(<port>,<index>", "ACK"),
				)

DEV_STATUS_CODES = {
	0:"DS_NOT_CONNECTED",
	1:"DS_NOT_INITIALIZED",
	2:"DS_DISABLED",
	3:"DS_RELEASED",
	4:"DS_NO_PART",
	5:"DS_HOLDING",
	6:"DS_OPERATING",
	7:"DS_FAULT"}


class GriplinkController:
	def __init__(self, gripper_ip="192.168.1.40", gripper_port=10001, port=0):
		self.gripper_address = gripper_ip
		self.gripper_port = gripper_port
		self.port = port  # the internal griplink port, 0...3

		self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
		self.socket.connect((self.gripper_address, self.gripper_port))

	def close_gripper(self, blocking=False, port=0, index=0):
		cmd_string = f"GRIP({port},{index})"
		self._send_msg(cmd_string)
		if blocking:
			self._recv_msg()
			while self.status() not in ("DS_NO_PART", "DS_HOLDING"):
				time.sleep(0.1)

	def open_gripper(self, blocking=False, port=0, index=0):
		cmd_string = f"RELEASE({port},{index})"
		self._send_msg(cmd_string)
		if blocking:
			self._recv_msg()
			while self.status() != "DS_RELEASED":
				time.sleep(0.1)

	def get_opening_width(self, port=0, index=0):
		"""
		index 0, finger position in [mm]
		"""
		cmd_string = f"VALUE[{port}][{index}]?"
		ret_string = f"VALUE[{port}][{index}]="
		self._send_msg(cmd_string)
		tmp = -1000
		for i in range(3):
			ret = self._recv_msg()
			if ret == "ACK":
				# this is the acknowledgement of the last command, re-try
				continue
			elif ret_string in ret:
				tmp = int(ret.lstrip(ret_string))
				break
		if tmp == -1000:
			logging.warning("Invalid gripper width query, defaulting to -1.")
		if index == 0:
			# device return micrometers, convert to mm
			tmp /= 1000
		return tmp

	def status(self, port=0):
		"""
		Return device status as string.
		"""
		cmd_string = f"DEVSTATE[{port}]?"
		ret_string = f"DEVSTATE[{port}]="
		self._send_msg(cmd_string)
		ret = self._recv_msg()
		tmp = int(ret.lstrip(ret_string))
		return DEV_STATUS_CODES[tmp]

	def id(self):
		cmd_string, ret_cls = commands["id"]
		self._send_msg(cmd_string)
		return ret_cls(self._recv_msg())

	def home(self, port=0):
		cmd_string = f"HOME({port})"
		self._send_msg(cmd_string)
		self._recv_msg()

	def _send_msg(self, msg):
		self.socket.send((msg+"\n").encode("ASCII"))

	def _recv_msg(self):
		raw_msg = self.socket.recvfrom(256)[0]
		msg = raw_msg.decode("ASCII").lstrip("\n")
		msg = [x for x in msg.split("\n") if len(x)][-1]
		return msg


if __name__ == "__main__":
	gl = GriplinkController()
	print(gl.get_opening_width())
	gl.close_gripper(blocking=True)
	print(gl.get_opening_width())
	gl.open_gripper(blocking=False)
	# time.sleep(1)
	print(gl.get_opening_width())







