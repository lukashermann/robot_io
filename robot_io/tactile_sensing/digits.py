import cv2
from digit_interface.digit import Digit
from digit_interface.digit_handler import DigitHandler


class Digits:
    """
    Interface class to get data from digits tactile camera.
    """

    def __init__(self,
                 device_a: str,
                 device_b: str,
                 resolution: str,
                 fps: str):
        print("Found digit sensors \n {}".format(DigitHandler.list_digits()))
        print("Supported streams: \n {}".format(DigitHandler.STREAMS))
        self.serial1 = device_a
        self.serial2 = device_b
        self.resolution = resolution
        self.fps = fps
        self.d1 = Digit(self.serial1)
        self.d2 = Digit(self.serial2)
        self.d1.connect()
        self.d2.connect()
        print("Connected digits")
        assert (resolution == "VGA" or resolution == "QVGA")
        self.d1.set_resolution(DigitHandler.STREAMS[resolution])
        self.d2.set_resolution(DigitHandler.STREAMS[resolution])
        assert ((fps == "60fps" or fps == "30fps") or fps == "15fps")
        self.d1.set_fps(DigitHandler.STREAMS[resolution]["fps"][fps])
        self.d2.set_fps(DigitHandler.STREAMS[resolution]["fps"][fps])
        print("Finished initializing digits")
        print(self.d1.info())

    def get_image(self):
        frame1 = self.d1.get_frame()
        frame2 = self.d2.get_frame()
        return frame1, frame2


def test_digit():
    d = Digits("D00020", "D00010", "QVGA", "30fps")
    while True:
        frame1, frame2 = d.get_image()
        cv2.imshow(f"Digit View {d.serial1}", frame1)
        cv2.imshow(f"Digit View {d.serial2}", frame2)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    test_digit()
