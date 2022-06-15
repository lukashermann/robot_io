import cv2
import hydra
import numpy as np


def click_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["drawing"] = True
        param["xy1"] = [x, y]
        param["xy2"] = [x, y]
    elif event == cv2.EVENT_MOUSEMOVE:
        if param["drawing"]:
            param["xy2"] = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        param["drawing"] = False
        param["xy2"] = [x, y]


def set_crop_coordinates(cam, resolution=None):
    """
    Get the coordinates of an image region of interest, by drawing a selection rectangle.
    Copy the values into a camera config to use them as default.

    Args:
        cam: Camera (Kinect4 or Realsense / Framos).
        resolution: Resolution defining the aspect ratio of the cropped ROI (only the aspect ratio is used here).
    """
    param = {"drawing": False}
    cv2.namedWindow("rgb")
    cv2.setMouseCallback("rgb", click_callback, param)
    print("Press ENTER to apply selection")
    rectangle_set = False
    while True:
        rgb, depth = cam.get_image()

        if "xy1" in param:
            xy1 = param["xy1"].copy()
            xy2 = param["xy2"].copy()
            xy1 = np.clip(xy1, [0, 0], [rgb.shape[1], rgb.shape[0]])
            xy2 = np.clip(xy2, [0, 0], [rgb.shape[1], rgb.shape[0]])
            x1 = min(xy1[0], xy2[0])
            y1 = min(xy1[1], xy2[1])
            x2 = max(xy1[0], xy2[0])
            y2 = max(xy1[1], xy2[1])

            width = x2 - x1
            height = y2 - y1

            if not (width == 0 or height == 0):
                ratio = resolution[0] / resolution[1] if resolution is not None else cam.resolution[0] / cam.resolution[1]
                if width / height > ratio:
                    width = height * ratio
                else:
                    height = width / ratio
                height = int(np.round(height))
                width = int(np.round(width))

                if xy1[0] > xy2[0]:
                    x = x2 - width
                else:
                    x = x1
                if xy1[1] > xy2[1]:
                    y = y2 - height
                else:
                    y = y1
                cv2.rectangle(rgb, (x, y), (x + width, y + height), (0, 255, 0), 2)
                rectangle_set = True
        cv2.imshow("rgb", rgb[:, :, ::-1])
        k = cv2.waitKey(1) % 256
        if k == 13 and rectangle_set:
            mask = np.zeros_like(depth)
            mask[y: y + height, x: x + width] = 1
            rgb[np.where(mask == 0)] = 0
            depth[np.where(mask == 0)] = 0

            cv2.imshow("rgb", rgb[:, :, ::-1])
            cv2.imshow("depth", depth)
            print("Press ENTER to finish selection, press c button to redo.")
            k = cv2.waitKey(0) % 256
            if k == 13:
                print(f"Image coordinates: ({y}, {x}, {y + height}, {x + width})")
                return
            else:
                continue




@hydra.main(config_path="../conf", config_name="set_crop_coordinates")
def main(cfg):
    cam = hydra.utils.instantiate(cfg.cam)
    set_crop_coordinates(cam, resolution=cfg.resolution)


if __name__ == "__main__":
    main()
