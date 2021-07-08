
import numpy as np

class RGBDCameraIntrinsics:
    def __init__(self, fx, fy, cx, cy, img_w, img_h, scale=1.0):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.img_w = img_w
        self.img_h = img_h
        self.scale = scale

    def rescale(self, scale):
        return RGBDCameraIntrinsics(self.fx*scale, self.fy*scale, self.cx*scale, self.cy*scale, self.img_w*scale, self.img_h*scale, scale)

def GetCameraParameters(camera_name, scale):
    if camera_name == "OrbbecAstra": params = RGBDCameraIntrinsics(570.342, 570.342, 320, 240, 640, 480)
    elif camera_name == "OrbbecAstraV2": params = RGBDCameraIntrinsics(581.1102688880432, 576.1734668524375, 305.7483731000336, 244.8215753417885, 640, 480)
    elif camera_name == "OrbbecAstraPro": params = RGBDCameraIntrinsics(553.797, 553.722, 320, 240, 640, 480)
    elif camera_name == "OrbbecPersee": params = RGBDCameraIntrinsics(553.797, 553.722, 320, 240, 640, 480)
    elif camera_name == "RealSenseD435": params = RGBDCameraIntrinsics(387.4065246582031, 387.4065246582031, 318.5951843261719, 241.4065399169922, 640, 480)
    return params.rescale(scale)


class CameraPose:
    def __init__(self, pose_vals):
        self.m11 = pose_vals[0]
        self.m12 = pose_vals[1]
        self.m13 = pose_vals[2]
        self.m14 = pose_vals[3]

        self.m21 = pose_vals[4]
        self.m22 = pose_vals[5]
        self.m23 = pose_vals[6]
        self.m24 = pose_vals[7]

        self.m31 = pose_vals[8]
        self.m32 = pose_vals[9]
        self.m33 = pose_vals[10]
        self.m34 = pose_vals[11]

    def rotationMatrix(self):
        row1 = [self.m11, self.m12, self.m13]
        row2 = [self.m21, self.m22, self.m23]
        row3 = [self.m31, self.m32, self.m33]
        return np.array([row1, row2, row3])

    def translationMatrix(self):
        return np.array([self.m14, self.m24, self.m34])


def ReadCameraPoses(str_filename):
    pose_file = open(str_filename, 'r')

    lines = pose_file.readlines()
    lines = [x.strip() for x in lines]

    pose_file.close()


    poses = []
    for l in lines:
        vals = l.split()
        vals = [float(v) for v in vals]
        poses.append(CameraPose(vals))

    return poses



