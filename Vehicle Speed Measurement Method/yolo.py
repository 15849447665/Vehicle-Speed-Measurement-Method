import colorsys
import os
import time

import cv2
import numpy as np
import torch
import math
import torch.nn as nn
from PIL import ImageDraw, ImageFont, Image

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox, DecodeBoxNP

'''
训练自己的数据集必看注释！
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/yolov7_weights.pth',
        "classes_path"      : 'model_data/coco_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #------------------------------------------------------#
        #   所使用到的yolov7的版本，本仓库一共提供两个：
        #   l : 对应yolov7
        #   x : 对应yolov7_x
        #------------------------------------------------------#
        "phi"               : 'l',
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
        #-------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        #-------------------------------#
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def phy_xy(self, x, y):
        x1 = (x - 960) * (3.36 / 0.67 / 1920)
        y1 = (y - 540) * (1.9 / 0.67 / 1080)
        print('phy_xy')
        print(x1, y1)
        return x1, y1

    def error_xy(self, x0, y0):
        # p35 37706  X
        # x1 = -7.4169 + (-18.6680 * x0) + (7.0921 * y0) + (-0.4608 * x0 * x0) + (79.9750 * x0 * y0) + (
        #             30.5644 * y0 * y0) + (-12.4306 * x0 * x0 * x0) + (1.4954 * x0 * x0 * y0) + (
        #                  -6.6072 * x0 * y0 * y0) + (-25.6356 * y0 * y0 * y0) + (10.5169 * x0 * x0 * x0 * y0) + (
        #                  -14.1171 * x0 * x0 * y0 * y0) + (86.5466 * x0 * y0 * y0 * y0) + (-22.3469 * y0 * y0 * y0 * y0) + (
        #                  -18.2625 * x0 * x0 * x0 * y0 * y0) + (12.3746 * x0 * x0 * y0 * y0 * y0) + (
        #                  -48.5676 * x0 * y0 * y0 * y0 * y0) + (15.3971 * y0 * y0 * y0 * y0 * y0)
        # p35 37706  Y
        # y1 = -61.8980 + (-10.5491 * x0) + (367.2367 * y0) + (-25.1937 * x0 * x0) + (8.3256 * x0 * y0) + (
        #             -34.2383 * y0 * y0) + (0.7759 * x0 * x0 * x0) + (-3.7204 * x0 * x0 * y0) + (
        #                  -2.8608 * x0 * y0 * y0) + (8.8504 * y0 * y0 * y0) + (-1.4681 * x0 * x0 * x0 * y0) + (
        #                  -41.3471 * x0 * x0 * y0 * y0) + (26.2590 * x0 * y0 * y0 * y0) + (
        #                  -212.2625 * y0 * y0 * y0 * y0) + (-0.6433 * x0 * x0 * x0 * y0 * y0) + (
        #                  26.5385 * x0 * x0 * y0 * y0 * y0) + (-17.9120 * x0 * y0 * y0 * y0 * y0) + (
        #                  123.9851 * y0 * y0 * y0 * y0 * y0)

        # p35 37750  X
        # x1 = -14.9320 + (-88.2863 * x0) + (17.9865 * y0) + (0.5997 * x0 * x0) + (158.9947 * x0 * y0) + (
        #             49.9187 * y0 * y0) + (-18.7769 * x0 * x0 * x0) + (1.5464 * x0 * x0 * y0) + (
        #                  -33.2561 * x0 * y0 * y0) + (-45.4874 * y0 * y0 * y0) + (19.7306 * x0 * x0 * x0 * y0) + (
        #                  -29.3468 * x0 * x0 * y0 * y0) + (81.1992 * x0 * y0 * y0 * y0) + (-32.3950 * y0 * y0 * y0 * y0) + (
        #                  -23.6579 * x0 * x0 * x0 * y0 * y0) + (28.1674 * x0 * x0 * y0 * y0 * y0) + (
        #                  -36.2895 * x0 * y0 * y0 * y0 * y0) + (16.5992 * y0 * y0 * y0 * y0 * y0)
        # # p35 37750  Y
        # y1 = -219.6445 + (-14.5546 * x0) + (533.4505 * y0) + (-28.4715 * x0 * x0) + (11.7928 * x0 * y0) + (
        #             -192.0110 * y0 * y0) + (1.2627 * x0 * x0 * x0) + (1.4479 * x0 * x0 * y0) + (
        #                  -2.0337 * x0 * y0 * y0) + (147.5152 * y0 * y0 * y0) + (-1.8651 * x0 * x0 * x0 * y0) + (
        #                  -46.8690 * x0 * x0 * y0 * y0) + (52.0492 * x0 * y0 * y0 * y0) + (
        #                  -136.8966 * y0 * y0 * y0 * y0) + (-1.6557 * x0 * x0 * x0 * y0 * y0) + (
        #                  36.4639 * x0 * x0 * y0 * y0 * y0) + (-38.7149 * x0 * y0 * y0 * y0 * y0) + (
        #                  20.4941 * y0 * y0 * y0 * y0 * y0)

        # p35 37780  X
        # x1 = -8.3770 + (-32.1853 * x0) + (5.7665 * y0) + (-0.0789 * x0 * x0) + (105.9549 * x0 * y0) + (
        #             32.7784 * y0 * y0) + (-15.0342 * x0 * x0 * x0) + (1.3554 * x0 * x0 * y0) + (
        #                  -1.4199 * x0 * y0 * y0) + (-30.0054 * y0 * y0 * y0) + (10.0302 * x0 * x0 * x0 * y0) + (
        #                  -19.2136 * x0 * x0 * y0 * y0) + (84.3529 * x0 * y0 * y0 * y0) + (-20.5358 * y0 * y0 * y0 * y0) + (
        #                  -20.6617 * x0 * x0 * x0 * y0 * y0) + (19.1994 * x0 * x0 * y0 * y0 * y0) + (
        #                  -54.7540 * x0 * y0 * y0 * y0 * y0) + (13.7672 * y0 * y0 * y0 * y0 * y0)
        # # # p35 37780  Y
        # y1 = -87.9829 + (-12.3474 * x0) + (427.2694 * y0) + (-29.7093 * x0 * x0) + (2.5185 * x0 * y0) + (
        #             -128.3360 * y0 * y0) + (0.9093 * x0 * x0 * x0) + (-4.9501 * x0 * x0 * y0) + (
        #                  9.7500 * x0 * y0 * y0) + (56.8412 * y0 * y0 * y0) + (-0.4316 * x0 * x0 * x0 * y0) + (
        #                  -40.4451 * x0 * x0 * y0 * y0) + (36.9997 * x0 * y0 * y0 * y0) + (
        #                  -143.6079 * y0 * y0 * y0 * y0) + (-2.7172 * x0 * x0 * x0 * y0 * y0) + (
        #                  25.3210 * x0 * x0 * y0 * y0 * y0) + (-33.0225 * x0 * y0 * y0 * y0 * y0) + (
        #                  78.6973 * y0 * y0 * y0 * y0 * y0)

        #p35 42706  X
        # x1 = -0.2571 + (-22.0120 * x0) + (7.3953 * y0) + (-0.2901 * x0 * x0) + (99.4785 * x0 * y0) + (-1.9245 * y0 * y0) + (-13.5610 * x0 * x0 * x0) + (-2.3728 * x0 * x0 * y0) + (29.4267 * x0 * y0 * y0) + (-7.7722 * y0 * y0 * y0) + (5.0833 * x0 * x0 * x0 * y0) + (2.0525 * x0 * x0 * y0 * y0) + (51.7597 * x0 * y0 * y0 * y0) + (0.5467 * y0 * y0 * y0 * y0) + (-20.0614 * x0 * x0 * x0 * y0 * y0) + (0.6702 * x0 * x0 * y0 * y0 * y0) + (-40.8456 * x0 * y0 * y0 * y0 * y0) + (2.4742 * y0 * y0 * y0 * y0 * y0)
        # #p35 42706  Y
        # y1 = -38.5028 + (-1.5665 * x0) + (385.0827 * y0) + (-30.2444 * x0 * x0) + (4.5925 * x0 * y0) + (-98.9086 * y0 * y0) + (0.1247 * x0 * x0 * x0) + (-14.4162 * x0 * x0 * y0) + (0.9234 * x0 * y0 * y0) + (14.3758 * y0 * y0 * y0) + (-0.9207 * x0 * x0 * x0 * y0) + (-30.4432 * x0 * x0 * y0 * y0) + (-2.6524 * x0 * y0 * y0 * y0) + (-78.9728 * y0 * y0 * y0 * y0) + (0.7777 * x0 * x0 * x0 * y0 * y0) + (19.1181 * x0 * x0 * y0 * y0 * y0) + (-0.5316 * x0 * y0 * y0 * y0 * y0) + (46.3973 * y0 * y0 * y0 * y0 * y0)

        # # p35 42750  X
        x1 = -15.2522 + (-134.6329 * x0) + (31.1477 * y0) + (1.2457 * x0 * x0) + (209.4534 * x0 * y0) + (
                45.2541 * y0 * y0) + (-20.9288 * x0 * x0 * x0) + (-5.1068 * x0 * x0 * y0) + (
                     -13.4090 * x0 * y0 * y0) + (-51.4881 * y0 * y0 * y0) + (20.6565 * x0 * x0 * x0 * y0) + (
                     -26.8270 * x0 * x0 * y0 * y0) + (73.0535 * x0 * y0 * y0 * y0) + (-24.8998 * y0 * y0 * y0 * y0) + (
                     -32.7422 * x0 * x0 * x0 * y0 * y0) + (32.3858 * x0 * x0 * y0 * y0 * y0) + (
                     -32.0020 * x0 * y0 * y0 * y0 * y0) + (9.4706 * y0 * y0 * y0 * y0 * y0)
        # # p35 42750  Y
        y1 = -253.5565 + (-20.6870 * x0) + (685.8706 * y0) + (-33.2300 * x0 * x0) + (8.7056 * x0 * y0) + (
                -220.4932 * y0 * y0) + (2.7088 * x0 * x0 * x0) + (-19.3262 * x0 * x0 * y0) + (
                     31.8231 * x0 * y0 * y0) + (52.0461 * y0 * y0 * y0) + (-1.4250 * x0 * x0 * x0 * y0) + (
                     -47.2180 * x0 * x0 * y0 * y0) + (41.8052 * x0 * y0 * y0 * y0) + (
                     -112.4528 * y0 * y0 * y0 * y0) + (-5.2377 * x0 * x0 * x0 * y0 * y0) + (
                     49.3879 * x0 * x0 * y0 * y0 * y0) + (-45.8098 * x0 * y0 * y0 * y0 * y0) + (
                     32.8989 * y0 * y0 * y0 * y0 * y0)

        # # p35 42780  X
        # x1 = -5.9100 + (-25.0650 * x0) + (3.5003 * y0) + (0.3044 * x0 * x0) + (126.5467 * x0 * y0) + (
        #         16.8795 * y0 * y0) + (-17.8052 * x0 * x0 * x0) + (0.5642 * x0 * x0 * y0) + (
        #              -5.8376 * x0 * y0 * y0) + (-14.7174 * y0 * y0 * y0) + (3.0642 * x0 * x0 * x0 * y0) + (
        #              -9.4175 * x0 * x0 * y0 * y0) + (54.6062 * x0 * y0 * y0 * y0) + (-8.9469 * y0 * y0 * y0 * y0) + (
        #              -16.0291 * x0 * x0 * x0 * y0 * y0) + (8.7596 * x0 * x0 * y0 * y0 * y0) + (
        #              -32.1359 * x0 * y0 * y0 * y0 * y0) + (5.6872 * y0 * y0 * y0 * y0 * y0)
        # # p35 42780  Y
        # y1 = -64.3642 + (-8.5548 * x0) + (430.1132 * y0) + (-34.5000 * x0 * x0) + (1.8063 * x0 * y0) + (
        #         -126.2842 * y0 * y0) + (0.8129 * x0 * x0 * x0) + (-12.0888 * x0 * x0 * y0) + (
        #              8.0169 * x0 * y0 * y0) + (53.6591 * y0 * y0 * y0) + (-0.5347 * x0 * x0 * x0 * y0) + (
        #              -29.3028 * x0 * x0 * y0 * y0) + (16.4161 * x0 * y0 * y0 * y0) + (
        #              -88.2244 * y0 * y0 * y0 * y0) + (-1.4111 * x0 * x0 * x0 * y0 * y0) + (
        #              12.5641 * x0 * x0 * y0 * y0 * y0) + (-14.8843 * x0 * y0 * y0 * y0 * y0) + (
        #              38.5150 * y0 * y0 * y0 * y0 * y0)

        # p35 46706  X
        # x1 = -3.9828 + (-53.7398 * x0) + (-2.5328 * y0) + (0.6534 * x0 * x0) + (157.6917 * x0 * y0) + (
        #         9.0654 * y0 * y0) + (-19.9770 * x0 * x0 * x0) + (1.5345 * x0 * x0 * y0) + (
        #              -2.1438 * x0 * y0 * y0) + (-4.6350 * y0 * y0 * y0) + (4.2059 * x0 * x0 * x0 * y0) + (
        #              -5.5631 * x0 * x0 * y0 * y0) + (35.9399 * x0 * y0 * y0 * y0) + (-3.5738 * y0 * y0 * y0 * y0) + (
        #              -17.3265 * x0 * x0 * x0 * y0 * y0) + (4.1601 * x0 * x0 * y0 * y0 * y0) + (
        #              -21.1754 * x0 * y0 * y0 * y0 * y0) + (1.2957 * y0 * y0 * y0 * y0 * y0)
        # # p35 46706  Y
        # y1 = -117.8747 + (-6.6393 * x0) + (448.6310 * y0) + (-34.4221 * x0 * x0) + (-1.2859 * x0 * y0) + (
        #         -125.6321 * y0 * y0) + (0.9227 * x0 * x0 * x0) + (-12.3354 * x0 * x0 * y0) + (
        #              13.0288 * x0 * y0 * y0) + (115.7461 * y0 * y0 * y0) + (0.5285 * x0 * x0 * x0 * y0) + (
        #              -24.8333 * x0 * x0 * y0 * y0) + (5.6688 * x0 * y0 * y0 * y0) + (
        #              -71.8018 * y0 * y0 * y0 * y0) + (-2.3350 * x0 * x0 * x0 * y0 * y0) + (
        #              10.9106 * x0 * x0 * y0 * y0 * y0) + (-8.7590 * x0 * y0 * y0 * y0 * y0) + (
        #              -13.9339 * y0 * y0 * y0 * y0 * y0)

        # p35 46750  X
        # x1 = -5.1390 + (-24.3563 * x0) + (5.1820 * y0) + (0.6993 * x0 * x0) + (140.5198 * x0 * y0) + (
        #         9.4907 * y0 * y0) + (-19.6572 * x0 * x0 * x0) + (-0.4917 * x0 * x0 * y0) + (
        #              1.6646 * x0 * y0 * y0) + (-12.2968 * y0 * y0 * y0) + (-1.3051 * x0 * x0 * x0 * y0) + (
        #              -5.7934 * x0 * x0 * y0 * y0) + (38.2853 * x0 * y0 * y0 * y0) + (-4.4687 * y0 * y0 * y0 * y0) + (
        #              -15.7732 * x0 * x0 * x0 * y0 * y0) + (6.1722 * x0 * x0 * y0 * y0 * y0) + (
        #              -27.3047 * x0 * y0 * y0 * y0 * y0) + (3.8470 * y0 * y0 * y0 * y0 * y0)
        # # p35 46750  Y
        # y1 = -54.5347 + (-6.5005 * x0) + (435.3944 * y0) + (-38.0750 * x0 * x0) + (4.7155 * x0 * y0) + (
        #         -114.4454 * y0 * y0) + (0.7094 * x0 * x0 * x0) + (-19.4825 * x0 * x0 * y0) + (
        #              5.7760 * x0 * y0 * y0) + (57.6480 * y0 * y0 * y0) + (-0.9655 * x0 * x0 * x0 * y0) + (
        #              -22.6668 * x0 * x0 * y0 * y0) + (7.8002 * x0 * y0 * y0 * y0) + (
        #              -78.4610 * y0 * y0 * y0 * y0) + (-0.7356 * x0 * x0 * x0 * y0 * y0) + (
        #              10.7547 * x0 * x0 * y0 * y0 * y0) + (-9.4465 * x0 * y0 * y0 * y0 * y0) + (
        #              10.6644 * y0 * y0 * y0 * y0 * y0)

        # p35 46780  X
        # x1 = -5.9773 + (-32.8943 * x0) + (8.5216 * y0) + (0.1252 * x0 * x0) + (139.6988 * x0 * y0) + (
        #         12.0082 * y0 * y0) + (-20.1540 * x0 * x0 * x0) + (-0.4223 * x0 * x0 * y0) + (
        #              13.4350 * x0 * y0 * y0) + (-19.6169 * y0 * y0 * y0) + (0.7772 * x0 * x0 * x0 * y0) + (
        #              -7.5867 * x0 * x0 * y0 * y0) + (44.0696 * x0 * y0 * y0 * y0) + (-5.5178 * y0 * y0 * y0 * y0) + (
        #              -19.0785 * x0 * x0 * x0 * y0 * y0) + (8.1045 * x0 * x0 * y0 * y0 * y0) + (
        #              -32.7066 * x0 * y0 * y0 * y0 * y0) + (6.3794 * y0 * y0 * y0 * y0 * y0)
        # # p35 46780  Y
        # y1 = -67.9353 + (-9.1597 * x0) + (463.9632 * y0) + (-39.3296 * x0 * x0) + (5.5324 * x0 * y0) + (
        #         -112.7522 * y0 * y0) + (0.9239 * x0 * x0 * x0) + (-21.3294 * x0 * x0 * y0) + (
        #              8.8979 * x0 * y0 * y0) + (45.1374 * y0 * y0 * y0) + (-0.9251 * x0 * x0 * x0 * y0) + (
        #              -25.9964 * x0 * x0 * y0 * y0) + (10.6875 * x0 * y0 * y0 * y0) + (
        #              -86.5102 * y0 * y0 * y0 * y0) + (-1.2952 * x0 * x0 * x0 * y0 * y0) + (
        #              16.5402 * x0 * x0 * y0 * y0 * y0) + (-12.8065 * x0 * y0 * y0 * y0 * y0) + (
        #              15.6184 * y0 * y0 * y0 * y0 * y0)
        print("error_xy")
        print(x1, y1)
        return x1, y1

    def world_xy(self, x1, y1, ds, f, alpha):
        alpha1 = math.cos(math.pi/180 * alpha)
        alpha1 = round(alpha1 , 4)
        x_world = ds * ((x1 * alpha1 / np.power(f * f + x1 * x1 * alpha1 * alpha1, 0.5)))
        y_world = ds * (f / np.power(f * f + x1 * x1 * alpha1 * alpha1, 0.5))
        print("world_xy")
        print(x_world, y_world)
        return x_world, y_world

    def fitF(self, x, y):
        # f = 7.8704 + 0.4792 * x1 + 6.8501 * y1 + (-1.3334 * x1 * x1) + 0.6497 * x1 * y1 + 3.0074 * y1 * y1 + (-0.0333 * x1 * x1 * x1) + (-1.9271 * x1 * x1 * y1) + 0.2337 * x1 * y1 * y1 + 0.0646 * x1 * x1 * x1 * x1 + 0.0247 * x1 * x1 * x1 * y1 + (-0.7968 * x1 * x1 * y1 * y1) + 0.0008691 * x1 * x1 * x1 * x1 * x1 + 0.0555 * x1 * x1 * x1 * x1 * y1 + 0.0522 * x1 * x1 * x1 * y1 * y1
        x = round(x, 4)
        y = round(y, 4)
        # p32 37706
        # f = 2.3015 + (-0.0351 * x) + (-0.4096 * y) + (-0.0947 * x * x) + (0.0113 * x * y) + (-0.1102 * y * y) + (0.0054 * x * x * x) + (-0.0014 * x * x * y) + (-0.00031515 * x * y * y)

        # p32 37750
        # f = 1.9504 + (-0.0394 * x) + (-0.4270 * y) + (-0.0778 * x * x) + (0.0161 * x * y) + (0.0454 * y * y) + (0.0064 * x * x * x) + (0.0026 * x * x * y) + (-0.0021 * x * y * y)

        # p32 37780
        # f = 2.2391 + (-0.0253 * x) + (-0.4387 * y) + (-0.0922 * x * x) + (0.0146 * x * y) + (-0.0908 * y * y) + (0.0042 * x * x * x) + (-0.00069128 * x * x * y) + (-0.0035 * x * y * y)

        # p32 42706
        # f = 2.0096 + (-0.0048 * x) + (-0.4464 * y) + (-0.0833 * x * x) + (-0.0021 * x * y) + (-0.0827 * y * y) + (0.0010 * x * x * x) + (0.00018485 * x * x * y) + (0.00077087 * x * y * y)

        # p32 42750
        f = 1.5523 + (-0.0251 * x) + (-0.4072 * y) + (-0.0606 * x * x) + (0.0133 * x * y) + (0.0689 * y * y) + (0.0043 * x * x * x) + (0.0021 * x * x * y) + (-0.0026 * x * y * y)

        # p32 42780
        # f = 1.9597 + (-0.0142 * x) + (-0.4580 * y) + (-0.0812 * x * x) + (0.0083 * x * y) + (-0.0798 * y * y) + (0.0024 * x * x * x) + (-0.0009266 * x * x * y) + (-0.0026 * x * y * y)

        # p32 46706
        # f = 1.6022 + (0.0024 * x) + (-0.4238 * y) + (-0.0655 * x * x) + (0.0050 * x * y) + (-0.0046 * y * y) + (
        #             -0.00029937 * x * x * x) + (0.00063547 * x * x * y) + (-0.0028 * x * y * y)

        # p32 46750
        # f = 1.7307 + (-0.0102 * x) + (-0.4660 * y) + (-0.0721 * x * x) + (0.0058 * x * y) + (-0.0712 * y * y) + (
        #         0.0018 * x * x * x) + (-0.00016586 * x * x * y) + (-0.0025 * x * y * y)

        # p32 46780
        # f = 1.7178 + (-0.0138 * x) + (-0.4710 * y) + (-0.0720 * x * x) + (0.0066 * x * y) + (-0.0707 * y * y) + (
        #             0.0022 * x * x * x) + (0.00018252 * x * x * y) + (-0.0022 * x * y * y)

        f = round(f, 4)
        print("fitF")
        print(f)
        return f

    def solved(self, x, y, f, alpha, h):
        alpha1 = math.tan(math.pi / 180 * alpha)
        alpha1 = round(alpha1, 4)
        d = np.abs((h * (f * f + x * x - (y * f * alpha1))) / ((f * alpha1 + y) * np.power(np.power(f,2) + np.power(x, 2), 0.5)))
        print("solved")
        print(d)
        return d

    def sudu(self, x1, x2, y1, y2):
        juli = np.power(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)), 0.5)
        v = juli / (1 / 30) / 1000
        return v


    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.fuse().eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   计数
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   是否进行目标的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            #i为第几个物体   c为对应物体类别  top_label存检测到的物体
            predicted_class = self.class_names[int(c)]  #类别名

            box             = top_boxes[i]  #坐标
            score           = top_conf[i]  #概率

            top, left, bottom, right = box  #四个坐标赋值

            #四舍五入将坐标调整为整数
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))
            midx = (left + right) / 2
            midy = bottom
            print(midx)
            print(midy)
            alpha = 37
            h = 750
            x1, y1 = self.phy_xy(midx, midy)
            print(x1)
            print(y1)
            f = self.fitF(x1, y1)
            print(f)
            d = self.solved(x1, y1, f, alpha, h)
            print(d)
            x_world, y_world = self.world_xy(x1, y1, d, f, alpha)
            print(x_world)
            print(y_world)
            x_error, y_error = self.error_xy(x1, y1)
            print(x_error)
            print(y_error)
            x_finish = x_world + x_error
            y_finish = y_world + y_error
            print(x_finish)
            print(y_finish)

            label = '{} {:.2f} {:.2f} {:.2f}'.format(predicted_class, score, x_finish, y_finish)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            #绘制边框
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            #绘制文本框
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            #写文本
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def video_detect_image(self, image, dir,  crop=False, count=False):
        # ---------------------------------------------------#
        #   计算输入图片的高和宽
        # ---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            # ---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            # ---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            # ---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            # ---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape,
                                                         image_shape, self.letterbox_image, conf_thres=self.confidence,
                                                         nms_thres=self.nms_iou)

            if results[0] is None:
                return image

            top_label = np.array(results[0][:, 6], dtype='int32')
            top_conf = results[0][:, 4] * results[0][:, 5]
            top_boxes = results[0][:, :4]
        # ---------------------------------------------------------#
        #   设置字体与边框厚度
        # ---------------------------------------------------------#
        font = ImageFont.truetype(font='model_data/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        # ---------------------------------------------------------#
        #   计数
        # ---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        # ---------------------------------------------------------#
        #   是否进行目标的裁剪
        # ---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_boxes)):
                top, left, bottom, right = top_boxes[i]
                top = max(0, np.floor(top).astype('int32'))
                left = max(0, np.floor(left).astype('int32'))
                bottom = min(image.size[1], np.floor(bottom).astype('int32'))
                right = min(image.size[0], np.floor(right).astype('int32'))

                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        # ---------------------------------------------------------#
        #   图像绘制
        # ---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            # i为第几个物体   c为对应物体类别  top_label存检测到的物体
            predicted_class = self.class_names[int(c)]  # 类别名
            box = top_boxes[i]  # 坐标
            score = top_conf[i]  # 概率

            top, left, bottom, right = box  # 四个坐标赋值

            # 四舍五入将坐标调整为整数
            top = max(0, np.floor(top).astype('int32'))
            left = max(0, np.floor(left).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom).astype('int32'))
            right = min(image.size[0], np.floor(right).astype('int32'))
            midx = (left + right) / 2
            # midy = (bottom + top) / 2
            midy = bottom
            alpha = 42
            h = 750
            x1, y1 = self.phy_xy(midx, midy)
            f = self.fitF(x1, y1)
            d = self.solved(x1, y1, f, alpha, h)
            x_world, y_world = self.world_xy(x1, y1, d, f, alpha)
            x_error, y_error = self.error_xy(x1, y1)
            x_finish = x_world + x_error
            y_finish = y_world + y_error
            v = 0
            print(len(dir))
            if predicted_class == 'truck':
                l = len(dir)
                ll = l + 1
                xx = {ll : [predicted_class, x_finish, y_finish]}
                if l == 0 :
                    dir.update(xx)
                elif l > 0:
                    v = self.sudu(dir[l][1], x_finish, dir[l][2], y_finish)
                    dir.update(xx)
            v = round(v, 2)
            vv = str(v) + 'm/s'
            label = '{} {:.2f} {}'.format(predicted_class, score, vv)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # 绘制边框
            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            # 绘制文本框
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            # 写文本
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   将图像输入网络当中进行预测！
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   将预测框进行堆叠，然后进行非极大抑制
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def convert_to_onnx(self, simplify, model_path):
        import onnx
        self.generate(onnx=True)

        im                  = torch.zeros(1, 3, *self.input_shape).to('cpu')  # image size(1, 3, 512, 512) BCHW
        input_layer_names   = ["images"]
        output_layer_names  = ["output"]
        
        # Export the model
        print(f'Starting export with onnx {onnx.__version__}.')
        torch.onnx.export(self.net,
                        im,
                        f               = model_path,
                        verbose         = False,
                        opset_version   = 12,
                        training        = torch.onnx.TrainingMode.EVAL,
                        do_constant_folding = True,
                        input_names     = input_layer_names,
                        output_names    = output_layer_names,
                        dynamic_axes    = None)

        # Checks
        model_onnx = onnx.load(model_path)  # load onnx model
        onnx.checker.check_model(model_onnx)  # check onnx model

        # Simplify onnx
        if simplify:
            import onnxsim
            print(f'Simplifying with onnx-simplifier {onnxsim.__version__}.')
            model_onnx, check = onnxsim.simplify(
                model_onnx,
                dynamic_input_shape=False,
                input_shapes=None)
            assert check, 'assert check failed'
            onnx.save(model_onnx, model_path)

        print('Onnx model save as {}'.format(model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"), "w", encoding='utf-8') 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 


#暂时不用管---------------------------------------------------------------#

class YOLO_ONNX(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改onnx_path和classes_path！
        #   onnx_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的onnx_path和classes_path参数的修改
        #--------------------------------------------------------------------------#
        "onnx_path"         : 'model_data/models.onnx',
        "classes_path"      : 'model_data/coco_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先验框对应的txt文件，一般不修改。
        #   anchors_mask用于帮助代码找到对应的先验框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   输入图片的大小，必须为32的倍数。
        #---------------------------------------------------------------------#
        "input_shape"       : [640, 640],
        #---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   该变量用于控制是否使用letterbox_image对输入图像进行不失真的resize，
        #   在多次测试后，发现关闭letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True
    }
    
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        import onnxruntime
        self.onnx_session   = onnxruntime.InferenceSession(self.onnx_path)
        # 获得所有的输入node
        self.input_name     = self.get_input_name()
        # 获得所有的输出node
        self.output_name    = self.get_output_name()

        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = self.get_classes(self.classes_path)
        self.anchors, self.num_anchors      = self.get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBoxNP(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        show_config(**self._defaults)
 
    def get_classes(self, classes_path):
        with open(classes_path, encoding='utf-8') as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names, len(class_names)
    
    def get_anchors(self, anchors_path):
        '''loads the anchors from a file'''
        with open(anchors_path, encoding='utf-8') as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
        return anchors, len(anchors)

    def get_input_name(self):
        # 获得所有的输入node
        input_name=[]
        for node in self.onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name
 
    def get_output_name(self):
        # 获得所有的输出node
        output_name=[]
        for node in self.onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_feed(self,image_tensor):
        # 利用input_name获得输入的tensor
        input_feed={}
        for name in self.input_name:
            input_feed[name]=image_tensor
        return input_feed
    
    #---------------------------------------------------#
    #   对输入图像进行resize
    #---------------------------------------------------#
    def resize_image(self, image, size, letterbox_image, mode='PIL'):
        if mode == 'PIL':
            iw, ih  = image.size
            w, h    = size

            if letterbox_image:
                scale   = min(w/iw, h/ih)
                nw      = int(iw*scale)
                nh      = int(ih*scale)

                image   = image.resize((nw,nh), Image.BICUBIC)
                new_image = Image.new('RGB', size, (128,128,128))
                new_image.paste(image, ((w-nw)//2, (h-nh)//2))
            else:
                new_image = image.resize((w, h), Image.BICUBIC)
        else:
            image = np.array(image)
            if letterbox_image:
                # 获得现在的shape
                shape       = np.shape(image)[:2]
                # 获得输出的shape
                if isinstance(size, int):
                    size    = (size, size)

                # 计算缩放的比例
                r = min(size[0] / shape[0], size[1] / shape[1])

                # 计算缩放后图片的高宽
                new_unpad   = int(round(shape[1] * r)), int(round(shape[0] * r))
                dw, dh      = size[1] - new_unpad[0], size[0] - new_unpad[1]

                # 除以2以padding到两边
                dw          /= 2  
                dh          /= 2
        
                # 对图像进行resize
                if shape[::-1] != new_unpad:  # resize
                    image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
                top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
                new_image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(128, 128, 128))  # add border
            else:
                new_image = cv2.resize(image, (w, h))

        return new_image
 
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
 
        image_data  = self.resize_image(image, self.input_shape, True)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #   h, w, 3 => 3, h, w => 1, 3, h, w
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)
 
        input_feed  = self.get_input_feed(image_data)
        outputs     = self.onnx_session.run(output_names=self.output_name, input_feed=input_feed)

        feature_map_shape   = [[int(j / (2 ** (i + 3))) for j in self.input_shape] for i in range(len(self.anchors_mask))][::-1]
        for i in range(len(self.anchors_mask)):
            outputs[i] = np.reshape(outputs[i], (1, len(self.anchors_mask[i]) * (5 + self.num_classes), feature_map_shape[i][0], feature_map_shape[i][1]))
        
        outputs = self.bbox_util.decode_box(outputs)
        #---------------------------------------------------------#
        #   将预测框进行堆叠，然后进行非极大抑制
        #---------------------------------------------------------#
        results = self.bbox_util.non_max_suppression(np.concatenate(outputs, 1), self.num_classes, self.input_shape, 
                    image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                
        if results[0] is None: 
            return image

        top_label   = np.array(results[0][:, 6], dtype = 'int32')
        top_conf    = results[0][:, 4] * results[0][:, 5]
        top_boxes   = results[0][:, :4]

        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image