#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ultralytics_ros
# Copyright (C) 2023-2024  Alpaca-zip
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import cv2
import cv_bridge
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from sensor_msgs.msg import Image , CameraInfo
from ultralytics import YOLO
from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
from ultralytics_ros.msg import YoloResult
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as TimeMsg

from collections import deque
from message_filters import SimpleFilter
import dataclasses

from message_filters import ApproximateTimeSynchronizer
import message_filters

def time_msg_to_sec(time: TimeMsg):
    return time.sec + (time.nanosec * 1e-9)



@dataclasses.dataclass
class ProcessPieces():
    time_stamp : float # This is needed on construction.
    detection_info : YoloResult = None  # place holder
    depth_image: np.ndarray = None
    camera_info: CameraInfo = None

    TIME_TOLERANCE_S = 0.02

    def SameTime(self,new_time : TimeMsg) ->bool:
        """Using time to check if 3 message is paired"""

        return abs(time_msg_to_sec(new_time) - self.time_stamp) < self.TIME_TOLERANCE_S

class TrackerNode(Node):
    def __init__(self):

        # Parameters.
        super().__init__("tracker_depth_node")
        self.declare_parameter("yolo_model", "yolov8n.pt")
        self.declare_parameter("input_topic", "image_raw")

        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("camera_info_topic", "image_raw")

        self.declare_parameter("result_topic", "yolo_result")
        self.declare_parameter("result_image_topic", "yolo_image")
        self.declare_parameter("conf_thres", 0.25)
        self.declare_parameter("iou_thres", 0.45)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("classes", list(range(80)))
        self.declare_parameter("tracker", "bytetrack.yaml")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("result_conf", True)
        self.declare_parameter("result_line_width", 1)
        self.declare_parameter("result_font_size", 1)
        self.declare_parameter("result_font", "Arial.ttf")
        self.declare_parameter("result_labels", True)
        self.declare_parameter("result_boxes", True)

        yolo_model = self.get_parameter("yolo_model").get_parameter_value().string_value

        input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        result_topic = (
            self.get_parameter("result_topic").get_parameter_value().string_value
        )
        result_image_topic = (
            self.get_parameter("result_image_topic").get_parameter_value().string_value
        )

        self.model = YOLO(f"{yolo_model}")
        # Fuses Conv2d and BatchNorm2d layers in the model for optimized inference.
        self.model.fuse()

        self.bridge = cv_bridge.CvBridge()

        # TODO remove
        # self.use_segmentation = yolo_model.endswith("-seg.pt")
        self.use_segmentation = True

        self.raw_mask_queue = deque(maxlen=2)
        self.raw_masks = None
        self.results_pub = self.create_publisher(YoloResult, result_topic, 1)
        self.result_image_pub = self.create_publisher(Image, result_image_topic, 1)
        self.result_depth_pub = self.create_publisher(Image, "/yolo_ros/masked_depth", 1)


        # Always process the visual image even if depth did not come (tracking have cases where we don't go further into process depth image)
        self.image_subs = self.create_subscription(Image, input_topic, self.image_callback, 1)

        # Atleast the image info need to be queued.
        self.results_subs = message_filters.Subscriber(self,YoloResult, result_topic, qos_profile= 4)
        self.depth_subs = message_filters.Subscriber(self,Image, depth_topic, qos_profile= 4)
        self.camera_info_subs = message_filters.Subscriber(self,CameraInfo, camera_info_topic, qos_profile= 4)

        self.depth_sync = ApproximateTimeSynchronizer([self.results_subs , self.depth_subs , self.camera_info_subs] , 5 , 0.2)
        self.depth_sync.registerCallback(self.depth_process_cb)
    def depth_process_cb(self,result_msg : YoloResult , depth_msg : Image , camera_info_msg : CameraInfo):
        # Go through and find out point for each mask.
        if not self.raw_mask_queue:
            self.get_logger().error(f"no raw mask at all: detection num: {len(result_msg.detections.detections)}")
            return
        raw_masks = self.raw_mask_queue.popleft()

        if len(raw_masks) != len(result_msg.detections.detections):
            self.get_logger().error(f"Different num of detection and mask ,raw mask {len(raw_masks)} mask num: {len(result_msg.masks)} , detection num: {len(result_msg.detections.detections)}")
            return


        depth_image = self.bridge.imgmsg_to_cv2(depth_msg , desired_encoding=depth_msg.encoding)
        print(f"msg size {depth_msg.width} {depth_msg.height} , depth size { depth_image.shape}")
        print(f"depth image type {depth_image.dtype}")
        out1 = ""
        out2 = ""
        for i in range(20,50):
            out1 += f", {depth_msg.data[i]}"
            out2 += f", {int(depth_image[0][i])}"
        print(f"msg : {out1}")
        print(f"cv  : {out2}")
        # exit(1)
        combined_mask = raw_masks[0]
        self.get_logger().warn(f"mask shape {combined_mask.shape}")
        for mask , detection in zip(raw_masks , result_msg.detections.detections):
            combined_mask = cv2.bitwise_or(combined_mask,mask)

        masked_depth = cv2.copyTo(depth_image , mask = combined_mask)
        # masked_depth = depth_image.copy()
        masked_msg = self.bridge.cv2_to_imgmsg(masked_depth, encoding=depth_msg.encoding)

        masked_msg.header = depth_msg.header

        # Publish depth image for debug.
        self.result_depth_pub.publish(masked_msg)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # Because the camera is mounted side ways, this need to be done.
        # TODO make this a param in future.
        rot_cv_image = cv2.rotate(cv_image,cv2.ROTATE_90_CLOCKWISE)

        # self.get_logger().warn(f"image shape {cv_image.shape}")
        conf_thres = self.get_parameter("conf_thres").get_parameter_value().double_value
        iou_thres = self.get_parameter("iou_thres").get_parameter_value().double_value
        max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        classes = (
            self.get_parameter("classes").get_parameter_value().integer_array_value
        )
        tracker = self.get_parameter("tracker").get_parameter_value().string_value
        device = self.get_parameter("device").get_parameter_value().string_value or None
        results = self.model.track(
            source=rot_cv_image,
            conf=conf_thres,
            iou=iou_thres,
            max_det=max_det,
            classes=classes,
            tracker=tracker,
            device=device,
            verbose=False,
            retina_masks=True,
        )

        if results is not None:
            yolo_result_msg = YoloResult()
            yolo_result_image_msg = Image()
            yolo_result_msg.detections = self.create_detections_array(results)
            # TODO, need to rotate detection back as well, but won't do for now.
            yolo_result_image_msg = self.create_result_image(results)
            # TODO we always want segmentation detection.
            if self.use_segmentation:
                yolo_result_msg.masks , raw_masks = self.create_segmentation_masks(results,cv2.ROTATE_90_COUNTERCLOCKWISE)
            self.raw_mask_queue.append(raw_masks)

            yolo_result_msg.header = msg.header
            # The cv2 to image actually overrides the entier message, so it looses header
            yolo_result_image_msg.header = msg.header
            self.results_pub.publish(yolo_result_msg)
            self.result_image_pub.publish(yolo_result_image_msg)


    def create_detections_array(self, results):
        detections_msg = Detection2DArray()
        bounding_box = results [0].boxes.xywh
        classes = results[0].boxes.cls
        confidence_score = results[0].boxes.conf
        for bbox, cls, conf in zip(bounding_box, classes, confidence_score):
            detection = Detection2D()
            detection.bbox.center.position.x = float(bbox[0])
            detection.bbox.center.position.y = float(bbox[1])
            detection.bbox.size_x = float(bbox[2])
            detection.bbox.size_y = float(bbox[3])
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = results[0].names.get(int(cls))
            hypothesis.hypothesis.score = float(conf)
            detection.results.append(hypothesis)
            detections_msg.detections.append(detection)
        return detections_msg

    def create_result_image(self, results):
        result_conf = self.get_parameter("result_conf").get_parameter_value().bool_value
        result_line_width = (
            self.get_parameter("result_line_width").get_parameter_value().integer_value
        )
        result_font_size = (
            self.get_parameter("result_font_size").get_parameter_value().integer_value
        )
        result_font = (
            self.get_parameter("result_font").get_parameter_value().string_value
        )
        result_labels = (
            self.get_parameter("result_labels").get_parameter_value().bool_value
        )
        result_boxes = (
            self.get_parameter("result_boxes").get_parameter_value().bool_value
        )
        plotted_image = results[0].plot(
            conf=result_conf,
            line_width=result_line_width,
            font_size=result_font_size,
            font=result_font,
            labels=result_labels,
            boxes=result_boxes,
        )
        rot_plotted_image = cv2.rotate(plotted_image,cv2.ROTATE_90_COUNTERCLOCKWISE)

        result_image_msg = self.bridge.cv2_to_imgmsg(rot_plotted_image, encoding="bgr8")
        return result_image_msg

    def create_segmentation_masks(self, results , rotate = None):
        mask_msgs = []
        np_masks = []
        for result in results:
            if hasattr(result, "masks") and result.masks is not None:
                for mask_tensor in result.masks:
                    mask_numpy = (
                        np.squeeze(mask_tensor.data.to("cpu").detach().numpy()).astype(
                            np.uint8
                        )
                        * 255
                    )
                    if rotate:
                        mask_numpy = cv2.rotate(mask_numpy, rotate)
                    np_masks.append(mask_numpy)
                    mask_image_msg = self.bridge.cv2_to_imgmsg(
                        mask_numpy, encoding="mono8"
                    )
                    mask_msgs.append(mask_image_msg)
        return mask_msgs , np_masks


def main(args=None):
    rclpy.init(args=args)
    node = TrackerNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
