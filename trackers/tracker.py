import os
import pickle
from ultralytics import YOLO
import supervision as sv
import sys
import cv2

sys.path.append("../")
from utils import get_bbox_center, get_bbox_width


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames, batch_size=20, confidence=0.1):
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                frames[i : i + batch_size], conf=confidence
            )
            detections.extend(detections_batch)
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        # Handle stub file logic
        if read_from_stub and stub_path and os.path.exists(stub_path):
            try:
                with open(stub_path, "rb") as f:
                    tracks = pickle.load(f)
                print("Stub file readed")
                return tracks
            except Exception as e:
                print(f"Error reading stub file: {e}")
                return None

        # Initialize track structure
        tracks = {"players": [], "referees": [], "ball": []}

        # Get detections
        detections = self.detect_frames(frames)

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Convert YOLO detections to supervision format
            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert 'goalkeeper' to 'player'
            for i, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[i] = cls_names_inv["player"]

            # Update tracker with detections
            tracked_objects = self.tracker.update_with_detections(detection_supervision)

            # Prepare frame data
            frame_players = {}
            frame_referees = {}
            frame_ball = {}

            for obj in tracked_objects:
                bbox = obj[0].tolist()
                cls_id = obj[3]
                track_id = obj[4]

                if cls_id == cls_names_inv["player"]:
                    frame_players[track_id] = {"bbox": bbox}
                elif cls_id == cls_names_inv["referee"]:
                    frame_referees[track_id] = {"bbox": bbox}

            # Handle ball detection
            for obj in detection_supervision:
                bbox = obj[0].tolist()
                cls_id = obj[3]

                if cls_id == cls_names_inv["ball"]:
                    frame_ball[1] = {"bbox": bbox}  # Assuming single ball detection

            # Append frame data to tracks
            tracks["players"].append(frame_players)
            tracks["referees"].append(frame_referees)
            tracks["ball"].append(frame_ball)

        # Save to stub if needed
        if stub_path:
            try:
                with open(stub_path, "wb") as f:
                    pickle.dump(tracks, f)
            except Exception as e:
                print(f"Error saving stub file: {e}")

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):

        # Draws an ellipse at the bottom center of the bounding box.

        # Args:
        #    frame (numpy.ndarray): The current video frame.
        #    bbox (list): Bounding box [x_min, y_min, x_max, y_max].
        #    color (tuple): Color of the ellipse (B, G, R).
        #    track_id (int): Track ID for the player.

        # Returns:
        #    numpy.ndarray: The updated frame with the ellipse drawn.

        y2 = int(bbox[3])
        x_center, _ = get_bbox_center(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        # Defining rectangle bbox
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width / 2
        x2_rect = x_center + rectangle_width / 2
        y1_rect = (y2 - rectangle_height / 2) + 15
        y2_rect = (y2 + rectangle_height / 2) + 15

        # Drawing the rectangle
        if track_id:
            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED,
            )
            # Defining text bbox
            x1_text = x1_rect + 12
            if track_id > 99:
                x1_text -= 10  # A little space for large numbers

            # Drawing the text
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

        return frame

    def draw_annotations(self, video_frames, tracks):
        # Annotates video frames with tracking information.

        # Args:
        #    video_frames (list): List of video frames.
        #    tracks (dict): Tracking data for players, referees, and the ball.

        # Returns:
        #    list: Annotated video frames.

        output_video_frames = []
        for i, frame in enumerate(video_frames):
            frame = frame.copy()

            # Get tracking information for the current frame
            player_dict = tracks["players"][i]
            ball_dict = tracks["ball"][i]
            referee_dict = tracks["referees"][i]

            # Draw Players
            for track_id, player in player_dict.items():
                bbox = player["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0, 0, 255), track_id)

            # Draw Referees
            for _, referee in referee_dict.items():
                bbox = referee["bbox"]
                frame = self.draw_ellipse(frame, bbox, (0, 255, 255))

            output_video_frames.append(frame)

        return output_video_frames
