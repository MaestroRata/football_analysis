import os
import pickle
from ultralytics import YOLO
import supervision as sv


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

    def draw_ellipse(self, frame, bbox, color, track_id)
        y2 = int(bbox[3])
        x2 
    
    def draw_annotations(self, video_frames, tracks):
        output_video_frames= []
        for i, frame in enumerate(video_frames):
            frame = frame.copy()
            
            player_dict = tracks['players'][i]
            ball_dict = tracks['ball'][i]
            referee_dict = tracks['referees'][i]
            
            # Draw Players
            for track_id, player in player_dict.items():
                frame = self.draw_ellipse(frame, player['bbox'],(0,0,255), track_id)