from utils import read_video, save_video, save_cropped_image
from trackers import Tracker
import cv2


def main():
    # Read Video
    print("Reading video...")
    video_frames = read_video("input_videos/08fd33_4.mp4")
    if not video_frames:
        print("Error: Failed to load video frames.")
        return
    print(f"Loaded {len(video_frames)} frames from the video.")

    # Initialize Tracker
    print("Initializing tracker...")
    tracker = Tracker("models/best.pt")

    # Get Object Tracks
    print("Tracking objects...")
    tracks = tracker.get_object_tracks(
        video_frames, read_from_stub=True, stub_path="stubs/track_stubs.pkl"
    )
    if not tracks:
        print("Error: Tracking failed or stub file could not be loaded.")
        return

    # Save cropped image of a player
    # print("Saving cropped image...")
    # for _, player in tracks["players"][0].items():
    #     bbox = player["bbox"]
    #     frame = video_frames[0]

    #     if save_cropped_image(frame, bbox, "output_videos/cropped_image.jpg"):
    #         print("Done. Cropped image saved")
    #     else:
    #         print("Error saving cropped image")
    #     break

    # Draw Object Tracks
    print("Drawing annotations...")
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    print("Saving video...")
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
