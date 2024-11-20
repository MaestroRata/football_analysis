from utils import read_video, save_video
from trackers import Tracker


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

    # Draw Object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks)

    # Save Video
    print("Saving video...")
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
