from utils import read_video, save_video, save_cropped_image
from classes import Tracker, TeamAssigner, PlayerBallAssigner
import numpy as np


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

    # Interpolate Ball positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

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

    # Assign Player Teams
    print("Assigning team colors...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])

    print("Assigning player teams...")
    for i, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(
                video_frames[i], track["bbox"], player_id
            )
            tracks["players"][i][player_id]["team"] = team
            tracks["players"][i][player_id]["team_color"] = team_assigner.team_colors[
                team
            ]

    # Assign Ball adquisition
    player_assigner = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_control.append(
                tracks["players"][frame_num][assigned_player]["team"]
            )
        else:
            team_ball_control.append(team_ball_control[-1])
    team_ball_control = np.array(team_ball_control)

    # Draw Object Tracks
    print("Drawing annotations...")
    output_video_frames = tracker.draw_annotations(
        video_frames, tracks, team_ball_control
    )

    # Save Video
    print("Saving video...")
    save_video(output_video_frames, "output_videos/output_video.avi")


if __name__ == "__main__":
    main()
