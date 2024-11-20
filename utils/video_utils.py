import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    # Check if the video file opened correctly
    if not cap.isOpened():
        print("Error: Unable to open video file.")
        return frames

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Video ended. Exiting ...")
            break

        # Add the frame to the list
        frames.append(frame)

    # Release the video capture object
    cap.release()

    # Return the list of frames
    return frames


def save_video(output_video_frames, output_video_path):
    # Check if the frame list is empty
    if not output_video_frames:
        print("Error: No frames to save.")
        return

    # Define video format and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
    height, width = output_video_frames[0].shape[:2]  # Frame dimensions
    fps = 24  # Frame rate (adjust as needed)

    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video file
    total_frames = len(output_video_frames)
    for i, frame in enumerate(output_video_frames):
        out.write(frame)
        print(f"Saved frame {i + 1}/{total_frames}")

    # Release the VideoWriter object
    out.release()
    print(f"Video saved to {output_video_path}")
