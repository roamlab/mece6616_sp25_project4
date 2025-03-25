import cv2

def save_video_with_cv2(video_path, frames, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Ensure frames are converted to uint8 format for cv2
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()