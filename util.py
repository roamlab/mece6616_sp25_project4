import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

def save_video_with_cv2(video_path, frames, fps=30):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame in frames:
        # Ensure frames are converted to uint8 format for cv2
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()

def show_video(video_path, width=10, height=6):
    # Read video
    cap = cv2.VideoCapture(video_path)
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    # Check if any frames were read
    if not frames:
        print("Error: No frames were read from the video")
        return None
    # Create animation
    fig = plt.figure(figsize=(width, height))
    plt.axis('off')
    im = plt.imshow(frames[0])
    def update(i):
        im.set_array(frames[i])
        return im,
    # Calculate interval based on original video FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = 1000/fps if fps > 0 else 50  # default to 20fps if fps not available
    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval,
        blit=True
    )
    plt.close()
    # Display in notebook
    return HTML(ani.to_html5_video())





















