import cv2
import numpy as np
from ultralytics import SAM
import supervision as sv

def track_basketball_elements(video_path, output_path):
    # Load the SAM model
    model = SAM("sam2_b.pt")

    # Initialize the video capture
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Define prompts for different elements
    prompts = {
        "basketball": [width // 2, height // 2],  # Center point
        "player": [width // 4, height * 3 // 4],  # Bottom-left area
        "hoop": [width * 3 // 4, height // 4],  # Top-right area
        "court_lines": [[0, 0], [width, 0], [width, height], [0, height]]  # Court boundaries
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run SAM on the frame
        results = model(frame, points=list(prompts.values()), labels=[1] * len(prompts))

        # Process and visualize results
        annotated_frame = frame.copy()
        for mask in results[0].masks:
            annotated_frame = sv.mask_on_image(annotated_frame, mask.data)

        # Write the annotated frame
        out.write(annotated_frame)

    # Release resources
    cap.release()
    out.release()

if __name__ == "__main__":
    video_path = "/workspace/data/raw/basketball_video.mp4"
    output_path = "/workspace/data/processed/tracked_basketball_video.mp4"
    track_basketball_elements(video_path, output_path)