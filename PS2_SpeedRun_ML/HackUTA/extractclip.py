import cv2

def time_to_frame(video_path, time_in_seconds):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the frames per second of the video
    frame_number = int(time_in_seconds * fps)  # Calculate frame number
    cap.release()
    return frame_number

def extract_clips(video_path, start_frame, end_frame, output_clip_path):
    cap = cv2.VideoCapture(video_path)
    
    # Set the starting point of the clip
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # Get the frames between start_frame and end_frame
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_clip_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width, frame_height))

    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break
        current_frame += 1
    
    cap.release()
    out.release()

# Example usage: inspect the video first
video_file = '4_missed_shots.webm'  # Change to your video file

# Example: Find the frame at 52 seconds
start_frame = time_to_frame(video_file, 52)
# print(f"Start Frame: {start_frame}")

end_frame = time_to_frame(video_file, 54)

# After inspecting, you can manually input your start_frame and end_frame
# start_frame = 100  # Replace with your noted start frame
# end_frame = 200    # Replace with your noted end frame
extract_clips(video_file, start_frame, end_frame, 'missed_shot_1.mp4')
