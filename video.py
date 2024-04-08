import cv2
import numpy as np



cap = cv2.VideoCapture(0)

if cap.isOpened() == False:
    print("Error reading video file")

# We need to set resolutions.
# so, convert them from float to integer.
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

size = (frame_width, frame_height)

# Below VideoWriter object will create
# a frame of above defined The output
# is stored in 'test_video.avi' file.
result = cv2.VideoWriter("test_video.avi", cv2.VideoWriter_fourcc(*"MJPG"), 30, size)


while True:
    ret, frame = cap.read()
    if ret == True:
        # Write the frame into the
        # file 'filename.avi'
        result.write(frame)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # Break the loop 
    else: 
        break

cap.release()
result.release()
cv2.destroyAllWindows()
