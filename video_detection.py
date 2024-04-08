import cv2
import numpy as np


def convert_avi_mp4(input, output): 
    import moviepy.editor as moviepy
    clip = moviepy.VideoFileClip(input)
    clip.write_videofile(output)
    print('Video was successfully converted to mp4!')


# Loading the cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")


# Defining a function that will do detections
def detect(gray, frame):
    faces = face_cascade.detectMultiScale(
        gray, 1.3, 5
    )  # image, scaleFactor, min_Neighbors
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y : y + h, x : x + w]
        roi_color = frame[y : y + h, x : x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        for ex, ey, ew, eh in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
    return frame


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
result = cv2.VideoWriter("test_video.avi", cv2.VideoWriter_fourcc(*"MJPG"), 60, size)


while True:
    ret, frame = cap.read()
    if ret == True:
        # Write the frame into the
        # file 'filename.avi'
        result.write(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detect(gray, frame)
        cv2.imshow("video", canvas)
        result.write(canvas)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        # Break the loop
    else:
        break

cap.release()
result.release()
cv2.destroyAllWindows()


convert_avi_mp4(input, output)
