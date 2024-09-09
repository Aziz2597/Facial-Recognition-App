import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

aziz_image = face_recognition.load_image_file(r"C:\Users\azizu\Desktop\Facial_Recognition\faces\aziz.jpg")
# Converting the facial features into numerical data (128-dimensional encodings).
aziz_encoding = face_recognition.face_encodings(aziz_image)[0] 

sohaib_image = face_recognition.load_image_file(r"C:\Users\azizu\Desktop\Facial_Recognition\faces\sohaib.jpg")
sohaib_encoding = face_recognition.face_encodings(sohaib_image)[0]

known_face_encodings = [aziz_encoding, sohaib_encoding]
known_face_names = ["Aziz", "Sohaib"]

students = known_face_names.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwr = csv.writer(f)

while True:
    ret, frame = video_capture.read()

    # Resize the frame to 1/4th size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the frame from BGR (OpenCV format) to RGB (face_recognition format)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # Calculate the Euclidian distance between the current face and each known face encoding
        face_distances = face_recognition.face_distance(
            known_face_encodings, face_encoding
        )

        # Find the index of the closest matching face
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(
                frame,
                name + " Present",
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType,
            )

            # If the person is still in the student list, mark them as present
            if name in students:
                students.remove(name) # Remove student from the list to avoid double counting
                current_time = now.strftime("%H:%M:%S")
                lnwr.writerow([name, current_time])

    cv2.imshow("Attendance", frame) # Display the video frame with text overlays
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()

# Optimizations that can be done:
# Database Optimization: Storing face encodings in a database and indexing them can speed up lookup and comparison.
# GPU Acceleration: Using hardware acceleration (e.g., via CUDA) to speed up face detection and encoding for large datasets.
# Clustering: Use clustering algorithms to group similar face encodings, reducing the number of comparisons needed during recognition.
# Preprocessing: Filtering frames with no faces or unnecessary objects before applying face recognition can reduce computational overhead.
# Parallelization: Use parallel processing to handle multiple frames or users simultaneously. 

# Libraries for Parallel Processing:
# concurrent.futures: A standard Python library that provides a simple interface for parallel execution using threads (ThreadPoolExecutor) or processes (ProcessPoolExecutor).
# multiprocessing: This library allows the execution of code in separate processes (true parallelism in Python) and is well-suited for CPU-bound tasks.

#ThreadPoolExecutor: The ThreadPoolExecutor manages a pool of threads and runs the process_frame function in parallel for each frame.
#executor.submit(): Each frame is submitted as a task to the thread pool, allowing multiple frames to be processed concurrently.
