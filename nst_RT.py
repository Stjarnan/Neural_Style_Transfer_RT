import imutils
from imutils.video import VideoStream
from imutils import paths
import itertools
import argparse
import cv2

# Runtime argument
parse = argparse.ArgumentParser()
parse.add_argument("-m", "--model", required=True, help="Path to directory containing models")
arguments = vars(parse.parse_args())

# Get all models (they all end with .t7)
# Create a list of the models
# Use itertools cycle function to loop/restart loop
model_directory = paths.list_files(arguments["model"], validExts=(".t7"))
model_directory = sorted(list(model_directory))

model = list(zip(range(0, len(model_directory)), (model_directory)))

iterate_model = itertools.cycle(model)
(model_id, model_path) = next(iterate_model)

# Load the style transfer model
# Initialize video stream
print("Loading model..")
net = cv2.dnn.readNetFromTorch(model_path)

print("Starting camera..")
cam = VideoStream(src=0).start()

# Loop frames in video
while True:
    frame = cam.read()

    # resize frame and get shape
    frame = imutils.resize(frame, width=600)
    original = frame.copy()
    (height, width) = frame.shape[:2]

    # Blob the frame, set input and perform forward pass
    blob = cv2.dnn.blobFromImage(frame, 1.0, (width, height),
        (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # Reshape output, revert mean subtraction and transpose output
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output = output / 255.0
    output = output.transpose(1, 2, 0)

    cv2.imshow("Style Transfer", output)
    # Capture keypress
    key = cv2.waitKey(1) & 0xFF

    if key == ord("n"):
        (model_id, model_path) = next(iterate_model)
        net = cv2.dnn.readNetFromTorch(model_path)
    elif key == ord("q"):
        break

print("Shutting down..")
cv2.destroyAllWindows()
cam.stop()