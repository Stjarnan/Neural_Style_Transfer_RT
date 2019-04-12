import argparse
import imutils
import time
import cv2

parse = argparse.ArgumentParser()
parse.add_argument("-m", "--model", required=True, help="Argument to pick model")
parse.add_argument("-i", "--image", required=True, help="Argument to pick the image we want to apply the style to")
args = vars(parse.parse_args())

# Load model
print("Loading transfer model...")
net = cv2.dnn.readNetFromTorch(args["model"])

# Load image, process it and get dimensions
img = cv2.imread(args["image"])
img = imutils.resize(img, width=600)
(img_height, img_weight) = img.shape[:2]

# Make blobs of image, set input to blob and perform forward pass
# Numbers are for mean subtraction
blob = cv2.dnn.blobFromImage(img, 1.0, (img_weight, img_height),
    (103.939, 116.779, 123.680), swapRB=False, crop=False)
net.setInput(blob)
output = net.forward()

# Reshape output, reverse mean subtraction and change channels (transpose)
output = output.reshape((3, output.shape[2], output.shape[3]))
output[0] += 103.939
output[1] += 116.779
output[2] += 123.680
output = output/255
output = output.transpose(1, 2, 0)

# Show results
cv2.imshow("Original", img)
cv2.imshow("Styled", output)
cv2.waitKey(0)