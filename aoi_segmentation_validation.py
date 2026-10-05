import cv2

# Read an image
img = cv2.imread('sandbox/myplot.png')

# Display the image
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# This is a comment