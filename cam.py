import cv2 
cap = cv2.VideoCapture(0) 
cv2.namedWindow("camera") 
key = -1 
while(key < 0): 
    success, img = cap.read() 
    cv2.imshow("camera", img) 
    key = cv2.waitKey(1) 
cv2.destroyAllWindows()