import cv2 as cv

model = cv.CascadeClassifier('cascade/cascade.xml')

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    rectangles = model.detectMultiScale(frame, minNeighbors=60)
    
    for obj in rectangles:
        cv.rectangle(frame, (obj[0], obj[1]), (obj[0] + obj[2], obj[1] + obj[3]), (0, 255, 0), 2)
    cv.imshow('cam', frame)
    if cv.waitKey(1) == ord('q'):
        print('Quitting')
        break
    
cap.release()
cv.destroyAllWindows()
