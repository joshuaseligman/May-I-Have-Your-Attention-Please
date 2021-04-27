import cv2 as cv
import time
import glob

def getMostRecent(paths):
    recent = 0
    for path in paths:
        num = int(path[path.rfind('\\') + 1 : path.index('.jpg')])
        recent = max(recent, num)
    return recent

pos = getMostRecent(glob.glob('positive/*.jpg')) + 1
neg = getMostRecent(glob.glob('negative/*.jpg')) + 1

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()

    scale = 1.25
    new_width = int(frame.shape[1] * scale)
    new_height = int(frame.shape[0] * scale)
    dim = (new_width, new_height)
    resized = cv.resize(frame, dim)
    cv.imshow('cam', resized)
    if cv.waitKey(1) == ord('q'):
        print('Quitting')
        break
    if cv.waitKey(1) == ord('p'):
        cv.imwrite('positive2/{}.jpg'.format(pos), resized)
        print('Saved Positive', pos)
        pos += 1
        time.sleep(1)
        print('Ready')
    if cv.waitKey(1) == ord('n'):
        cv.imwrite('negative/{}.jpg'.format(neg), resized)
        print('Saved Negative', neg)
        neg += 1
        time.sleep(1)
        print('Ready')
    
cap.release()
cv.destroyAllWindows()
