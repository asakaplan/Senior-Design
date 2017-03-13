import cv2
import numpy as np
import sys
import tkinter as tk
import glob

#global variables
cascadePath = sys.argv[1]
#templatePath = sys.argv[2]
faceCascade = cv2.CascadeClassifier(cascadePath)
template = 0
sizes = [80, 100, 120, 140, 160, 180, 200, 220, 240, 260]
threshold = 0.725
ix, iy = -1, -1
ievent = 0
templateName = ''
master = 0
e = 0

#video feed source and windows
video_capture = cv2.VideoCapture(0)
cv2.namedWindow('Video')
cv2.namedWindow('Recognized')


#This simply returns and destroys the text box window
def get_window_text():
    global templateName
    templateName = e.get()
    master.destroy()


#Freezes current process to enter text for clicked image
def create_new_text_window():
    global master
    global e
    master = tk.Tk()
    e = tk.Entry(master)
    e.pack()
    e.focus_set()
    b = tk.Button(master, text="That'll do!", width=10, command=get_window_text)
    b.pack()
    master.mainloop()


#Mouse callback function to get position and click event
def get_mouse_position_onclick(event, x, y, flags, param):
    global ix, iy
    global ievent
    if event == cv2.EVENT_LBUTTONDOWN:
        ix, iy = x, y
        ievent = event


def execute_main_loop():
    global ix, iy
    global ievent
    global master
    global templateName
    xTemp = 0
    yTemp = 0
    currentSize = 0

    while True:
        # Capture frame-by-frame
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        #Get all face images in face directory
        for filename in glob.glob('faces/*.png'):
            template = cv2.imread(filename, 0)
            
            # Match a Template:
            for size in sizes:
                tempTemplate = cv2.resize(template, (size, size))
                res = cv2.matchTemplate(gray, tempTemplate, cv2.TM_CCOEFF_NORMED)
                loc = np.where(res >= threshold)
                inner = 0
                xTemp = 0
                yTemp = 0
                currentSize = 0
                for point in zip(*loc[::-1]):
                    detection = frame[point[1]:point[1] + size, point[0]:point[0] + size]
                    # NOTE: its img[y: y + h, x: x + w]
                    cv2.rectangle(frame, point, (point[0] + size, point[1] + size), (0, 0, 255), 4)
                    personname = filename[6:-4]
                    cv2.putText(frame, personname, (point[0], point[1]), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                    xTemp = point[0]
                    yTemp = point[1]
                    currentSize = size
                    cv2.imwrite('detections/detection_' + str(point[0] - point[0] % 100) + '_' + str(point[1] - point[1] % 100) + '_' + filename, detection)
                    inner = 1
                    break
                #if there was a rectangle drawn on the previous size no need to keep going though the larger sizes
                if inner == 1:
                    break

            #Get faces
            faces = faceCascade.detectMultiScale(
                gray,
                scaleFactor=1.5,
                minNeighbors=3,
                minSize=(80, 80),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            # Draw a rectangle around the faces
            for (x, y, w, h) in faces:
                if xTemp < 1 or ((x > xTemp - currentSize / 4) and (x + w < xTemp + w + currentSize / 4) and (y > yTemp - currentSize / 4) and (y + h < yTemp + h + currentSize / 4)):
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 1)
                    possibleFace = frame[y:y + h, x:x + w]
                    #cv2.imwrite('templates/template_' + str(x - x%100) + '_' + str(y - y%100) + '.png', possibleFace)
                    if (ievent == cv2.EVENT_LBUTTONDOWN) and (x < ix) and (x + w > ix) and (y < iy) and (y + h > iy):
                        cv2.imshow('Recognized', possibleFace)
                        ix, iy = -1, -1
                        create_new_text_window()
                        cv2.imwrite('faces/' + templateName + '.png', possibleFace)


            # Display the resulting frame
            cv2.imshow('Video', frame)

        #Quit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cv2.setMouseCallback('Video', get_mouse_position_onclick)
execute_main_loop()
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
