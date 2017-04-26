import socket
import threading
import cv2
import time
import os
import pickle
from subprocess import call
import Tkinter as tk
from constants import *
import numpy as np
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)
exitCode = False
notEnoughData = True
needsUpdating = True
#This simply returns and destroys the text box window
def get_window_text(_=None):
    global master
    templateName = e.get()
    socketData.send(pickle.dumps(cv2.resize(recFace, (96, 96))) +  boundary +  pickle.dumps(templateName) +  boundary)
    cv2.destroyWindow("Recognized")
    master.destroy()
    master.quit()
def isValid(val):
    return bool(val) and val!=float("nan")

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
    def destroyWindow():
        cv2.destroyWindow("Recognized")
        master.quit()
    master.protocol("WM_DELETE_WINDOW", destroyWindow)

    master.bind('<Return>', get_window_text)
    master.mainloop()

def connectPort(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print("Attempting connection with port %d"%port)
    soc.connect(("127.0.0.1",port))
    #print('Connected with out ' + addr[0] + ":" + str(addr[1]))
    return soc

#Mouse callback function to get position and click event
def get_mouse_position_onclick(event, ix, iy, flags, param):
    global rects, socketData, tempFrame, recFace, curFrame
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, ((x,y),(x2,y2),_,__) in enumerate(rects):
            if (x < ix) and (x2> ix) and (y < iy) and (y2 > iy):
                recFace = curFrame[y:y2,x:x2] #Or tempframe depending on what you're going for
                cv2.namedWindow('Recognized')
                cv2.imshow('Recognized', recFace)
                ix, iy = -1, -1
                #create_new_text_window()
                executor.submit(create_new_text_window)
                break
def main():
        try:
           os.remove(videoReceive)
        except Exception:
           pass

        global socketVideo, socketData, connVideo, connData,  curFrame

        socketData = connectPort(PORT_DATA)
        time.sleep(.5)
        socketVideo = connectPort(PORT_VIDEO)
        global ix, iy, ievent, master, templateName, rects, texts, curFrame, templates, faceFiles, curFrame, needsUpdating
        rects = []
        texts = []
        cv2.namedWindow('Video')

        cv2.setMouseCallback('Video', get_mouse_position_onclick)
        os.mkfifo(videoReceive)
        call(["perl -MFcntl -e 'fcntl(STDIN, 1031, 524288) or die $!' <> %s"%videoReceive], shell=True)
        threading.Thread(target=dataDataReceive).start()
        threading.Thread(target=dataVideoReceive).start()

        cap = cv2.VideoCapture(videoReceive)
        while not cap.isOpened():
            cap = cv2.VideoCapture(videoReceive)
            time.sleep(.1)
            print("Wait for the header")

        while not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),0)):
                print("Still waiting for the header")
                time.sleep(.1)
        print(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        while True:
            # Capture frame-by-frame
            flag, frame = cap.read()
            if not flag:
                print("Frame not ready")
                continue
            curFrame = np.copy(frame)
            #if needsUpdating:
            #    needsUpdating = False

            for rect in rects:
                tempRect = [(rect[0][0]-2,rect[0][1]-2), (rect[1][0]+2,rect[1][1]+2)]+list(rect[2:])
                cv2.rectangle(frame,*(tempRect))

            for text in texts:
                cv2.putText(frame,*(text))
            # Display the resulting frame
            cv2.imshow('Video', frame)

            #Quit when q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
def dataDataReceive():
    global rects, texts,  socketData, needsUpdating, tempFrame
    dataString = ""
    while not exitCode:
        dataData = socketData.recv(2**10)
        dataString +=dataData
        if dataString.count("]")>=2:
            needsUpdating = True
            ind = dataString.find("]]")
            dataTemp = dataString[:ind+2]
            dataString = dataString[ind+2:]
            [rects, texts] = eval(dataTemp)#Technically kinda vulnerable, but the connection itself is secure
            tempFrame = curFrame
def dataVideoReceive():
    global notEnoughData,  socketVideo
    totalData = 0
    tempFile = open(videoReceive, "w+b")


    while not exitCode:
        dataVideo = socketVideo.recv(2**15)
        tempFile.write(dataVideo)
    tempFile.close()
if __name__ == '__main__':
    try:
        main()
    except:
        try:
            print("In Interrupt")
            exitCode = True
            os.unlink(videoSend)
            s.close()
            cv2.destroyAllWindows()
            print("End of interrupt")
        finally:
            raise
