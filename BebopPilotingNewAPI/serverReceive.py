import socket
import sys
import threading
import cv2
import time
import os
from server import PORT_VIDEO, PORT_DATA
HOST = '0.0.0.0'

videoReceive = "PLEASEWORK.avi"
exitCode = False
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

def connectPort(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect(("localhost",port))
    print('Connected with out ' + addr[0] + ":" + str(addr[1]))
    return soc

#Mouse callback function to get position and click event
def get_mouse_position_onclick(event, ix, iy, flags, param):
    global rects, curFrame, templates, faceFiles
    if event == cv2.EVENT_LBUTTONDOWN:
        for idx, ((x,y),(x2,y2),_,__) in enumerate(rects):
            if (x < ix) and (x2> ix) and (y < iy) and (y2 > iy):
                cv2.imshow('Recognized', curFrame[y:y2,x:x2])
                ix, iy = -1, -1
                create_new_text_window()
                cv2.imwrite('faces/' + templateName + '.png', curFrame[y:y2,x:x2])
                templates= [cv2.imread('faces/' + templateName + '.png', 0)]+templates
                faceFiles =[templateName+".png"]+faceFiles
                break
def main():
        try:
           os.remove(videoReceive)
        except Exception:
           pass

        global socketVideo, socketData, connVideo, connData

        socketVideo = connectPort(PORT_VIDEO)
        socketData = connectPort(PORT_DATA)
        global ix, iy, ievent, master, templateName, rects, texts, curFrame, templates, faceFiles
        rects = []
        texts = []
        manager = multiprocessing.Manager()

        threading.Thread(target=dataReceive).start()

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            for rect in rects:
                cv2.rectangle(frame,*(rect))

            for text in texts:
                cv2.putText(frame,*(text))
            # Display the resulting frame
            cv2.imshow('Video', frame)

            #Quit when q is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            i = 0
            os.remove(videoReceive)
            tempFile = open(videoReceive,"wb")

def dataReceive():
    global rects, texts
    try:
        os.remove(videoReceive)
    except:
        pass
    tempFile = open(videoReceive, "w")
    dataString = ""
    while not exitCode:
        dataVideo = socketVideo.recv(2**15)
        tempFile = open(videoReceive,"ab")
        tempFile.write(dataVideo)
        tempFile.close()
        dataData = socketData.recv(2**10)
        dataString +=dataData
        if dataString.count("]")>=2:
            dataString = dataString[1:] #strip first [
            ind = dataString.find("]]")
            dataTemp = dataString[:ind+2]
            dataString = dataString[ind+2:]
            [rects, texts] = exec(dataTemp)#Technically kinda vulnerable, but the connection itself is secure

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("In Interrupt")
        exitCode = True
        os.unlink(videoSend)
        s.close()
        cv2.destroyAllWindows()
        print("End of interrupt")
        raise
