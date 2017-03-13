import socket
import sys
import threading
import cv2
import time
import os
HOST = ''
PORT = 8000
PORT_VIDEO = 8001
PORT_DATA = 8002

videoSend = "videoOut.avi"
videoReceive = "videoTemp.avi"
exitCode = False

def setupSocket(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR,1)
    print( 'Socket created')
    try:
            soc.bind((HOST,PORT))
    except socket.error as msg:
            print('Bind failed. Error Code : ' + str(msg))
            sys.exit()
    print('Socket bind complete')
    soc.listen(10)
    print('Socket now listening')
    conn, addr = s.accept()
    print('Connected with ' + addr[0] + ":" + str(addr[1]))
    return soc, conn

def isValid(val):
    return bool(val) and val!=float("nan")

def main():
        try:
           os.remove(videoSend)
           os.remove(videoReceive)
        except Exception:
           pass
        os.mkfifo(videoSend)

        global s, conn
        s, conn = setupSocket(PORT)
        global sOut
        sOut, connOut = setupSocket(PORT_OUT)
        print('Connected with out ' + addr[0] + ":" + str(addr[1]))

        i = 0
        threading.Thread(target=dataReceive).start()
        global cap
        cap = cv2.VideoCapture(videoReceive)
        while not cap.isOpened():
            cap = cv2.VideoCapture(videoReceive)
            time.sleep(.1)
            print("Wait for the header")

        while not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT),0)) or not isValid(round(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH),0)):
                print("Still waiting for the header")
                time.sleep(.1)
        print(cap.get(cv2.cv.CV_CAP_PROP_FPS))
        sourceFPS = int(round(cap.get(cv2.cv.CV_CAP_PROP_FPS),0))
        sourceDimensions = (int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
        print("FOURCC:",cap.get(cv2.cv.CV_CAP_PROP_FOURCC))
        global outputVideo
        outputVideo = cv2.VideoWriter(videoSend, cv2.cv.CV_FOURCC(*'XVID'), sourceFPS, sourceDimensions, 1)
        print("Success: ", outputVideo.isOpened())
        threading.Thread(target=dataSend).start()

        pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
        while True:
            flag, frame = cap.read()
            if flag:
                # The frame is ready and already captured
                #cv2.imshow('video', frame)
                pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)

                outputVideo.write(cv2.flip(frame,0))
                pos_frame = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                if(pos_frame%1000==0):print(str(pos_frame)+" frames")
            else:
                # The next frame is not ready, so we try to read it again
                cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos_frame-1)
                print("frame is not ready")
                # It is better to wait for a while for the next frame to be ready
                time.sleep(.5)
            if cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES) == cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT):
                # If the number of captured frames is equal to the total number of frames,
                # we stop
                break



def dataReceive():
    tempFile = open(videoReceive, "w")
    tempFile.close() #Delete old video
    while not exitCode:
        data = conn.recv(2**15)
        tempFile = open(videoReceive,"ab")
        #print("Heyo here's some data: %d"%len(data))
        tempFile.write(data)
        tempFile.close()
        #Process data
        #conn.send(data)

def dataSend():
    lastLength = 0
    outp = None
    while not exitCode:
        print("Trying to open output file")
        try:
            outp = open(videoSend, "r+b")
            break
        except Exception as e:
            time.sleep(.1)

    for line in outp:
        #outp is a fifo, so this will continue to go until the program is exited
        if exitCode:break
        connOut.send(line)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("In Interrupt")
        exitCode = True
        os.unlink(videoSend)
        s.close()
        cap.release()
        outputVideo.release()
        cv2.destroyAllWindows()
        print("End of interrupt")
        raise
