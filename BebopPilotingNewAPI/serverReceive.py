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
def connectPort(port):
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect(("localhost",port))
    print('Connected with out ' + addr[0] + ":" + str(addr[1]))
    return soc

def main():
        try:
           os.remove(videoSend)
           os.remove(videoReceive)
        except Exception:
           pass

        global socketVideo, socketData, connVideo, connData

        socketVideo = connectPort(PORT_VIDEO)
        socketData = connectPort(PORT_DATA)

        i = 0
        os.remove(videoReceive)
        tempFile = open(videoReceive,"wb")
        while not exitCode:
            data = conn.recv(2**15)
            tempFile.write(data)

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
