# Real-time Face Recognition Drone Surveillance System


## Executive Summary
This product will involve the process of sending a video feed through a specialized facial recognition software in order to correctly identify a face from a database. Currently, the surveillance industry requires the use of trained personnel in order to guard a property. With the introduction of this software application, a property owner could purchase some kind of video equipment coupled with this software and choose which people to allow on the property at certain times. When the feed records an image of an unrecognizable face it will trigger an alert on the server. One main advantage this software will allow is for the customer to greatly cut down on security man-hours. The software could also potentially make a property even more secure and allow for cameras in places that normally would not be possible by using drone surveillance.

When this product hits the shelves, not only the surveillance industry will be interested, but there could be many other potential applications. The system could be even be repurposed into finding a missing person which would simply require an image of a person to look for. Once the system is trained on the missing person, several drones could set out on an automated flight plan to find and pinpoint the location of the person via implanted GPS devices. On top of this, the network can be trained on more than just facial classifiers allowing recognition of license plates, vehicles, and many other kinds of objects.

While this product is fully functional and successfully identifies known and unknown faces given a video feed, there are still many areas that this could be extended. Given the time and effort, this product could become the future for object identification on any given surveillance system. 

# Product Description
Our goal was to develop an Ubuntu Linux application that connects a Parrot drone image feed to the VCU GPU (Graphical Processing Unit) array in order to do image recognition of various forms. At the moment, there is open source facial recognition library known as OpenFace (https://cmusatyalab.github.io/openface/), which was used in the product. However, certain additions needed to be made to allow the drone to recognize a face from a database and recognize when face is unknown. OpenFace provides functionality to reduce the dimensionality of a face to 128 data points, and we decided to use an ensemble of 30 classifiers that would vote on the result of classification.

# Architecture

![Diagram](https://raw.githubusercontent.com/asakaplan/Senior-Design/master/images/image2.png)

We designed a system that would be able to communicate between three computers, through port connections, to run the three aspects of the system.

1. Server - Receives video data, performs the facial recognition, and outputs video with facial recognition results.
2. Server Receive - Receives facial recognition data and video and outputs the information to the media player.
3. Run - Receives video from drone and outputs to server. The program has capability for automated and manual flight of the drone.

## Product Usage
Typical use of this product will first involve setting up a connection between a camera and the computer running the software and setting up a connection between the computer and the server.

Once the connections are set up, the user should open three separate terminals. One terminal on the server for the running the first command,

`python2 server.py`

Another on the local computer for running the second command

`python2 serverReceive.py`

And lastly, another terminal on the local computer for running the last command to start the drone program, 

`./run`

When all three programs are up and running, the user should point the camera at a face and click on the box surrounding the face that is displayed on the media player. The user should then enter a name to correspond with the face that is being added and click the button labeled “That’ll do!” to add the face to the directory containing the known faces. 

In order for the user to delete a person from the directory of known faces, they must manually navigate to the “Faces” directory and delete the folder labeled by the name associated with the added face.
Note: The more faces you add of the person, the more accurate the recognition will be.


## Demo Pictures

![Flying](https://raw.githubusercontent.com/asakaplan/Senior-Design/master/images/image3.png)

![Recognized Face](https://raw.githubusercontent.com/asakaplan/Senior-Design/master/images/image4.png)

![Unrecognized Face](https://raw.githubusercontent.com/asakaplan/Senior-Design/master/images/image5.png)


[VCU sponsored](https://raw.githubusercontent.com/asakaplan/Senior-Design/master/images/image1.png)
### Team Members
Asa Kaplan

Christopher Butler

Jacob Segal

### Advisor
Alberto Cano Rojas, Ph.D.

