/*
Copyright (C) 2014 Parrot SA

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
* Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in
the documentation and/or other materials provided with the
distribution.
* Neither the name of Parrot nor the names
of its contributors may be used to endorse or promote products
derived from this software without specific prior written
permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
SUCH DAMAGE.
*/
/**
* @file BebopPiloting.c
* @brief This file contains sources about basic arsdk example sending commands to a bebop drone for piloting it and make it jump it and receiving its battery level
* @date 15/01/2015
*/

/*****************************************
*
*             include file :
*
*****************************************/
#include <stdlib.h>
#include <curses.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/ioctl.h>
#include <netinet/in.h>
#include <netdb.h>

#include <sys/socket.h>

#include <libARSAL/ARSAL.h>
#include <libARController/ARController.h>
#include <libARDiscovery/ARDiscovery.h>

#include "BebopPiloting.h"
#include "ihm.h"


/*****************************************
*
*             define :
*
*****************************************/
#define TAG "SDKExample"

#define ERROR_STR_LENGTH 2048

#define BEBOP_IP_ADDRESS "192.168.42.1"
#define BEBOP_DISCOVERY_PORT 44444

#define DISPLAY_WITH_MPLAYER 1

#define FIFO_DIR_PATTERN "/tmp/arsdk_XXXXXX"
#define FIFO_NAME "arsdk_fifo"

#define IHM


#define NUM_COMMANDS 2
#define SPEED_BUFFER_SIZE 5
#define SOCKET_BUFFER_SIZE 32768


/*****************************************
*
*             private header:
*
****************************************/


/*****************************************
*
*             implementation :
*
*****************************************/

static char fifo_dir[] = FIFO_DIR_PATTERN;
static char fifo_name[128] = "";

int gIHMRun = 1;
char gErrorStr[ERROR_STR_LENGTH];
IHM_t *ihm = NULL;

FILE *videoOut = NULL;
int frameNb = 0;
ARSAL_Sem_t stateSem;
int isBebop2 = 0;
int sockfd=0;
char bufferRead[SOCKET_BUFFER_SIZE] = {0};
char bufferWrite[SOCKET_BUFFER_SIZE] = {0};
boolean stopCommand = false;
char* ipAddress = "127.0.0.1";
int port = 8000;

static void signal_handler(int signal)
{
  gIHMRun = 0;
}

int main (int argc, char *argv[])
{
  // local declarations
  int failed = 0;
  ARDISCOVERY_Device_t *device = NULL;
  ARCONTROLLER_Device_t *deviceController = NULL;
  eARCONTROLLER_ERROR error = ARCONTROLLER_OK;
  eARCONTROLLER_DEVICE_STATE deviceState = ARCONTROLLER_DEVICE_STATE_MAX;
  pid_t child = 0;

  /* Set signal handlers */
  struct sigaction sig_action = {
    .sa_handler = signal_handler,
  };

  int ret = sigaction(SIGINT, &sig_action, NULL);
  if (ret < 0)
  {
    ARSAL_PRINT(ARSAL_PRINT_ERROR, "ERROR", "Unable to set SIGINT handler : %d(%s)",
    errno, strerror(errno));
    return 1;
  }
  ret = sigaction(SIGPIPE, &sig_action, NULL);
  if (ret < 0)
  {
    ARSAL_PRINT(ARSAL_PRINT_ERROR, "ERROR", "Unable to set SIGPIPE handler : %d(%s)",
    errno, strerror(errno));
    return 1;
  }


  if (mkdtemp(fifo_dir) == NULL)
  {
    ARSAL_PRINT(ARSAL_PRINT_ERROR, "ERROR", "Mkdtemp failed.");
    return 1;
  }
  snprintf(fifo_name, sizeof(fifo_name), "%s/%s", fifo_dir, FIFO_NAME);

  if(mkfifo(fifo_name, 0666) < 0)
  {
    ARSAL_PRINT(ARSAL_PRINT_ERROR, "ERROR", "Mkfifo failed: %d, %s", errno, strerror(errno));
    return 1;
  }

  ARSAL_Sem_Init (&(stateSem), 0, 0);

  ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "Select your Bebop : Bebop (1) ; Bebop2 (2)");
  isBebop2 = 0;

  if(isBebop2)
  {
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "-- Bebop 2 Piloting --");
  }
  else
  {
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "-- Bebop Piloting --");
  }

  if (!failed)
  {
    if (DISPLAY_WITH_MPLAYER)
    {
      // fork the process to launch mplayer
      if ( (child = fork()) == 0)
      {
        execlp("xterm", "xterm", "-e", "mplayer", "-demuxer",  "h264es", fifo_name, "-benchmark", "-really-quiet", NULL);
        ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "Missing mplayer, you will not see the video. Please install mplayer and xterm.");
        return -1;
      }
    }

    if (DISPLAY_WITH_MPLAYER)
    {
      videoOut = fopen(fifo_name, "w");
    }
  }

  #ifdef IHM
  ihm = IHM_New (&onInputEvent);
  if (ihm != NULL)
  {
    gErrorStr[0] = '\0';
    ARSAL_Print_SetCallback (customPrintCallback); //use a custom callback to print, for not disturb ncurses IHM

    if(isBebop2)
    {
      IHM_PrintHeader (ihm, "-- Bebop 2 Piloting --");
    }
    else
    {
      IHM_PrintHeader (ihm, "-- Bebop Piloting --");
    }
  }
  else
  {
    ARSAL_PRINT (ARSAL_PRINT_ERROR, TAG, "Creation of IHM failed.");
    failed = 1;
  }
  #endif

  // create a discovery device
  if (!failed)
  {
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "- init discovey device ... ");
    eARDISCOVERY_ERROR errorDiscovery = ARDISCOVERY_OK;

    device = ARDISCOVERY_Device_New (&errorDiscovery);

    if (errorDiscovery == ARDISCOVERY_OK)
    {
      ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "    - ARDISCOVERY_Device_InitWifi ...");
      // create a Bebop drone discovery device (ARDISCOVERY_PRODUCT_ARDRONE)

      if(isBebop2)
      {
        errorDiscovery = ARDISCOVERY_Device_InitWifi (device, ARDISCOVERY_PRODUCT_BEBOP_2, "bebop2", BEBOP_IP_ADDRESS, BEBOP_DISCOVERY_PORT);
      }
      else
      {
        errorDiscovery = ARDISCOVERY_Device_InitWifi (device, ARDISCOVERY_PRODUCT_ARDRONE, "bebop", BEBOP_IP_ADDRESS, BEBOP_DISCOVERY_PORT);
      }

      if (errorDiscovery != ARDISCOVERY_OK)
      {
        failed = 1;
        ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "Discovery error :%s", ARDISCOVERY_Error_ToString(errorDiscovery));
      }
    }
    else
    {
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "Discovery error :%s", ARDISCOVERY_Error_ToString(errorDiscovery));
      failed = 1;
    }
  }

  // create a device controller
  if (!failed)
  {
    deviceController = ARCONTROLLER_Device_New (device, &error);

    if (error != ARCONTROLLER_OK)
    {
      ARSAL_PRINT (ARSAL_PRINT_ERROR, TAG, "Creation of deviceController failed.");
      failed = 1;
    }
    else
    {
      IHM_setCustomData(ihm, deviceController);
    }
  }

  if (!failed)
  {
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "- delete discovey device ... ");
    ARDISCOVERY_Device_Delete (&device);
  }

  // add the state change callback to be informed when the device controller starts, stops...
  if (!failed)
  {
    error = ARCONTROLLER_Device_AddStateChangedCallback (deviceController, stateChanged, deviceController);

    if (error != ARCONTROLLER_OK)
    {
      ARSAL_PRINT (ARSAL_PRINT_ERROR, TAG, "add State callback failed.");
      failed = 1;
    }
  }

  // add the command received callback to be informed when a command has been received from the device
  if (!failed)
  {
    error = ARCONTROLLER_Device_AddCommandReceivedCallback (deviceController, commandReceived, deviceController);

    if (error != ARCONTROLLER_OK)
    {
      ARSAL_PRINT (ARSAL_PRINT_ERROR, TAG, "add callback failed.");
      failed = 1;
    }
  }
  // add the frame received callback to be informed when a streaming frame has been received from the device
  if (!failed)
  {
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "- set Video callback ... ");
    error = ARCONTROLLER_Device_SetVideoStreamCallbacks (deviceController, decoderConfigCallback, didReceiveFrameCallback, NULL , NULL);

    if (error != ARCONTROLLER_OK)
    {
      failed = 1;
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "- error :%s", ARCONTROLLER_Error_ToString(error));
    }
  }

  if (!failed)
  {
    IHM_PrintInfo(ihm, "Connecting ...");
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "Connecting ...");
    error = ARCONTROLLER_Device_Start (deviceController);

    if (error != ARCONTROLLER_OK)
    {
      failed = 1;
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "- error :%s", ARCONTROLLER_Error_ToString(error));
    }
  }
  if (!failed)
  {
    // wait state update update
    ARSAL_Sem_Wait (&(stateSem));

    deviceState = ARCONTROLLER_Device_GetState (deviceController, &error);

    if ((error != ARCONTROLLER_OK) || (deviceState != ARCONTROLLER_DEVICE_STATE_RUNNING))
    {
      failed = 1;
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "- deviceState :%d", deviceState);
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "- error :%s", ARCONTROLLER_Error_ToString(error));
    }
  }
  // Set settings like safety parameters
  if (!failed)
  {
    //Tell the drone to return to start after 5 seconds of disconnect
    deviceController->aRDrone3->sendGPSSettingsHomeType(deviceController->aRDrone3, ARCOMMANDS_ARDRONE3_GPSSETTINGS_HOMETYPE_TYPE_TAKEOFF);
    deviceController->aRDrone3->sendGPSSettingsReturnHomeDelay(deviceController->aRDrone3, (uint16_t)2);
    //Set boundary box and turn it on
    deviceController->aRDrone3->sendPilotingSettingsMaxAltitude(deviceController->aRDrone3, (float)3);
    deviceController->aRDrone3->sendPilotingSettingsMaxDistance(deviceController->aRDrone3, (float)3);
    deviceController->aRDrone3->sendPilotingSettingsNoFlyOverMaxDistance(deviceController->aRDrone3, (uint8_t)1);
    //Change type of stream to be best for gpu computations
    deviceController->aRDrone3->sendMediaStreamingVideoStreamMode(deviceController->aRDrone3, ARCOMMANDS_ARDRONE3_MEDIASTREAMING_VIDEOSTREAMMODE_MODE_HIGH_RELIABILITY_LOW_FRAMERATE);
    deviceController->aRDrone3->sendPictureSettingsVideoResolutions(deviceController->aRDrone3, ARCOMMANDS_ARDRONE3_PICTURESETTINGS_VIDEORESOLUTIONS_TYPE_REC720_STREAM720);



  }
  if (!failed)
  {

    setupSocket();
    pthread_t thread;
    int result_code = pthread_create(&thread, NULL, listenSocket, (void*)"TEST");
    assert( !result_code );
  }


  // send the command that tells to the Bebop to begin its streaming
  if (!failed)
  {
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "- send StreamingVideoEnable ... ");
    error = deviceController->aRDrone3->sendMediaStreamingVideoEnable (deviceController->aRDrone3, 1);
    if (error != ARCONTROLLER_OK)
    {
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "- error :%s", ARCONTROLLER_Error_ToString(error));
      failed = 1;
    }
  }

  if (!failed)
  {
    IHM_PrintInfo(ihm, "Running ... ('t' to takeoff ; Spacebar to land ; 'e' for emergency ; Arrow keys and ('r','f','d','g') to move ; 'q' to quit)");

    #ifdef IHM
    while (gIHMRun)
    {
      usleep(50);
    }
    #else
    int i = 20;
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "- sleep 20 ... ");
    while (gIHMRun && i--)
    sleep(1);
    #endif
  }

  #ifdef IHM
  IHM_Delete (&ihm);
  #endif

  // we are here because of a disconnection or user has quit IHM, so safely delete everything
  if (deviceController != NULL)
  {


    deviceState = ARCONTROLLER_Device_GetState (deviceController, &error);
    if ((error == ARCONTROLLER_OK) && (deviceState != ARCONTROLLER_DEVICE_STATE_STOPPED))
    {
      IHM_PrintInfo(ihm, "Disconnecting ...");
      ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "Disconnecting ...");

      error = ARCONTROLLER_Device_Stop (deviceController);

      if (error == ARCONTROLLER_OK)
      {
        // wait state update update
        ARSAL_Sem_Wait (&(stateSem));
      }
    }

    IHM_PrintInfo(ihm, "");
    ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "ARCONTROLLER_Device_Delete ...");
    ARCONTROLLER_Device_Delete (&deviceController);
    shutdown(sockfd, 2);
    if (DISPLAY_WITH_MPLAYER)
    {
      fflush (videoOut);
      fclose (videoOut);

      if (child > 0)
      {
        kill(child, SIGKILL);
      }
    }
  }

  ARSAL_Sem_Destroy (&(stateSem));

  unlink(fifo_name);
  rmdir(fifo_dir);

  ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "-- END --");

  return EXIT_SUCCESS;
}

void setupSocket()
{
  int portno;
  struct sockaddr_in serv_addr;
  struct hostent *server;

  portno = 8000;
  sockfd = socket(AF_INET, SOCK_STREAM, 0);
  if (sockfd < 0)
  IHM_PrintInfoXY2(ihm, 14, 0, "ERROR opening socket", 0, 0);
  server = gethostbyname("127.0.0.1");
  if (server == NULL) {
    IHM_PrintInfoXY2(ihm, 15, 0,"ERROR, no such host\n", 0, 0);
  }
  bzero((char *) &serv_addr, sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  bcopy((char *)server->h_addr,
  (char *)&serv_addr.sin_addr.s_addr,
  server->h_length);
  serv_addr.sin_port = htons(portno);
  if (connect(sockfd,(struct sockaddr *) &serv_addr,sizeof(serv_addr)) < 0)
  IHM_PrintInfoXY2(ihm, 16, 0,"ERROR connecting",0,0);

}

void writeSocket(uint8_t* data, int length){


    IHM_PrintInfoF(ihm, "Writing: %d", length);
  int n = write(sockfd,data, length);
  if (n < 0)
  IHM_PrintInfoXY2(ihm, 17, 0,"ERROR writing to socket",0,0);
}

void* listenSocket(void* argument)
{
  int count = 0;
  IHM_PrintBuffer(ihm, "WAITING");
  while(1) {
    bzero(bufferRead,SOCKET_BUFFER_SIZE);
    int n = recv(sockfd,bufferRead,SOCKET_BUFFER_SIZE, MSG_WAITALL);
    count+=n;
    //IHM_PrintBuffer(ihm, "Testicles");
    if(n>0){
      IHM_PrintBufferSize(ihm, n);
      fwrite(bufferRead, n, 1, videoOut);
      fflush (videoOut);
    }
  }
  return "Testing";
}
/*****************************************
*
*             private implementation:
*
****************************************/

// called when the state of the device controller has changed
void stateChanged (eARCONTROLLER_DEVICE_STATE newState, eARCONTROLLER_ERROR error, void *customData)
{
  ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "    - stateChanged newState: %d .....", newState);

  switch (newState)
  {
    case ARCONTROLLER_DEVICE_STATE_STOPPED:
    ARSAL_Sem_Post (&(stateSem));
    //stop
    gIHMRun = 0;

    break;

    case ARCONTROLLER_DEVICE_STATE_RUNNING:
    ARSAL_Sem_Post (&(stateSem));
    break;

    default:
    break;
  }
}
boolean wasTakingOff = false;
boolean wasMoving = false;
int counter1 = 0;
int counter2 = 0;

float bufferSpeed[SPEED_BUFFER_SIZE*3] = {0.0f}; // multiply by 3 for x y z
int bufferPosition = 0;
float accumulatedSpeed[3] = {0.0f};
float dXSum=0, dYSum=0, dZSum = 0;

// called when a command has been received from the drone
void commandReceived (eARCONTROLLER_DICTIONARY_KEY commandKey, ARCONTROLLER_DICTIONARY_ELEMENT_t *elementDictionary, void *customData)
{

  ARCONTROLLER_Device_t *deviceController = customData;

  if (deviceController != NULL)
  {
    // if the command received is a battery state changed
    if (commandKey == ARCONTROLLER_DICTIONARY_KEY_COMMON_COMMONSTATE_BATTERYSTATECHANGED)
    {
      ARCONTROLLER_DICTIONARY_ARG_t *arg = NULL;
      ARCONTROLLER_DICTIONARY_ELEMENT_t *singleElement = NULL;

      if (elementDictionary != NULL)
      {
        // get the command received in the device controller
        HASH_FIND_STR (elementDictionary, ARCONTROLLER_DICTIONARY_SINGLE_KEY, singleElement);

        if (singleElement != NULL)
        {
          // get the value
          HASH_FIND_STR (singleElement->arguments, ARCONTROLLER_DICTIONARY_KEY_COMMON_COMMONSTATE_BATTERYSTATECHANGED_PERCENT, arg);

          if (arg != NULL)
          {
            // update UI
            batteryStateChanged (arg->value.U8);
          }
          else
          {
            ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "arg is NULL");
          }
        }
        else
        {
          ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "singleElement is NULL");
        }
      }
      else
      {
        ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "elements is NULL");
      }
    }
  }

  if (commandKey == ARCONTROLLER_DICTIONARY_KEY_COMMON_COMMONSTATE_SENSORSSTATESLISTCHANGED)
  {
    ARCONTROLLER_DICTIONARY_ARG_t *arg = NULL;

    if (elementDictionary != NULL)
    {
      ARCONTROLLER_DICTIONARY_ELEMENT_t *dictElement = NULL;
      ARCONTROLLER_DICTIONARY_ELEMENT_t *dictTmp = NULL;

      eARCOMMANDS_COMMON_COMMONSTATE_SENSORSSTATESLISTCHANGED_SENSORNAME sensorName = ARCOMMANDS_COMMON_COMMONSTATE_SENSORSSTATESLISTCHANGED_SENSORNAME_MAX;
      int sensorState = 0;

      HASH_ITER(hh, elementDictionary, dictElement, dictTmp)
      {
        // get the Name
        HASH_FIND_STR (dictElement->arguments, ARCONTROLLER_DICTIONARY_KEY_COMMON_COMMONSTATE_SENSORSSTATESLISTCHANGED_SENSORNAME, arg);
        if (arg != NULL)
        {
          sensorName = arg->value.I32;
        }
        else
        {
          ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "arg sensorName is NULL");
        }

        // get the state
        HASH_FIND_STR (dictElement->arguments, ARCONTROLLER_DICTIONARY_KEY_COMMON_COMMONSTATE_SENSORSSTATESLISTCHANGED_SENSORSTATE, arg);
        if (arg != NULL)
        {
          sensorState = arg->value.U8;

          ARSAL_PRINT(ARSAL_PRINT_INFO, TAG, "sensorName %d ; sensorState: %d", sensorName, sensorState);
        }
        else
        {
          ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "arg sensorState is NULL");
        }
      }
    }
    else
    {
      ARSAL_PRINT(ARSAL_PRINT_ERROR, TAG, "elements is NULL");
    }
  }
  if ((commandKey == ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGEVENT_MOVEBYEND) && (elementDictionary != NULL))
  {
    ARCONTROLLER_DICTIONARY_ARG_t *arg = NULL;
    ARCONTROLLER_DICTIONARY_ELEMENT_t *element = NULL;
    HASH_FIND_STR (elementDictionary, ARCONTROLLER_DICTIONARY_SINGLE_KEY, element);
    if (element != NULL)
    {
      float dX, dY, dZ, dPsi;
      HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGEVENT_MOVEBYEND_DX, arg);
      if (arg != NULL)
      {
        dX = arg->value.Float;
      }
      HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGEVENT_MOVEBYEND_DY, arg);
      if (arg != NULL)
      {
        dY = arg->value.Float;
      }
      HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGEVENT_MOVEBYEND_DZ, arg);
      if (arg != NULL)
      {
        dZ = arg->value.Float;
      }
      HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGEVENT_MOVEBYEND_DPSI, arg);
      if (arg != NULL)
      {
        dPsi = arg->value.Float;
      }
      dXSum+=dX;
      dYSum+=dY;
      dZSum+=dZ;
      HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGEVENT_MOVEBYEND_ERROR, arg);
      if (arg != NULL)
      {
        IHM_PrintVelocity(ihm, dXSum,dYSum,dZSum,false,false, counter1++);
        moveCommands(deviceController);
        //eARCOMMANDS_ARDRONE3_PILOTINGEVENT_MOVEBYEND_ERROR error = arg->value.I32;
      }
    }
  }
  if ((commandKey == ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGSTATE_FLYINGSTATECHANGED) && (elementDictionary != NULL))
  {

    ARCONTROLLER_DICTIONARY_ARG_t *arg = NULL;
    ARCONTROLLER_DICTIONARY_ELEMENT_t *element = NULL;
    HASH_FIND_STR (elementDictionary, ARCONTROLLER_DICTIONARY_SINGLE_KEY, element);
    if (element != NULL)
    {
      HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGSTATE_FLYINGSTATECHANGED_STATE, arg);
      if (arg != NULL)
      {
        eARCOMMANDS_ARDRONE3_PILOTINGSTATE_FLYINGSTATECHANGED_STATE state = arg->value.I32;
        if(state == ARCOMMANDS_ARDRONE3_PILOTINGSTATE_FLYINGSTATECHANGED_STATE_HOVERING){
          //moveCommands(deviceController);
          //moveCommands(deviceController);
        }
        //IHM_PrintInfo(ihm, "The state changed to "+ state);
        if(state == ARCOMMANDS_ARDRONE3_PILOTINGSTATE_FLYINGSTATECHANGED_STATE_TAKINGOFF || state == ARCOMMANDS_ARDRONE3_PILOTINGSTATE_FLYINGSTATECHANGED_STATE_USERTAKEOFF){
          wasTakingOff = true;
        }
        else if(wasTakingOff){
          //This is a bullshit command because the api is broken

          IHM_PrintInfoF(ihm, "In the taking off thing: %d", counter2++);

          deviceController->aRDrone3->sendPilotingMoveBy(deviceController->aRDrone3, 0, 0, 0,0);
          moveCommands(deviceController);

          wasTakingOff=false;
        }
      }
    }
  }
  float epsilon = .05f;
  /*if ((commandKey == ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGSTATE_SPEEDCHANGED) && (elementDictionary != NULL))
  {
  ARCONTROLLER_DICTIONARY_ARG_t *arg = NULL;
  ARCONTROLLER_DICTIONARY_ELEMENT_t *element = NULL;
  HASH_FIND_STR (elementDictionary, ARCONTROLLER_DICTIONARY_SINGLE_KEY, element);
  if (element != NULL)
  {
  bool stopped = true;
  float speed[3] = {0.0f};
  HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGSTATE_SPEEDCHANGED_SPEEDX, arg);
  if (arg != NULL)
  {
  speed[0] = arg->value.Float;
}
else{
return;
}
HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGSTATE_SPEEDCHANGED_SPEEDY, arg);
if (arg != NULL)
{
speed[1] = arg->value.Float;
}
HASH_FIND_STR (element->arguments, ARCONTROLLER_DICTIONARY_KEY_ARDRONE3_PILOTINGSTATE_SPEEDCHANGED_SPEEDZ, arg);
if (arg != NULL)
{
speed[2] = arg->value.Float;
}

for(int i = 0;i<3;i++){
bufferSpeed[bufferPosition*3+i] = speed[i];
}

bufferPosition = (bufferPosition+1)%SPEED_BUFFER_SIZE;

for(int i = 0;i<3;i++){
accumulatedSpeed[i]-= bufferSpeed[bufferPosition*3+i]/SPEED_BUFFER_SIZE;
accumulatedSpeed[i]+= speed[i]/SPEED_BUFFER_SIZE;
stopped&=fabsf(accumulatedSpeed[i])<epsilon;
}

//IHM_PrintVelocity(ihm, accumulatedSpeed[0], accumulatedSpeed[1], accumulatedSpeed[2],stopped, wasMoving, counter2++);
if (stopped && wasMoving && !stopCommand)
{
moveCommands(deviceController);
wasMoving = false;
}
if(!stopped && !stopCommand){
wasMoving = true;
}
}
}*/
}

void batteryStateChanged (uint8_t percent)
{
  // callback of changing of battery level

  if (ihm != NULL)
  {
    IHM_PrintBattery (ihm, percent);
  }

}

eARCONTROLLER_ERROR decoderConfigCallback (ARCONTROLLER_Stream_Codec_t codec, void *customData)
{
  if (videoOut != NULL)
  {
    if (codec.type == ARCONTROLLER_STREAM_CODEC_TYPE_H264)
    {
      if (DISPLAY_WITH_MPLAYER)
      {
        writeSocket(codec.parameters.h264parameters.spsBuffer, codec.parameters.h264parameters.spsSize);
        writeSocket(codec.parameters.h264parameters.ppsBuffer, codec.parameters.h264parameters.ppsSize);
      }
    }

  }
  else
  {
    ARSAL_PRINT(ARSAL_PRINT_WARNING, TAG, "videoOut is NULL.");
  }

  return ARCONTROLLER_OK;
}


eARCONTROLLER_ERROR didReceiveFrameCallback (ARCONTROLLER_Frame_t *frame, void *customData)
{
  if (videoOut != NULL)
  {
    if (frame != NULL)
    {
      if (DISPLAY_WITH_MPLAYER)
      {
        writeSocket(frame->data, frame->used);
      }
    }
    else
    {
      ARSAL_PRINT(ARSAL_PRINT_WARNING, TAG, "frame is NULL.");
    }
  }
  else
  {
    ARSAL_PRINT(ARSAL_PRINT_WARNING, TAG, "videoOut is NULL.");
  }

  return ARCONTROLLER_OK;
}


// IHM callbacks:

void onInputEvent (eIHM_INPUT_EVENT event, void *customData)
{
  // Manage IHM input events
  ARCONTROLLER_Device_t *deviceController = (ARCONTROLLER_Device_t *)customData;
  eARCONTROLLER_ERROR error = ARCONTROLLER_OK;

  switch (event)
  {
    case IHM_INPUT_EVENT_EXIT:
    stopCommand = true;
    IHM_PrintInfo(ihm, "IHM_INPUT_EVENT_EXIT ...");
    gIHMRun = 0;
    break;
    case IHM_INPUT_EVENT_EMERGENCY:
    if(deviceController != NULL)
    {
      stopCommand = true;
      // send a Emergency command to the drone
      error = deviceController->aRDrone3->sendPilotingEmergency(deviceController->aRDrone3);
    }
    break;
    case IHM_INPUT_EVENT_LAND:
    if(deviceController != NULL)
    {
      stopCommand = true;
      // send a landing command to the drone
      error = deviceController->aRDrone3->sendPilotingLanding(deviceController->aRDrone3);
    }
    break;
    case IHM_INPUT_EVENT_TAKEOFF:
    if(deviceController != NULL)
    {
      stopCommand = false;
      // send a takeoff command to the drone
      error = deviceController->aRDrone3->sendPilotingTakeOff(deviceController->aRDrone3);
    }
    break;
    case IHM_INPUT_EVENT_UP:
    if(deviceController != NULL)
    {
      // set the flag and speed value of the piloting command
      error = deviceController->aRDrone3->setPilotingPCMDGaz(deviceController->aRDrone3, 50);
    }
    break;
    case IHM_INPUT_EVENT_DOWN:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDGaz(deviceController->aRDrone3, -50);
    }
    break;
    case IHM_INPUT_EVENT_RIGHT:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDYaw(deviceController->aRDrone3, 50);
    }
    break;
    case IHM_INPUT_EVENT_LEFT:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDYaw(deviceController->aRDrone3, -50);
    }
    break;
    case IHM_INPUT_EVENT_FORWARD:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDPitch(deviceController->aRDrone3, 50);
      error = deviceController->aRDrone3->setPilotingPCMDFlag(deviceController->aRDrone3, 1);
    }
    break;
    case IHM_INPUT_EVENT_BACK:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDPitch(deviceController->aRDrone3, -50);
      error = deviceController->aRDrone3->setPilotingPCMDFlag(deviceController->aRDrone3, 1);
    }
    break;
    case IHM_INPUT_EVENT_ROLL_LEFT:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDRoll(deviceController->aRDrone3, -70);
      error = deviceController->aRDrone3->setPilotingPCMDFlag(deviceController->aRDrone3, 1);
    }
    break;
    case IHM_INPUT_EVENT_ROLL_RIGHT:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMDRoll(deviceController->aRDrone3, 70);
      error = deviceController->aRDrone3->setPilotingPCMDFlag(deviceController->aRDrone3, 1);
    }
    break;
    case IHM_INPUT_EVENT_NONE:
    if(deviceController != NULL)
    {
      error = deviceController->aRDrone3->setPilotingPCMD(deviceController->aRDrone3, 0, 0, 0, 0, 0, 0);
    }
    break;
    default:
    break;
  }

  // This should be improved, here it just displays that one error occured
  if (error != ARCONTROLLER_OK)
  {
    IHM_PrintInfo(ihm, "Error sending an event");
  }
}
int pos = 0;
float commands[NUM_COMMANDS][4] = {{1.0f,0.0f,0.0f,0.0f},{-1.0f,0.0f,0.0f,0.0f}};//,{0.0f,1.0f,0.0f,0.0f},{0.0f,0.0f,0.0f,1.0f}};,{2.0f,0.0f,0.0f,1.5708f},{2.0f,0.0f,0.0f,1.5708f}};//{{0.0f,1.0f,0.0f,1.5708f},{1.0f,0.0f,0.0f,1.5708f},{0.0f,-1.0f,-0.0f,1.5708f},{-1.0f,0.0f,0.0f,1.5708f}};
void moveCommands(ARCONTROLLER_Device_t *deviceController)
{

  //IHM_PrintInfoF(ihm, "In moveCommands %d", pos + 1);
  if(!stopCommand){
    deviceController->aRDrone3->sendPilotingMoveBy(deviceController->aRDrone3, commands[pos%NUM_COMMANDS][0], commands[pos%NUM_COMMANDS][1], commands[pos%NUM_COMMANDS][2],commands[pos%NUM_COMMANDS][3]);
    pos++;
  }
}
int customPrintCallback (eARSAL_PRINT_LEVEL level, const char *tag, const char *format, va_list va)
{
  // Custom callback used when ncurses is runing for not disturb the IHM

  if ((level == ARSAL_PRINT_ERROR) && (strcmp(TAG, tag) == 0))
  {
    // Save the last Error
    vsnprintf(gErrorStr, (ERROR_STR_LENGTH - 1), format, va);
    gErrorStr[ERROR_STR_LENGTH - 1] = '\0';
  }

  return 1;
}
