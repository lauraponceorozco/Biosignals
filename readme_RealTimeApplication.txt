Notes about the folder Application_Real_Time

General:
- The robot control uses the NeuraPy API provided by NEURA and  was developed together witn NeuroTUM. 
- Main author: Christian Ritter, co-authors: Esther Utasá, Phillip Wagner
- To connect to the robot, set up the Ethernet connection according to the NeuraPy Documentation.
- Do not operate the robot without prior safety instructions.
- The grid positions are precalibrated and saved in the folder 'data/'

Folders and Files:
- The implementations are located in the folder 'Robot/'
- application_drawing.py: 
    - Run this file to draw circles on the whiteboard using keyboard input. 
    - When starting the scipt, you will be asked to enter a digit from 1 to 9 corresponding to the desired grid
    - After entering the digit, the robot will draw a circle in that grid
- application_BCI.py:
    - Run this file to draw a circle based on real-time EEG recordings
    - Start this scipt first. It will wait for confirmation that the EEG recording is finished (key press)
    - Run the OpenVibe paradigm ('p300-speller-1-acquisition-RealTime-1trial.xml') that only performs one trail with a specified number of repetitions (currently: 12)
    - This paradim will save the EEG stream as CSV in 'cur_rec/rec.csv'. Old recordings will be overwritten
    - After key press, script will classify the grid label based on the recorded EEG data and draw a circle in the classified grid
