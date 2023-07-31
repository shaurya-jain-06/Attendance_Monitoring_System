...............................Attendance Monitoring system..........................................................

\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

.......Installing packages:

Run the following command to install all the required packages.

`	`$ pip install -r requirements.txt

.......Project Structure:

`	`|--strongsort

`	`|--weights

`	`|--InceptionResnet.py

`	`|--facenet\_keras\_weights.h5

`	`|--main.py

`	`|--multi\_tracker\_zoo.py

`	`|--train.py

`	`|--yolov8n-face.pt

|--Faces

|--Person1

|-- image 1

|-- image 2

.

.

|--Person2

|--Person3...

\-----------------------------------------------------------------------------------------------

.......About files and folders

strongsort,weights--files related to the tracker

multi\_tracker\_zoo.py--is the tracker code which is used to initiate multiple tracker instances

yolov8n-face.pt--file needed for the face detector

InceptionResnet.py--is the file that has the architecture of the Inception Resnet network

facenet\_keras\_weights.h5--is the file that contain the weights for the Inception resnet network

train.py--is the file used to train the model on the faces of the people you want to

recognize

main.py--is the main project file that runs the system, it comprises of a face detection module,

tracking module, recognition module and database module. The face detection and tracking

modules run on their own as the required weights and files are already mentioned and used.

The recognition module will provide predictions based on the pictures provided in the Faces

folder as mentioned below. The database module updates the database on its own. It creates

a database under the name 'attendance' and creates a new table for each month giving new

entries for each tracker frame created through a mysql connection. The mysql connection must

be made on localhost port with username 'sqluser' and password 'password'. However, these can

be changed as per the user by modifying the credentials in the code too.

Faces--is the file where you need to put the images for training of each person as shown in

the file organisation. The pictures of each person must be put in their own seperate folder

with the folder name being their id or name. Preferably use atleast 25-30 pictures of each person

for training and include around 10 pictures of each person's face from an elevated angle.

\------------------------------------------------------------------------------------------------------------------

.........Running the project

REQUIRES PYTHON VERSION 3.10

To run the project

First train the system on your photos using the following command after the pictures have

been organised as mentioned before

$ python3 train.py

To run the project after training use the following commannd



`	`$ python3 main.py //This makes the project to use the laptop camera as default for

video source

To run the project on a video file that you have use the following command

$ python main.py --source video\_path

To save the results

$ python main.py --source video\_path --save-vid




