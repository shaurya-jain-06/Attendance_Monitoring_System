
# Attendance Monitoring System

## Installing Packages
Run the following command to install all the required packages:
```bash
$ pip install -r requirements.txt
```

## Project Structure
```
|-- strongsort
|-- weights
|-- InceptionResnet.py
|-- facenet_keras_weights.h5
|-- main.py
|-- multi_tracker_zoo.py
|-- train.py
|-- yolov8n-face.pt
|-- Faces
    |-- Person1
        |-- image1
        |-- image2
        .
        .
    |-- Person2
    |-- Person3...
```

## About Files and Folders
- **strongsort, weights**: Files related to the tracker.
- **multi_tracker_zoo.py**: The tracker code used to initiate multiple tracker instances.
- **yolov8n-face.pt**: File needed for the face detector.
- **InceptionResnet.py**: Contains the architecture of the Inception Resnet network.
- **facenet_keras_weights.h5**: Contains the weights for the Inception Resnet network.
- **train.py**: Used to train the model on the faces of the people you want to recognize.
- **main.py**: The main project file that runs the system, comprising a face detection module, tracking module, recognition module, and database module. The face detection and tracking modules run on their own as the required weights and files are already mentioned and used. The recognition module provides predictions based on the pictures provided in the Faces folder. The database module updates the database on its own, creating a database under the name 'attendance' and creating a new table for each month, giving new entries for each tracker frame created through a MySQL connection. The MySQL connection must be made on localhost port with username 'sqluser' and password 'password'. However, these can be changed by modifying the credentials in the code.
- **Faces**: This is where you need to put the images for training of each person as shown in the file organization. The pictures of each person must be put in their own separate folder with the folder name being their ID or name. Preferably use at least 25-30 pictures of each person for training and include around 10 pictures of each person's face from an elevated angle.

## Running the Project
**Requires Python Version 3.10**

To train the system on your photos after the pictures have been organized as mentioned before:
```bash
$ python3 train.py
```

To run the project using the laptop camera as the default video source:
```bash
$ python3 main.py
```

To run the project on a video file:
```bash
$ python main.py --source video_path
```

To save the results:
```bash
$ python main.py --source video_path --save-vid
```
