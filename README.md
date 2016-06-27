# multisensor-concept-event-detection

Contains the implementation of experiments for a video concept and event detection framework developed for the MULTISENSOR project. The framework relies on DCNN features and Support Vector Machines (SVM) classification algorithm. A three-fold cross validation (CV) is executed to evaluate performance. The code has been developed and tested in Python, version 3.5.1, 64-bit.

# Description

In this code, a dataset that contains 106 videos from news reports is utilized. Videos are categorized into nine concepts/events. Note that one video may be relevant to zero or more of these concepts/events. The dataset is available at: http://mklab-services.iti.gr/multisensor/data/Event_Detection_Dataset_MS.rar . Key frames are already extracted from the video shots because training is made on images. Total number of key frames on this dataset is 2826. DCNN features are extracted from the key frames based on the Caffe models trained in the work of (Markatopoulou et al., 2016).  Using a random balanced split on the dataset for each concept/event, where the videos are divided into three chunks, a three-fold CV is performed using two chunks for training purposes and the remaining chunk for testing. The classification algorithm used in this code is SVMs, where it’s “c” parameter is tuned using grid search. Output of this module is the evaluation per concept/event on videos in terms of accuracy and F-score.

# Input arguments

This code is used for the aforementioned dataset but it can be used for other datasets provided that all image (key frame) names are number identifiers starting from 1 and ending to the total number of key frames.  The paths for the following files must be provided as arguments:

 - Feature vector file. File must contain one vector per line with a space separating each value of the vector. Order must be ascending in terms of the image file name.
 - Key frame annotation file. This file contains the annotation per concept/event of all key frames. Concepts/events are identified in terms of these IDs: [001, 002, 003, 004, 005, 006, 007, 008, 009]. Each line of the file has the following format (order of image names does not matter here): <br />
[concept_id] 1 [image_name_without_file_extension] [annotation_value (-1 or 1)] 1 <br />
Example lines: <br />
001 1 101 1 1 <br />
001 1 102 -1 1 <br />
002 1 101 -1 1 <br />
002 1 102 -1 1
 - Video annotation file. This file contains the annotation per concept of all videos. Concepts/events are identified in terms of the same ids as in the key frame annotation file. Each line of the file has the following format: <br />
[concept_id] 1 [video_name_without_file_extension] [annotation_value (-1 or 1)] 1 
 - Video-to-image mapping file. It shows which images correspond to which video. It is considered that key frames extracted from a video are assigned to a range of image IDs (e.g. 11-17) Each line of the file has the following format: <br />
[video_name_without_file_extension] [first_frame_image_id]-[last_frame_image_id] <br />
Example: <br />
video name: dg20130420_cottbus_sd_avc.mp4 <br />
key frames extracted from the video: 10.jpg, 11.jpg, 12.jpg, 13.jpg, 14.jpg <br />
line in the mapping file: dg20130420_cottbus_sd_avc 10-14


# References

F. Markatopoulou, V. Mezaris, I. Patras, "Online Multi-Task Learning for Semantic Concept Detection in Video", Proc. IEEE Int. Conf. on Image Processing (ICIP 2016), Phoenix, AZ, USA, Sept. 2016.

# Version
1.0.0
