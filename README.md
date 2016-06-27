# multisensor-concept-event-detection

Contains the implementation of experiments for a video concept and event detection framework developed for the MULTISENSOR  project. The framework relies on DCNN features and SVM classifier. A three-fold cross validation (CV) is executed to evaluate performance. The code has been developed and tested in Python, version 3.5.1, 64-bit.

# Description


# Input arguments

This code is used for the aforementioned dataset but it can be used for other datasets provided that all image (key frames) names are number identifiers starting from 1 and ending to the total number of key frames.  The paths for the following files must be provided as arguments:

 - Feature vector file. File must contain one vector per line with a space separating each value of the vector. Order must be ascending in terms of the image file name.
 - Key frame annotation file. This file contains the annotation per concept/event of all key frames. Concepts/events are identified in terms of these IDs: [001, 002, 003, 004, 005, 006, 007, 008, 009]. Each line of the file has the following format (order of image names does not matter here):
<concept_id> 1 <image_name_without_file_extension> <annotation_value (-1 or 1)> 1 
Example lines: 
001 1 101 1 1
001 1 102 -1 1
002 1 101 -1 1
002 1 102 -1 1
 - Video annotation file. This file contains the annotation per concept of all videos. Concepts/events are identified in terms of the same ids as in the key frame annotation file. Each line of the file has the following format:
<concept_id> 1 <video_name_without_file_extension> <annotation_value (-1 or 1)> 1 
 - Video-to-image mapping. This file shows which images correspond to which video. It is considered that a video key frames consist of range image IDs (e.g. 11-17) Each line of the file has the following format: 
<video_name_without_file_extension> <first_frame_image_id>-<last_frame_image_id>
Example:
video name: dg20130420_cottbus_sd_avc.mp4
key frames extracted from the video: 10.jpg, 11.jpg, 12.jpg, 13.jpg, 14.jpg
line in the mapping file: dg20130420_cottbus_sd_avc 10-14


# References

F. Markatopoulou, V. Mezaris, I. Patras, "Online Multi-Task Learning for Semantic Concept Detection in Video", Proc. IEEE Int. Conf. on Image Processing (ICIP 2016), Phoenix, AZ, USA, Sept. 2016.

# Version
1.0.0
