from sklearn import svm, cross_validation
from sklearn.grid_search import GridSearchCV
import numpy as np
from collections import OrderedDict
import warnings
import sys
import time


def main(arguments):
    start = time.time()
    print()
    print("CONCEPT AND EVENT DETECTION MODULE START")
    print()

    features_file = shot_annotation_file = video_annotation_file = video_mapping_file = ""
    features = labels = images_per_video = video_labels = ""
    warnings.filterwarnings("ignore")

    if len(arguments) == 1:
        print("Default file configurations will be used.")
        print("Features file: shot_features.txt")
        print("Shot annotations file: shot_annotation_all.txt")
        print("Video annotations file: video_annotation_all.txt")
        print("Video-to-shot mapping file: mapping.txt")
    elif len(arguments) == 5:
        features_file = arguments[1]
        shot_annotation_file = arguments[2]
        video_annotation_file = arguments[3]
        video_mapping_file = arguments[4]
        print("Features file:", features_file)
        print("Shot annotations file:", shot_annotation_file)
        print("Video annotations file:", video_annotation_file)
        print("Video-to-shot mapping file:", video_mapping_file)
    else:
        print("Arguments given:", len(arguments))
        print("Arguments must have this format:")
        print("<path_to_features_file> <path_to_shot_annotation_file> <path_to_video_annotation_file>"
              " <path_to_video_mapping_file>")
        print()
        print("Exiting module...")
        exit()
    print()

    concept_ids = ['001', '002', '003', '004', '005', '006', '007', '008', '009']
    mean_accuracies = dict()
    mean_f_scores = dict()
    positives = dict()
    for concept_id in concept_ids:
        fold = 1
        print("CONCEPT ID:", concept_id)
        print("------------")

        if len(arguments) == 1:
            features, labels, images_per_video, video_labels = read_input(concept_id)
        elif len(arguments) == 5:
            features, labels, images_per_video, video_labels = read_input(concept_id, features_file,
                                                                          shot_annotation_file, video_annotation_file,
                                                                          video_mapping_file)

        accuracies = list()
        f_scores = list()

        # stratified split
        video_labels = OrderedDict(sorted(video_labels.items()))
        video_names_list = list()
        video_labels_list = list()
        for key, value in video_labels.items():
            video_names_list.append(key)
            video_labels_list.append(value)

        positive = video_labels_list.count(1)
        positives[concept_id] = positive
        video_names_list = np.array(video_names_list)
        video_labels_list = np.array(video_labels_list)

        sss = cross_validation.StratifiedKFold(video_labels_list, n_folds=3)
        for train_index, test_index in sss:
            video_names_train, video_names_test = video_names_list[train_index], video_names_list[test_index]

            print("FOLD", fold, "...")
            fold += 1
            accuracy, precision, recall, f_score = train_evaluate(train_set=video_names_train,
                                                                  test_set=video_names_test,
                                                                  features=features, labels=labels,
                                                                  images_per_video=images_per_video)
            print("Accuracy:", accuracy)
            print("Precision:", precision)
            print("Recall:", recall)
            print("F-score:", f_score)
            print()
            accuracies.append(accuracy)
            f_scores.append(f_score)

        mean_accuracy = np.mean(accuracies)
        mean_f_score = np.mean(f_scores)
        mean_accuracies[concept_id] = mean_accuracy
        mean_f_scores[concept_id] = mean_f_score
        print("3-fold cross validation f-score:", mean_f_score)
        print()

    print("RESULTS")
    print("-------")
    for concept_id in concept_ids:
        print("ID:", concept_id, "\tAccuracy:", mean_accuracies[concept_id],
              "\tF-score:", mean_f_scores[concept_id])

    sum_f_scores = 0
    for k, v in mean_f_scores.items():
        sum_f_scores += v
    avg_f_score = sum_f_scores / len(mean_f_scores)
    print("Macro average f-score:", avg_f_score)

    # print elapsed time
    end = time.time()
    elapsed = end - start
    print()
    print("Elapsed time (seconds):", elapsed)


def read_input(concept_id, features_file="shot_features.txt", shot_annotation_file="shot_annotation_all.txt",
               video_annotation_file="video_annotation_all.txt", video_mapping_file="mapping.txt"):
    """
    Read features, shot/video annotation and video-to-image mapping from external files.
    """

    # read features file
    features = dict()
    feature_file = open(features_file, "r")
    start = 1
    for feature_vector in feature_file:
        if feature_vector.strip() != '':
            features[start] = feature_vector.strip()
            start += 1
    feature_file.close()

    # read shot annotations
    shot_labels = dict()
    shot_labels_file = open(shot_annotation_file, "r")
    for shot_label_line in shot_labels_file:
        if shot_label_line.strip() != '':
            splitted = shot_label_line.strip().split()
            if splitted[0] == concept_id:
                shot_id = int(splitted[2])
                shot_label = int(splitted[3])
                shot_labels[shot_id] = shot_label
    shot_labels_file.close()

    # read video annotations
    video_labels = dict()
    video_labels_file = open(video_annotation_file, "r")
    for video_label_line in video_labels_file:
        if video_label_line.strip() != '':
            splitted = video_label_line.strip().split()
            if splitted[0] == concept_id:
                video_name = splitted[2]
                video_label = int(splitted[3])
                video_labels[video_name] = video_label
    video_labels_file.close()

    # map images to videos
    images_per_video = dict()
    images_per_video_file = open(video_mapping_file, "r")
    for line in images_per_video_file:
        split = line.strip().split()
        video_name = split[0]
        second_split = split[1].split("-")
        start = int(second_split[0])
        end = int(second_split[1])
        images_per_video[video_name] = list()
        for i in range(start, end + 1):
            images_per_video[video_name].append(i)
    images_per_video_file.close()

    return features, shot_labels, images_per_video, video_labels


def train_evaluate(train_set, test_set, features, labels, images_per_video):
    """
    Train classification models on training shots and evaluate them on testing videos
    """

    X_train = list()
    Y_train = list()

    # form train set
    for video in train_set:
        video_images = images_per_video[video]
        for image in video_images:
            feature_string = features[image]
            features_list = [float(i) for i in feature_string.split()]
            label = labels[image]
            X_train.append(features_list)
            Y_train.append(label)

    # train classifier
    param_grid = {'C': [10 ** i for i in range(-8, 9)]}
    clf = GridSearchCV(svm.SVC(class_weight='balanced'), param_grid=param_grid, scoring='f1')
    clf.fit(X_train, Y_train)
    print("The best parameters are %s with a score of %0.2f"
          % (clf.best_params_, clf.best_score_))

    test_video_labels = dict()
    test_video_predicted_labels = dict()

    # form test set and predict
    for video in test_set:
        video_images = images_per_video[video]
        test_video_labels[video] = -1
        test_video_predicted_labels[video] = -1
        for image in video_images:
            feature_string = features[image]
            features_list = [float(i) for i in feature_string.split()]
            label = labels[image]
            # if at least one shot has label value '1', then the whole video has label value '1'
            if label == 1:
                test_video_labels[video] = 1
            prediction = clf.predict([features_list])
            if prediction[0] == 1:
                test_video_predicted_labels[video] = 1

    accuracy, precision, recall, f_score = calculate_evaluation_metrics(test_video_labels, test_video_predicted_labels)
    return accuracy, precision, recall, f_score


def calculate_evaluation_metrics(gt_labels, predicted_labels):
    """
    Evaluate predictions using accuracy, precision, recall and f-score
    Input: ground truth labels and predicted labels
    """

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for key in gt_labels.keys():
        test_video_label = gt_labels[key]
        test_video_predicted_label = predicted_labels[key]
        if test_video_label == 1 and test_video_predicted_label == 1:
            TP += 1
        elif test_video_label == -1 and test_video_predicted_label == -1:
            TN += 1
        elif test_video_label == -1 and test_video_predicted_label == 1:
            FP += 1
        elif test_video_label == 1 and test_video_predicted_label == -1:
            FN += 1

    accuracy = (TP + TN)/(TP + TN + FP + FN)
    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP/(TP + FP)
    if (TP + FN) == 0:
        recall = 0
    else:
        recall = TP/(TP + FN)
    if (precision + recall) == 0:
        f_score = 0
    else:
        f_score = (2 * precision * recall)/(precision + recall)
    return accuracy, precision, recall, f_score

if __name__ == '__main__':
    main(sys.argv)
