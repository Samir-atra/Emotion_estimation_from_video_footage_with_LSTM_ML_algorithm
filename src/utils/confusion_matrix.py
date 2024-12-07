"""
find the confusion matrix for the model trained and evaluated on the 
test set
"""
import numpy as np
import tensorflow as tf

errors_count = {"0":0, "1":0, "2":0}
ground_truth = []
errors = []
for i in test_labels_set:
    i = np.argmax(i)
    ground_truth.append(i)
for j in range(len(test_blend_set)):
    if predictions[j] != ground_truth[j]:
        errors.append(test_index_set[j])             # creat a list of the indices of the misclassified images in the testset
cm=tf.math.confusion_matrix(ground_truth,predictions,num_classes=3,dtype=tf.dtypes.int32,)        # calculate the confusion matrix

for i in range(1646):
    if predictions[i]==ground_truth[i]:
        continue
    elif predictions[i] != ground_truth[i]:
        errors_count[str(ground_truth[i])] = errors_count[str(ground_truth[i])] + 1

