# dataset analysis/calculations

categories_counts = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0}
skipped = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0}


with open("fer2013.csv", mode= "r") as data:                    # open the dataset file
  csvFile = csv.reader(data)
  next(csvFile)
  for lines in csvFile:                                         # loop throught the training instances
    if lines[0] in categories_c:
      categories_counts[lines[0]] = categories_counts[lines[0]] + 1
    image = np.array(str(lines[1]).split(' ')).reshape(48, 48, 1).astype(np.uint8)     # build the image from a list of numbers
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=image)
    detection_result = detector.detect(rgb_frame)                                       # detect the face in the image
    # check if the image get recognized or skipped by mediapipe
    if detection_result.face_blendshapes == []:                                         # check if mediapipe is able to detect a face in the image
        skipped[lines[0]] = skipped[lines[0]] + 1
    else:
        # img = plt.imshow(image)
        # plt.show()
        continue
print(categories_counts)
print(skipped)



# # Processing, cleaning and visualizing the fer2013 dataset  

fullset = []
Training_set = []
validation_set = []
test_set = []
class_counter = {"0":0, "1":0, "2":0, "3":0, "4":0, "5":0, "6":0}


with open("training_set_full.csv", mode= "r") as data:
  csvFile = csv.reader(data)
  next(csvFile)
  # iterate over the dataset and append the images to a list, for limited numbers according to it's classes, to creat the training set
  for lines in csvFile:
      if lines[0] == "0" and class_counter['0'] < 1500:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '1'
        fullset.append(lines)
      elif lines[0] == "1" and class_counter['1'] < 1500:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '1'
        fullset.append(lines)
      elif lines[0] == '2' and class_counter['2'] < 1500:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '1'
        fullset.append(lines) 
      elif lines[0] == '3' and class_counter['3'] < 4000:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '0'
        fullset.append(lines) 
      elif lines[0] == '4' and class_counter['4'] < 4000:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '2'
        fullset.append(lines) 
      elif lines[0] == '5' and class_counter['5'] < 1500:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '1'
        fullset.append(lines) 
      elif lines[0] == '6' and class_counter['6'] < 1500:
        class_counter[lines[0]] = class_counter[lines[0]] + 1
        lines[0] = '1'
        fullset.append(lines) 

print(class_counter)
# plot the required image with annotations
plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])
annotated_image = draw_landmarks_on_image(rgb_frame.numpy_view(), detection_result)
cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))



# Augmenting the training set images


augmented_training_set = []
training_images = []
training_labels = []

# create an image list and a labels list for the training dataset
for i in range(math.floor(len(training_set_hus))):
    image = np.array(training_set_hus[i][1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    training_images.append(image)
    training_labels.append(int(training_set_hus[i][0]))


# rescaling and augmenting images models
rescaling1 = tf.keras.Sequential([ 
  tf.keras.layers.Rescaling(1./255)                         # scale down the images pixel values
])

rescaling2 = tf.keras.Sequential([                          # scale up the pixel values
  tf.keras.layers.Rescaling(1.*255)
])

augment = tf.keras.Sequential([                             # augment by random flipping and random rotation
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.1)
])

for ele in range(len(training_images)):
# scale down the image and augment
  img = training_images[ele]
  label = training_labels[ele]
  image = rescaling1(img)
  aug_image = augment(image)
#   scale up the image cast to an integer and transform into a numpy array for it to be understood by mediapipe
  aug_image = rescaling2(aug_image)
  aug_image = tf.cast(aug_image, tf.uint8)
  aug_image = np.array(aug_image)
  flatten_image = aug_image.flatten()                                            # flatten the augmented image, to be used in creating the csv file
  flat_aug_image = [flatten_image[i] for i in range(0,len(flatten_image),3)]
  # flattt = np.reshape(flat_aug_image,(48,48))
  # plt.imshow(flattt)
  # plt.show()
  frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=aug_image)
  detection_result = detector.detect(frame)
  if detection_result.face_blendshapes == []:
    continue
  else:
    element = [training_labels[ele]]
    for i in flat_aug_image:
      element.append(i)
    augmented_training_set.append([element[0],str(element[1:]).replace(',',"").replace('[','').replace(']',''),'Training'])

for images in augmented_training_set:
  image = np.array(images[1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
  image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  plt.imshow(image)
  plt.show()

csv_writer("training_set_full.csv", ['emotion','pixels'], augmented_training_set)





# data normalization

mean = tf.math.reduce_mean(X_train,axis=0)            # find the mean for the dataset
stddev = tf.math.reduce_std(X_train, axis=0)          # find the standard deviation
mean = np.array(mean).T
stddev = np.array(stddev).T
csv_writer("mean_and_std.csv",'beedoo', mean)
csv_writer("mean_and_std.csv",'beedoo', stddev)

# normalize data
norm = tf.keras.layers.Normalization(axis=1)
norm.adapt(X_train)
print(X_train[0])
XX_train = norm(X_train)
print(XX_train[0])
XX_train = np.array(X_train)
XX_train = X_train[1]
plt.scatter(X_train[1],XX_train)

norm.adapt(X_train)
X_val = norm(X_val)
X_val = np.array(X_val)
