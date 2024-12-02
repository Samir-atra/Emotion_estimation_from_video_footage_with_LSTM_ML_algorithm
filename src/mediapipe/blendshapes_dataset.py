# creating blendshapes dataset        (styled)

set = []
images = []
labels = []
full_set = []
indices = []

with open("validation_set_full.csv", mode= "r") as data:            # load the dataset that will be processed
  csvFile = csv.reader(data)
  next(csvFile)
  for lines in csvFile:
    set.append(lines)
  for i in range(len(set)):    
    image = np.array(set[i][1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    images.append(image)                               # create a list of the images
    labels.append(int(set[i][0]))                      # create a list of the labels
    # indices.append(int(set[i][2]))
    
    
arr = np.zeros((len(images), 36))                    # create an array with the size of dataset and number of blendshapes to be used 

# arr[0,:]= blendS_to_print
for ele in range(len(images)-1):
  img = images[ele]
  label = labels[ele]
  # img_index = indices[ele]
  frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=img)
  detection_result = detector.detect(frame)
  cat_counter = 0
  for category in detection_result.face_blendshapes[0]:
    if str(category.index) in blends_to_print:                # if the blendshape from the detector is in the blends_to_pront list then include it in the array
      arr[ele, cat_counter] = category.score
      cat_counter += 1
    else:
       continue
  arr[ele, 34] = label
  # arr[ele, 35] = img_index

     
# fields = ["emotion", "pixels", "Index"]
csv_writer("blends_val_all_emotion.csv",'beedoo', arr)             # create a file from the array


