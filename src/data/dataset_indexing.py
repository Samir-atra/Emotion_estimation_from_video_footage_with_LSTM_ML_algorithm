# indexing the testset

test_set_hus = []

with open("test_set_full.csv", mode= "r") as data:
  csvFile = csv.reader(data)
  next(csvFile)
  for lines in csvFile:
      imageee = np.array(lines[1].split(' ')).reshape(48, 48, 1).astype(np.uint8)
      imageee = cv2.cvtColor(imageee, cv2.COLOR_GRAY2RGB)
    #   plt.imshow(image)
    #   plt.show()
      frameee = mp.Image(image_format=mp.ImageFormat.SRGB,data=imageee)
      detection_result = detector.detect(frameee)
      if detection_result.face_blendshapes == []:
        continue
      else:
        test_set_hus.append(lines)
        
nums = np.array([i for i in range(0,1648)])               # create a list of numbers to be appended as indices
nums = np.reshape(nums, (1648,1))
test_set_hus = np.array(test_set_hus)
test_set_hus= np.delete(test_set_hus,2,1)                 # delete the third column which contains the name of the split
test_set_hus = np.hstack((test_set_hus, nums))            # add the indices column
fields = ["emotion", "pixels", "Index"]
csv_writer("test_set_full_index.csv", fields, test_set_hus)          # creat an indexed dataset
