"""visualize misclassified images from the test set"""

test_blend_set = []
with open("test_set_full_index.csv", mode= "r") as test_data:
  csvFile = csv.reader(test_data)
  next(csvFile)
  for lines in csvFile:
      test_blend_set.append(lines[0:52])
      # imageee = np.array(str(lines[1]).split(' ')).reshape(48, 48, 1).astype(np.uint8)
      # imageee = cv2.cvtColor(imageee, cv2.COLOR_GRAY2RGB)
      # plt.imshow(imageee)
      # plt.show()
for ind in errors:
      print(ind)
      imageee = np.array(str(test_blend_set[int(float(ind))][1]).split(' ')).reshape(48, 48, 1).astype(np.uint8)
      imageees = cv2.cvtColor(imageee, cv2.COLOR_GRAY2RGB)
      frame = mp.Image(image_format=mp.ImageFormat.SRGB,data=imageees)
      detection_result = detector.detect(frame)
      annotated_image = draw_landmarks_on_image(frame.numpy_view(), detection_result)
      plt.imshow(annotated_image)
      plt.show()
      plt.imshow(imageees)
      plt.show()


# create classes lists for it to be produced as datasets
for ins in traindata:
  if float(ins[52]) == 0:
    class1.append(ins)
  elif float(ins[52]) == 1:
    class2.append(ins)
  elif float(ins[52]) == 2:
    class3.append(ins)
    
csv_writer("class1.csv",'beedoo', class1)
csv_writer("class2.csv",'beedoo', class2)
csv_writer("class3.csv",'beedoo', class3)

# visualizations for the features from the classes and 
plt.figure(figsize = [40,40])
y = np.linspace(0,1999,1999)
for j in range(1,52):
  plt.subplot(10,6,j)                                         # create a subplot with the number of features
  class1_slice = [float(i[j]) for i in blends_set[1:2000]]
# class2_slice = [float(i[25]) for i in class2[1:200]]
  plt.title(f"{j}")
  plt.scatter(y,class1_slice,color='blue')
plt.scatter(y,class2_slice, color='red')
print(class1[1:100])
corr_mat = np.corrcoef(class1_slice, class2_slice)            # find the correlation matrix
print(corr_mat)