# visualize misclassified images from the test set

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
