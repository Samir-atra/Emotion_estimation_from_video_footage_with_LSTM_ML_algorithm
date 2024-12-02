# download and upload images from and into colab

# download
img = cv2.imread("images.jpg")
cv2_imshow(img)

# upload images to colab
from google.colab import files
uploaded = files.upload()

for filename in uploaded:
  content = uploaded[filename]
  with open(filename, 'wb') as f:
    f.write(content)

if len(uploaded.keys()):
  IMAGE_FILE = next(iter(uploaded))
  print('Uploaded file:', IMAGE_FILE)
  
  
  # load data for LSTM model         ()

traindata = []
blend_set = []
labels_set = []
class1=[]
class2=[]
class3=[]
with open("blends_train_full_set.csv", mode= "r") as data:
  csvFile = csv.reader(data)
  next(csvFile)
  for line in csvFile:
    traindata.append(line[:])
  np.random.shuffle(traindata)
  for lines in traindata:
      blend_set.append(lines[0:52])
      labels_set.append(lines[52])
blends_set = np.array(blend_set, dtype=np.float64)        # create an array of the blendshapes
labels_set = np.array(labels_set, dtype=np.float64)       # create an array of the labels

#Reshaping Array
X_train = np.reshape(blends_set, (22515, 52,1))
Y_train = np.reshape(labels_set, (22515,1)).astype('int')
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes=3)       # encode the labels as one-hot

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

# load validation data
valdata = []
val_blend_set = []
val_labels_set = []
with open("blends_val_all_emotion.csv", mode= "r") as val_data:
  csvFile = csv.reader(val_data)
  next(csvFile)
  for line in csvFile:
    valdata.append(line[:])
  np.random.shuffle(valdata)
  for lines in valdata:
      val_blend_set.append(lines[0:34])
      val_labels_set.append(lines[34])
val_blend_set = np.array(val_blend_set, dtype=np.float64)
val_labels_set = np.array(val_labels_set, dtype=np.float64)
X_val = np.reshape(val_blend_set, (1657, 34,1))
y_val = np.reshape(val_labels_set, (1657,1)).astype('int')
y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)


  
  
  