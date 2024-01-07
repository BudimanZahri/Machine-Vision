import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from skimage.feature import hog
from skimage import exposure
from mlxtend.data import loadlocal_mnist

images, labels = loadlocal_mnist(
    images_path='images/mnist-dataset/train-images-idx3-ubyte',
    labels_path='images/mnist-dataset/train-labels-idx1-ubyte'
)
# print(images,labels,'aaa')


num_images_to_process = 10000
hog_features = []
for i in range(num_images_to_process):
    image = images[i]
    fd, hog_image = hog(image.reshape((28, 28)), orientations=8, pixels_per_cell=(4, 4),
                        cells_per_block=(1, 1), visualize=True)
    hog_features.append(fd)

hog_features = np.array(hog_features)

X_train, X_test, y_train, y_test = train_test_split(hog_features, labels[:num_images_to_process], test_size=0.2, random_state=42)

svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Confusion Matrix:\n", conf_matrix)