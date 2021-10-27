from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import time
import pickle
import numpy as np
import cv2
import os
import csv

""" -- CHANGELOG --
    v2 changes:
        - increased n_components from 25 to 250 of PCA
        - removed random_state from PCA
        - adjusted possible parameters for svc
    v3 changes:
        - reduced n_components from 250 to 150
        - adjusted possible parameters for svc
        - removing interpolation step from image resizing
    v4 changes:  
        - adjusted possible parameters for svc
    v5 changes:
        - adjusted possible parameters for svc
    v6 changes:
        - large overhaul to the svc parameters
        - perform an EXPANSIVE search for the best parameter
"""


# Function to load images and filenames from a directory
def load_image_data(folder, flag):
    data = []
    names = []
    size = 0

    # Read every file from the folder directory
    for filename in os.listdir(folder):

        # Read in the image, convert it to grayscale, and resize it to 150x150
        # Also, save the current filename
        names.append(filename)
        img = cv2.imread(os.path.join(folder, filename))
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized_img = cv2.resize(grey_img, (150, 150))

        # Flatten the array of 150x150 pixels into a row vector
        img_array = np.array(resized_img)
        img_data = img_array.flatten()
        data.append(img_data)
        size = size + 1

    # Store every row vector of pixels into a matrix
    data_array = np.zeros((size, 22500))
    for i in range(size):
        data_array[i] = data[i]

    if flag:
        return data_array, names
    else:
        return data_array


# Function to load all of the training data into a single matrix
def load_training_data():

    # Load training images into separate arrays
    building_data = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/train/buildings/', False)
    forest_data = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/train/forest/', False)
    glacier_data = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/train/glacier/', False)
    mountain_data = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/train/mountain/', False)
    sea_data = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/train/sea/', False)
    street_data = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/train/street/', False)

    # Combine the arrays into matrices
    stack1 = np.vstack((building_data, forest_data))
    stack2 = np.vstack((glacier_data, mountain_data))
    stack3 = np.vstack((sea_data, street_data))
    tempStack = np.vstack((stack1, stack2))

    labels = []
    # Create the training labels for the data
    for i in range(len(building_data)):
        labels.append(0)
    for j in range(len(forest_data)):
        labels.append(1)
    for k in range(len(glacier_data)):
        labels.append(2)
    for x in range(len(mountain_data)):
        labels.append(3)
    for y in range(len(sea_data)):
        labels.append(4)
    for z in range(len(street_data)):
        labels.append(5)

    return np.vstack((tempStack, stack3)), np.array(labels)


# Function to write results to a csv file
def write_to_csv(prediction_data, file):
    csv_file = open("SVM_results_v6.csv", "w", newline='')
    writer = csv.writer(csv_file)
    writer.writerow(["Id", "Category"])
    for i in range(len(prediction_data)):
        writer.writerow([file[i], prediction_data[i]])
    csv_file.close()
    print("Successfully written to SVM_results_v6.csv")


# Main
if __name__ == '__main__':
    print("Current time:", (time.ctime(time.time())), flush=True)

    # Load the training data and labels
    start_time = time.perf_counter()
    training_data, training_labels = load_training_data()
    load_time = time.perf_counter()
    print("Time to load training data:", (load_time - start_time), flush=True)

    # Image preprocessing
    pca = PCA(n_components=150, whiten=True)
    svc = SVC(kernel='rbf', class_weight='balanced')
    model = make_pipeline(pca, svc)
    preproc_time = time.perf_counter()
    print("Time to perform preprocessing:", (preproc_time - load_time), flush=True)

    # Create an extensive list of C and gamma parameters for svc
    svc_C = [0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000]
    svc_gamma = [0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 1, 10, 100, 1000]

    # Perform grid search CV to find the optimal parameter
    param_grid = {'svc__C': svc_C, 'svc__gamma': svc_gamma}
    grid = GridSearchCV(model, param_grid)
    param_time = time.perf_counter()
    print("Time to find the optimal parameters:", (param_time - preproc_time), flush=True)

    # Fit the data
    grid.fit(training_data, training_labels)
    model = grid.best_estimator_
    fit_time = time.perf_counter()
    print(grid.best_params_)
    print("Time to fit the data:", (fit_time - param_time), flush=True)

    # Save the model as a pickle file
    pickle_file = open('SVM_model_v6.sav', 'wb')
    pickle.dump(model, pickle_file)
    pickle_file.close()
    pickle_time = time.perf_counter()
    print("Time to pickle the model:", (pickle_time - fit_time), flush=True)
    # model = pickle.load(open("SVM_model_v6.sav", "rb"))

    # Create the predictive model and write the contents to a csv file
    test_images, filenames = load_image_data('C:/Users/Marcelo/PycharmProjects/CMPE188/test/', True)
    SVM_model = model.predict(test_images)
    predict_time = time.perf_counter()
    print("Time to predict the test data:", (predict_time - pickle_time), flush=True)

    write_to_csv(SVM_model, filenames)
    csv_time = time.perf_counter()
    print("Time to write data to a csv:", (csv_time - pickle_time), flush=True)
