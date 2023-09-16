from math import sqrt

def split_data(data_set):
    labels = []
    for rows in data_set:
        # removing the last element(label) from data set and appending that into labels.
        labels.append(rows.pop(-1)) 
        # returning modified data_set (without labels) and labels separately.
    return data_set, labels

def measure_distance(data_set, test_data, distance_metric):
    
    # We are calling the split_data() function and passing the data_set
    # which will return labels and modified data_set.
    sdata = split_data(data_set)
    
    # train_data will contain the modified data set
    train_data = sdata[0]
    
    # labels will contain the respective label of each record/row/patient
    labels = sdata[1]
    
    # temp array will contain the individual distance of each row and then
    # it will be summed up and added into the distances array for every row
    temp = []
    
    # distances array contain the summed up distance for every patient/row/record
    distances = []
    
    # Euclidean Distance
    if distance_metric == 'euclidean':
        # Loop for every patient in the data
        for i in range(len(train_data)):
            # Loop for every feature in a row/record/patient
            for j in range(7):
                # Applying euclidean distance formula
                distance = sqrt((train_data[i][j] - test_data[j]) ** 2)
                # Appending the distance in temp array
                temp.append(distance)
                # As all the features of one row/record/patient are stored in temp
                # we will jump in this condition
                if j == 6:
                    # here we are summing up all the distances of one row/record/patient
                    # and adding them in the distances array
                    distances.append(sum(temp))
                    # emptying the temp array for next row/record/patient
                    temp = []

    # Manhatten Distance
    # Same as Euclidean distance except the formula
    elif distance_metric == 'manhatten':
        for i in range(len(train_data)):
            for j in range(7):
                distance = abs((train_data[i][j] - test_data[j]))
                temp.append(distance)
                if j == 6:
                    distances.append(sum(temp))
                    temp = []
                    
    # result array for distances that we have calculated along with their labels
    result = []
    # Loop for giving each distance its respective label
    for i in range(0, len(distances)):
        row = [distances[i], labels[i]]
        result.append(row)
    return result

# Funciton definition in which we have passed the data_set of patients
# which is raw at the moment, i.e. labels are present in the data_set.
# test_data which is to be classified whether or not it has diabetes.
# distance_metric to specify Euclidean or Manhatten distance.
# K is the number of neighbours.
def knn_classifier(data_set, test_data, distance_metric, k):
    
    # Here we are calling the measure_distance function, and sorting it in 
    # ascending order for checking the k neighbours
    result = sorted(measure_distance(data_set, test_data, distance_metric))
    
    # count for counting the max number of labels in k neighbours
    count = 0
    
    # Loop till k elements in the result array
    for i in range(k):
        if result[i][1] == 0:
            count += 1
    if count > k // 2:
        return "You do not have diabetes"
    return "You have diabetes"


# I have added this dataset manually which isn't a good approach but I wanted
# to do it without any library. But it is not applicable to every situation so 
# if you want to use pandas library for data set you can use this code
# import pandas as pd
# data = pd.read_csv(diabetes_dataset.csv)
# and the rest of the code is same.

data = [[6, 148, 72, 35, 0, 33.6, 0.627, 50, 1],
           [1, 85, 66, 29, 0, 26.6, 0.351, 31, 0],
           [8, 183, 64, 0, 0, 23.3, 0.672, 32, 1],
           [1, 89, 66, 23, 94, 28.1, 0.167, 21, 0],
           [0, 137, 40, 35, 168, 43.1, 2.288, 33, 1],
           [5, 116, 74, 0, 0, 25.6, 0.201, 30, 0],
           [3, 78, 50, 32, 88, 31, 0.248, 26, 1],
           [10, 115, 0, 0, 0, 35.3, 0.134, 29, 0],
           [2, 197, 70, 45, 543, 30.5, 0.158, 53, 1],
           [8, 125, 96, 0, 0, 0, 0.232, 54, 1],
           [4, 110, 92, 0, 0, 37.6, 0.191, 30, 0],
           [10, 168, 74, 0, 0, 38, 0.537, 34, 1],
           [10, 139, 80, 0, 0, 27.1, 1.441, 57, 0],
           [1, 189, 60, 23, 846, 30.1, 0.398, 59, 1],
           [5, 166, 72, 19, 175, 25.8, 0.587, 51, 1],
           [7, 100, 0, 0, 0, 30, 0.484, 32, 1],
           [0, 118, 84, 47, 230, 45.8, 0.551, 31, 1],
           [7, 107, 74, 0, 0, 29.6, 0.254, 31, 1],
           [1, 103, 30, 38, 83, 43.3, 0.183, 33, 0],
           [1, 115, 70, 30, 96, 34.6, 0.529, 32, 1],
           [3, 126, 88, 41, 235, 39.3, 0.704, 27, 0],
           [8, 99, 84, 0, 0, 35.4, 0.388, 50, 0],
           [7, 196, 90, 0, 0, 39.8, 0.451, 41, 1],
           [9, 119, 80, 35, 0, 29, 0.263, 29, 1],
           [11, 143, 94, 33, 146, 36.6, 0.254, 51, 1],
           [10, 125, 70, 26, 115, 31.1, 0.205, 41, 1],
           [7, 147, 76, 0, 0, 39.4, 0.257, 43, 1],
           [1, 97, 66, 15, 140, 23.2, 0.487, 22, 0],
           [13, 145, 82, 19, 110, 22.2, 0.245, 57, 0],
           [5, 117, 92, 0, 0, 34.1, 0.337, 38, 0],
           [5, 109, 75, 26, 0, 36, 0.546, 60, 0],
           [3, 158, 76, 36, 245, 31.6, 0.851, 28, 1],
           [3, 88, 58, 11, 54, 24.8, 0.267, 22, 0],
           [6, 92, 92, 0, 0, 19.9, 0.188, 28, 0],
           [10, 122, 78, 31, 0, 27.6, 0.512, 45, 0],
           [4, 103, 60, 33, 192, 24, 0.966, 33, 0],
           [11, 138, 76, 0, 0, 33.2, 0.42, 35, 0],
           [9, 102, 76, 37, 0, 32.9, 0.665, 46, 1],
           [2, 90, 68, 42, 0, 38.2, 0.503, 27, 1],
           [4, 111, 72, 47, 207, 37.1, 1.39, 56, 1],
           [3, 180, 64, 25, 70, 34, 0.271, 26, 0],
           [7, 133, 84, 0, 0, 40.2, 0.696, 37, 0],
           [7, 106, 92, 18, 0, 22.7, 0.235, 48, 0],
           [9, 171, 110, 24, 240, 45.4, 0.721, 54, 1],
           [7, 159, 64, 0, 0, 27.4, 0.294, 40, 0],
           [0, 180, 66, 39, 0, 42, 1.893, 25, 1],
           [1, 146, 56, 0, 0, 29.7, 0.564, 29, 0],
           [2, 71, 70, 27, 0, 28, 0.586, 22, 0],
           [7, 103, 66, 32, 0, 39.1, 0.344, 31, 1]]

# test data that I have copied fromt the data set for checking
test = [10, 161, 68, 23, 132, 25.5, 0.326, 47]

# calling the function with appropriate values
print(knn_classifier(data, test, 'euclidean', 7))
