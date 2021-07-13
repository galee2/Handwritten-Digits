import csv
import math
import numpy as np

# helper functions for question outputs
def z(w, x, b):
    return np.dot(w, x) + b
def logisitc_activation(z_val):
    if z_val < 0: # prevent overflow error for math.exp()
        return 1 - (1 / (1 + math.exp(z_val)))
    else: 
        return 1 / (1 + math.exp(-z_val))

def q1(feature_vector):
    q1 = ''
    for num in feature_vector: # store feature vector to a string
        q1 = q1 + str(round(num,2)) + ', '
    q1 = q1[:-2] # remove last ', ' from string
    with open('P1\P1_Q1.txt', 'w') as filehandle: # write feature vector to a text file
        filehandle.writelines("%s\n" % q1)
    filehandle.close()
def q2(w, b):
    q2 = ''
    for weight in w: # store weights vector to a string
        q2 = q2 + str(round(weight,4)) + ', '
    q2 = q2 + str(round(b,4)) # add bias to end of string
    with open('P1\P1_Q2.txt', 'w') as filehandle: # write weights and bias to a text file
        filehandle.writelines("%s\n" % q2)
    filehandle.close()
def q3(a):
    q3 = ''
    for a_val in a: # store test image's activation function values to a string
        q3 = q3 + str(round(a_val,2)) + ', '
    q3 = q3[:-2] # remove last ', ' from string
    with open('P1\P1_Q3.txt', 'w') as filehandle: # write activation values to a text file
        filehandle.writelines("%s\n" % q3)
    filehandle.close()
def q4(a):
    q4 = ''
    for a_val in a: # compute test image's predicted label and store to a string
        if a_val < 0.5: q4 = q4 + '0, '
        else: q4 = q4 + '1, '
    q4 = q4[:-2] # remove last ', ' from string
    with open('P1\P1_Q4.txt', 'w') as filehandle: # write predicted labels to a text file
        filehandle.writelines("%s\n" % q4)
    filehandle.close()
def q5(w_1, b_1):
    with open('P1\P1_Q5.txt', 'w') as filehandle: # write weights and bias to a text file
        for i in range(len(w_1)):
            q5_w = ''
            for weight in w_1[i]: # store weights vector for pixel i to a string
                q5_w = q5_w + str(round(weight,4)) + ', '
            q5_w = q5_w[:-2] # remove last ', ' from string
            filehandle.writelines("%s\n" % q5_w) # write line to file
        q5_b = ''
        for bias in b_1: # store bias vector to a string
            q5_b = q5_b + str(round(bias,4)) + ', '
        q5_b = q5_b[:-2] # remove last ', ' from string
        filehandle.writelines("%s\n" % q5_b) # write line to file
    filehandle.close()
def q6(w_2, b_2):
    q6 = ''
    for weight in w_2: # store weights vector to a string
        q6 = q6 + str(round(weight,4)) + ', '
    q6 = q6 + str(round(b_2,4)) # add bias to end of string
    with open('P1\P1_Q6.txt', 'w') as filehandle: # write weights and bias to a text file
        filehandle.writelines("%s\n" % q6)
    filehandle.close()
def q7(a_2):
    q7 = ''
    for a_val in a_2: # store test image's activation function values to a string
        q7 = q7 + str(round(a_val,2)) + ', '
    q7 = q7[:-2] # remove last ', ' from string
    with open('P1\P1_Q7.txt', 'w') as filehandle: # write activation values to a text file
        filehandle.writelines("%s\n" % q7)
    filehandle.close()
def q8(a_2):
    q8 = ''
    for a_val in a_2: # compute test image's predicted label and store to a string
        if a_val < 0.5: q8 = q8 + '0, '
        else: q8 = q8 + '1, '
    q8 = q8[:-2] # remove last ', ' from string
    with open('P1\P1_Q8.txt', 'w') as filehandle: # write predicted labels to a text file
        filehandle.writelines("%s\n" % q8)
    filehandle.close()

feature_matrix = []
labels = []
m = 784 # total number of pixels
n = 0 # number of images in training set

# parse csv file of training data
with open('P1\P1_Training_Set.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader: # iterate over all rows in the csv file
        if row[0] == '3':
            labels.append(0) # if image is a 3, label as 0
            feature_matrix.append(np.divide(list(map(int, row[1:])), 255.0)) # add to feature matrix
            n = n + 1
        elif row[0] == '6':
            labels.append(1) # if image is a 6, label as 1
            feature_matrix.append(np.divide(list(map(int, row[1:])), 255.0)) # add to feature matrix
            n = n + 1
csv_file.close()
feature_matrix = np.array(feature_matrix)
#print(f"feature matrix size = {len(feature_matrix)} x {len(feature_matrix[0])}")
#print(f"labels vector size = {len(labels)} x 1")
#print(n)

# parse csv file of test data
test_data = []
with open('P1\P1_Test_Set.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=",")
    for row in csv_reader: # iterate over all rows in the csv file
        test_data.append(np.divide(list(map(int, row)), 255.0)) # add to feature matrix
csv_file.close()
test_data = np.array(test_data)

################################################# PART 1 #################################################

# Question 1
q1(feature_matrix[0]) # call helper function q1()

## train logistic regression
w = np.random.uniform(low = -1.0, high = 1.0, size = m)
b = np.random.uniform(low = -1.0, high = 1.0)

alpha = 0.1
c_old = 10000
c_new = 10
iterations = 0

while abs(c_old - c_new) > 0.0001 and c_new > 1 and iterations < 1000:
    c_old = c_new
    c_new = 0
    iterations = iterations + 1

    # compute activation function and cost for each image
    a = [0]*n
    for i in range(n): 
        # compute activation function
        z_val = z(w, feature_matrix[i], b)
        a[i] = logisitc_activation(z_val)

        # compute cost
        if labels[i] == 0: # y=0
            diff = 1 - a[i]
            if diff <= 0.0001: # a~1
                c_new = c_new + 10000
            else: c_new = c_new - math.log(diff)
        elif labels[i] == 1: # y=1
            if a[i] <= 0.0001: # a~0
                c_new = c_new + 10000
            else: c_new = c_new - math.log(a[i])
        else:
            c_new = c_new - (labels[i] * math.log(a[i]) + (1 - labels[i]) * math.log(1 - a[i]))


    # update weights and bias based on activation function values
    a_y = np.subtract(a, labels)
    for j in range(m):
        w[j] = w[j] - alpha * np.dot(a_y, feature_matrix[:, j])
    b = b - alpha * sum(a_y)

    alpha = alpha / math.sqrt(iterations) # update alpha value

#TODO: DELETE
print(c_new)
print(c_old)

# Question 2
q2(w, b) # call helper function q2()

# Question 3
q3_a = [0]*200
for i in range(200): # compute activation function for each test image
    z_val = z(w, test_data[i], b)
    q3_a[i] = logisitc_activation(z_val)
q3(q3_a) # call helper function q3()

# Question 4
q4(q3_a) # call helper function q4()

################################################# PART 2 #################################################
h = 28
w_1 = np.random.uniform(low = -1.0, high = 1.0, size = (m, h))
b_1 = np.random.uniform(low = -1.0, high = 1.0, size = h)
w_2 = np.random.uniform(low = -1.0, high = 1.0, size = h)
b_2 = np.random.uniform(low = -1.0, high = 1.0)

alpha = 0.001
c_old = 10000
c_new = 10
iterations = 0

while iterations < 15000 or abs(c_old - c_new) > 0.0001:
    c_old = c_new
    c_new = 0
    iterations = iterations + 1
    print(iterations)

    image_list = [x for x in range(n)] # list of training image indexes
    np.random.shuffle(image_list) # shuffle images for stochastic gradient descent
    i = image_list[0] # index of random image

    # find activation values for hidden layer
    a_1 = [0]*h
    for j in range(h):
        z_val = z(w_1[:, j], feature_matrix[i], b_1[j])
        a_1[j] = logisitc_activation(z_val)

    # find output activation value
    a_2 = 0
    z_val = z(w_2, a_1, b_2)
    a_2 = logisitc_activation(z_val)

    # update weights and biases for each layer based on activation values
    dC_db_2 = (a_2 - labels[i]) * a_2 * (1 - a_2)
    b_2 = b_2 - (alpha * dC_db_2)
    for j in range(h):
        dC_db_1 = (a_2 - labels[i]) * a_2 * (1 - a_2) * w_2[j] * a_1[j] * (1 - a_1[j])
        dC_dw_1 = np.multiply(feature_matrix[i], dC_db_1)
        w_1[:, j] = np.subtract(w_1[:, j], np.multiply(dC_dw_1, alpha))
        b_1[j] = b_1[j] - (alpha * dC_db_1)

        dC_dw_2 = dC_db_2 * a_1[j]
        w_2[j] = w_2[j] - (alpha * dC_dw_2)

    c_new = 0.5 * math.pow((labels[i] - a_2), 2) # find cost
    alpha = alpha / math.sqrt(iterations) # update alpha value

print(c_old)
print(c_new)

# Question 5
q5(w_1, b_1)

# Question 6
q6(w_2, b_2)

# Question 7
q7_a_1 = np.zeros((200, h))
q7_a_2 = [0]*200
for i in range(200): 
    for j in range(h): # compute first layer activation function for each test image
        z_val = z(test_data[i], w_1[:, j], b_1[j])
        q7_a_1[i][j] = logisitc_activation(z_val)
    # compute second layer activation function for each test image
    z_val = z( q7_a_1[i], w_2, b_2)
    q7_a_2[i] = logisitc_activation(z_val)
q7(q7_a_2)

# Question 8
q8(q7_a_2)
