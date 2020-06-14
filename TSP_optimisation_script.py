import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from timeit import time
import operator

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn import metrics

gifts = pd.read_csv('gifts.csv')
"""""
First we call csv_creation to create the new excel files that we are gonna work with
We basically order the function to create as many files as we want.
Input type is a list that contains the sizes of the new files we want to create
"""



"""
In case you want to enter a file that is not created by this code run this function with your file:
That needs to be done cause when creating the files with csv_creation we re index to 0-size
We later use this column when creating an array to calculate distances so it is important that the file contains this.
"""


def csv_modification(your_file):
    new_file = pd.DataFrame(pd.read_csv(your_file))
    new_file.index = range(new_file.shape[0])
    new_file.to_csv('your_new_file.csv')


# Function creating the 3 csv files
# We have as index range(N) , so that when we convert it in np array, our first column is gonna be this range
def csv_creation(lengths):
    files = []
    for length in lengths:
        filename = 'gifts' + str(length) + '.csv'
        files.append(filename)
        new = gifts.sample(length)
        new.index = range(length)
        new.to_csv(filename)
    return files


# Function that calculates haversine distance between 2 points
def haversine(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(np.radians(lat1)) \
        * np.cos(np.radians(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    d = radius * c
    return d


"""
All Functions that hava data as parameter receive csv files , and the ones that have dataset receive arrays.
"""

""""
Function that creates an array with distances between gifts
The last column is the distance between each gift and North Pole
for example distances[3,2] is the distance between gifts 2 and 1
"""


def distances_cr(data):
    north_pole = (90.0, 0.0)
    gifts1 = np.array(pd.read_csv(data))
    distances1 = np.zeros([gifts1.shape[0], gifts1.shape[0] + 1])
    for i in range(distances1.shape[0] - 1):
        distances1[i, -1] = haversine(gifts1[i, 2:4], north_pole)
        for j in range(i + 1, distances1.shape[0]):
            distances1[i, j] = haversine(gifts1[i, 2:4], gifts1[j, 2:4])
            distances1[j, i] = distances1[i, j]
    distances1[-1, -1] = haversine(gifts1[-1, 2:4], north_pole)

    return distances1


""""
Function that creates the initial trip IDs
We use as maximum weight per trip the heaviest gift weight * 2 (N of gifts)
We want 2 gifts in order to complete all the neighbourhood movements
We don't check for overweight trips, this is done later by the objective_f function.
The trip creation is the most important factor for getting a good weariness.
ALSO IMPORTANT: In every algorithm we add 2 columns in our dataset. One for the trip ID , and one for the index of the gifts.
"""


def trip_cr(dataset):
    trips = np.zeros([dataset.shape[0], 1])
    x = dataset
    trips[0, 0] = 1
    trip_sum = 1
    max_weight = np.max(x[:, 4]) * 2
    sum = x[0, 4]
    i = 1
    while i < x.shape[0]:
        if x[i, 4] + sum <= max_weight:
            trips[i, 0] = trip_sum
            sum += x[i, 4]
            i += 1
        else:
            trip_sum += 1
            trips[i, 0] = trip_sum
            sum = x[i, 4]
            i += 1

    if trips[-1, 0] != trips[-2, 0]:
        trips[-2, 0] = trips[-1, 0]
    if trips[-1, 0] < 3:
        k = int(x.shape[0] / 3)
        l = 2 * int(x.shape[0] / 3)
        for i in range(k):
            trips[i, 0] = 1
        for i in range(k, l):
            trips[i, 0] = 2
        for i in range(l, x.shape[0]):
            trips[i, 0] = 3

    return trips


"""
Random search heuristic algorithm
We take one random solution, then shuffle the dataset 1000*N times and return the best result
"""


def random_search(data):
    x = np.array(pd.read_csv(data))
    distances = distances_cr(data)
    np.random.shuffle(x)  # Initialize 1st random solution by shuffling
    best_route = trip_cr2(x)
    x = np.c_[x, best_route]
    x = np.c_[x, range(x.shape[0])]
    best = objective_f(x, distances)[0]  # 1st solution
    solutions = 10 * x.shape[0]
    i = 0
    while i < solutions:
        x1 = np.copy(x)
        np.random.shuffle(x1[:, :5])
        temp, x1 = objective_f(x1, distances)
        if temp < best:  # Shuffle until 5th column, so we shuffle the trip characteristics
            best = temp
            best_route = x1[:, 5]
            x = np.copy(x1)

        i += 1
    print(end-start) # Print times
    return best, best_route
    # Return best weariness and trip sequence(we want to check the trips)


"""
Runs function we need with 30 different seeds and creates a list with the optimal weariness
"""


def r_seed(data, function, a=0.95):
    best_30 = []
    start= time.time()
    if function != random_search:
        for i in range(1, 31):
            random.seed(i)
            best_30.append(function(data, a)[0])
    else:
        for i in range(1, 31):
            random.seed(i)
            best_30.append(function(data)[0])
    end= time.time()
    time1= end-start
    print('Time for 30 seeds of algorithm: '+str(function)+ 'is '+ time1 )
    return best_30


"""
Receives 1 trip ID array, and returns its weariness
Column with index 0 in our dataset contains the gift number (0- N-1)
This way we can call distances function which returns haversine between gifts
For example distances (5,4) is the distance between gift 4 and gift 3
"""


def weight_array(dataset, distances):
    sleigh = 10
    x = dataset
    weight = np.sum(x[:, 4]) + sleigh
    d = distances[int(x[0, 0]), -1] * weight

    if x.shape[0] > 1:
        for i in range(1, x.shape[0]):
            weight -= x[i - 1, 4]
            d += distances[int(x[i - 1, 0]), int(x[i, 0])] * weight
    d += distances[int(x[-1, 0]), -1] * sleigh

    return d


# Objective Function, returns total wariness for all trips
def objective_f(dataset, distances):
    result = 0
    x = dataset

    for i in np.unique(x[:, 5]):
        # We calculate every trip seperately
        temp = x[x[:, 5] == i]
        # We check the weights, and if a trip is overweight we correct it by passing its last gift on to the next trip
        # We continue while the trip is overweight
        # In our case this never happens, but the control should be included
        while np.sum(temp[:, 4]) > 1000:
            temp[-1, 5] = i + 1
            x[int(temp[-1, 6]), 5] = i + 1
            temp = np.delete(temp, -1, axis=0)

        result = result + weight_array(temp, distances)

    return result, x


"""
Metaheuristic algorithm with 1 neighborhood movement
First we take one random solution, and then instead of shuffling we make 1 neighborhood movement
Movement is swapping 2 gifts from different trips
In every algorithm we add 2 columns in our dataset. One for the trip ID , and one for the index of the gifts.
The big difference between random search and SA, is that in SA we only calculate the modified trips weariness
As a result SA is much faster!
"""


def simulated_annealing(data, schedule=0.95):  # default a =0.95, but user can use any a he wants
    x = np.array(pd.read_csv(data))
    distances = distances_cr(data)
    np.random.shuffle(x)
    best_route = trip_cr(x)
    x = np.c_[x, best_route]
    x = np.c_[x, range(x.shape[0])]
    best = objective_f(x, distances)[0]
    top_best = best  # SA can accept some worst weariness values, so we need to store the best weariness as well
    not_accepted = 0
    solutions = 1000 * x.shape[0]
    i = 0
    T = 1
    a = schedule
    while i < solutions and not_accepted < not_accepted:
        # We end the loop after solutions*0.05 iterations go by without accepting a new solution
        x1 = np.copy(x)
        k = np.random.randint(x.shape[0], size=1)  # We pick one random gift
        other = x[x[:, 5] != x[k, 5]]
        l = int(np.random.choice(other[:, 6]))  # Now we pick one gift with different trip id
        t1 = x[k, 5]
        t2 = x[l, 5]
        # We swap the 2 gifts, and also their tripID and their index(the 2 columns we have inserted)
        x1[k, :], x1[l, :] = x1[l, :], x1[k, :]
        x1[k, 5], x1[l, 5] = x1[l, 5], x1[k, 5]
        x1[k, 6], x1[l, 6] = x1[l, 6], x1[k, 6]
        trips3 = x[x[:, 5] == t1]
        trips4 = x[x[:, 5] == t2]
        wrw1 = weight_array(trips3, distances) + weight_array(trips4, distances)
        # We only calculate the weariness of the modified trips, and then add their difference in the previous weariness
        trip1 = x1[x1[:, 5] == t1]
        trip2 = x1[x1[:, 5] == t2]
        # We check if the new trips are overweight, and if they are we send the hole dataset in the objective function
        if np.sum(trip1[:, 4]) > 1000 or np.sum(trip2[:, 4]) > 1000:
            temp, x1 = objective_f(x1, distances)
        else:
            wrw2 = weight_array(trip1, distances) + weight_array(trip2, distances)
            temp = best + (wrw2 - wrw1)

        value = (best - temp) / T  # The value that goes in the SA check
        if temp <= top_best:  # Top best is the best weariness and not the last we accepted
            top_best = temp
            x = x1
            best_route = x[:, 5]
            best = temp
            not_accepted = 0
        # Here we can see that we can accept a worst weariness, if 2nd part of if is TRUE
        elif temp < best or random.random() < np.exp(value):
            x = x1
            best = temp
            best_route = x[:, 5]
            not_accepted = 0
        else:  # Number of consecutive iterations without accepting a solution
            not_accepted += 1
        if T > 0.5:
            T = a * T
        i += 1
    start = time.time()
    # We return the best weariness + the trip IDS, not the last weariness we accepted
    return top_best, best_route


# Creates csv file with the results table
def results(dataset, function):
    table = pd.DataFrame(index=['Dataset 1', 'Dataset 2', 'Dataset 3'], columns=['Minimum', 'Maximum', 'Mean', 'Std'])
    for i in range(len(dataset)):
        table.iloc[i, :] = np.min(dataset[i]), np.max(dataset[i]), np.mean(dataset[i]), np.std(dataset[i])
    table.to_csv('Results' + function + '.csv')


"""
Simulated Annealing algorithm with 2 neighborhood movements
After choosing this algorithm for extra testing with different schedules, we modified it so that a is now a parameter
"""


def SA_double_movement(data, schedule=0.95):  # default a =0.95, but user can use any a he wants
    x = np.array(pd.read_csv(data))
    distances = distances_cr(data)
    np.random.shuffle(x)
    best_route = trip_cr(x)
    x = np.c_[x, best_route]  # Adding the trip IDS column
    x = np.c_[x, range(x.shape[0])]  # Adding the index column
    best = objective_f(x, distances)[0]
    top_best = best
    not_accepted = 0
    solutions = 1000 * x.shape[0]
    i = 0
    T = 1
    a = schedule
    while i < solutions and not_accepted < x.shape[0] * 0.05:
        # We terminate the loop, after solutions*0.05 iterations go by without accepting a new solution
        # NM4: 1st swap between two gifts , one from trip k and one from trip l

        x1 = np.copy(x)
        k = np.random.randint(x.shape[0], size=1)
        other = x[x[:, 5] != x[k, 5]]
        l = int(np.random.choice(other[:, 6]))
        t1 = x[k, 5]
        t2 = x[l, 5]
        x1[k, :], x1[l, :] = x1[l, :], x1[k, :]
        x1[k, 5], x1[l, 5] = x1[l, 5], x1[k, 5]
        x1[k, 6], x1[l, 6] = x1[l, 6], x1[k, 6]
        trips3 = x[x[:, 5] == t1]
        trips4 = x[x[:, 5] == t2]
        # We create 2 new data sets, x1 has the neighborhood move 1, x2 is the one with NM 2
        wrw1 = weight_array(trips3, distances) + weight_array(trips4, distances)
        trip1 = x1[x1[:, 5] == t1]
        trip2 = x1[x1[:, 5] == t2]
        if np.sum(trip1[:, 4]) > 1000 or np.sum(trip2[:, 4]) > 1000:
            temp1, x1 = objective_f(x1, distances)
        else:
            wrw2 = weight_array(trip1, distances) + weight_array(trip2, distances)
            temp1 = best + (wrw2 - wrw1)
        x2, p1, p2, p3 = three_way_suffix(x)
        trips5 = x[x[:, 5] == p1]
        trips6 = x[x[:, 5] == p2]
        trips7 = x[x[:, 5] == p3]
        wrw3 = weight_array(trips5, distances) + weight_array(trips6, distances) + weight_array(trips7, distances)
        trip3 = x2[x2[:, 5] == p1]
        trip4 = x2[x2[:, 5] == p2]
        trip5 = x2[x2[:, 5] == p3]
        # We check if new trips are overweight, and they aren't we calculate their weariness
        # If they are overweight we send the hole dataset in the objective
        if np.sum(trip3[:, 4]) > 1000 or np.sum(trip4[:, 4]) > 1000 or np.sum(trip5[:, 4]) > 1000:
            temp2, x2 = objective_f(x2, distances)
        else:  # Difference between previous and current modified trips
            wrw4 = weight_array(trip3, distances) + weight_array(trip4, distances) + weight_array(trip5, distances)
            # New weariness is the previous + the difference between new and old modified trips' weariness
            temp2 = best + (wrw4 - wrw3)
        # The dataset after the movement with the best weariness, is the one that we chose to persid with the Annealing
        if temp2 >= temp1:
            best_move = temp1
            new_x = x1
        else:
            best_move = temp2
            new_x = x2
        value = (best - best_move) / T  # Value we will check in the probability check
        if best_move <= top_best:
            top_best = best_move
            best = best_move
            x = new_x
            best_route = x[:, 5]
            not_accepted = 0
        elif best_move < best or random.random() < np.exp(value):
            x = np.copy(new_x)
            best = best_move
            best_route = x[:, 5]
            not_accepted = 0
        else:
            # Number of iterations not accepted
            not_accepted += 1
        if T > 0.05:
            T *= a
        i += 1

    return top_best, best_route

    # Neighborhood move 6, 3-way suffix


"""
2nd Neighborhood move, NM6
We slice 3 trips , and swap the sliced trips' IDS so that:
Trip1 slice ID= Trip3 ID
Trip2 slice ID= Trip1 ID
Trip3 slice ID= Trip2 ID
"""


def three_way_suffix(dataset):
    # 1st we chose 3 random trips
    x = dataset
    # First we select three trips
    trips = np.unique(x[:, 5])
    t1 = random.choice(trips)
    trips = trips[trips != t1]
    t2 = random.choice(trips)
    trips = trips[trips != t2]
    t3 = random.choice(trips)
    t1, t2, t3 = sorted([t1, t2, t3])

    # Now we will slice the trips randomly
    # We can slice all the trips, cause of the trip_cr function creating 2+ gifts trips
    x1 = x[x[:, 5] == t1]
    x2 = x[x[:, 5] == t2]
    x3 = x[x[:, 5] == t3]
    r1 = random.choice(range(1, x1.shape[0]))
    r2 = random.choice(range(1, x2.shape[0]))
    r3 = random.choice(range(1, x3.shape[0]))
    x1 = x1[r1:]
    x2 = x2[r2:]
    x3 = x3[r3:]
    x1[:, 5] = t3
    x2[:, 5] = t1
    x3[:, 5] = t2
    l1 = x2.shape[0] - x1.shape[0]
    l2 = l1 + x3.shape[0] - x2.shape[0]

    # After generating random index to slice, we need to reconstruct the original dataset
    # Column 6 is the index that we added, so we use it to know the exact position we insert and delete

    x = np.delete(x, slice(int(x1[0, 6]), int(x1[-1, 6]) + 1), axis=0)

    x = np.insert(x, int(x1[0, 6]), x2, axis=0)

    x = np.delete(x, slice(int(x2[0, 6]) + l1, int(x2[-1, 6]) + l1 + 1), axis=0)

    x = np.insert(x, int(x2[0, 6]) + l1, x3, axis=0)

    x = np.delete(x, slice(int(x3[0, 6]) + l2, int(x3[-1, 6]) + l2 + 1), axis=0)

    x = np.insert(x, int(x3[0, 6]) + l2, x1, axis=0)

    x[:, 6] = sorted(x[:, 6])  # We sort the index column

    return x, t1, t2, t3  # We return the dataset after the movement + the trips that were modified


# First Visualisation function
def make_plots1(data1, data2, data3):
    figure = plt.figure(facecolor='lavender')
    plt.suptitle('Comparison Plots', fontfamily='Tahoma', fontsize=16, fontweight='bold')

    # We fit all the data sets in one picture
    ax1 = plt.subplot(2, 2, 1)
    data1.boxplot(column=['RS', 'SA', 'SA2'], showmeans=True, medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax1.set_title('Gift10', fontstyle='oblique', fontweight='bold')
    ax1.set_ylabel('WRW', fontfamily='Tahoma', fontweight='bold')

    ax2 = plt.subplot(2, 2, 2)
    data2.boxplot(column=['RS', 'SA', 'SA2'], showmeans=True, medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax2.set_title('Gift100', fontstyle='oblique', fontweight='bold')

    ax3 = plt.subplot(2, 2, 3)
    data3.boxplot(column=['RS', 'SA', 'SA2'], showmeans=True, medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax3.set_title('Gift1000', fontstyle='oblique', fontweight='bold')
    ax3.set_ylabel('WRW', fontfamily='Tahoma', fontweight='bold')

    # Edit layout, get the color, and save in pdf for better quality
    plt.tight_layout(rect=[0, 0.1, 1, 0.85])
    plt.savefig('PlotComparison1.pdf', facecolor=figure.get_facecolor())
    plt.show()

    return plt.show()


# Visualise the varying a and random search
def make_plots2(data1, data2, data3):
    figure = plt.figure(facecolor='lavender')
    plt.suptitle('Comparison Plots', fontfamily='Tahoma', fontsize=16, fontweight='bold')
    # We fit all the data sets in one picture
    ax1 = plt.subplot(2, 2, 1)
    data1.boxplot(column=['RS', '0.5', '0.8', '0.9', '0.95', '0.98'], showmeans=True,
                  medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax1.set_title('Gift10', fontstyle='oblique', fontweight='bold')
    ax1.set_ylabel('WRW', fontfamily='Tahoma', fontweight='bold')

    ax2 = plt.subplot(2, 2, 2)
    data2.boxplot(column=['RS', '0.5', '0.8', '0.9', '0.95', '0.98'], showmeans=True,
                  medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax2.set_title('Gift100', fontstyle='oblique', fontweight='bold')

    ax3 = plt.subplot(2, 2, 3)
    data3.boxplot(column=['RS', '0.5', '0.8', '0.9', '0.95', '0.98'], showmeans=True,
                  medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax3.set_title('Gift1000', fontstyle='oblique', fontweight='bold')
    ax3.set_ylabel('WRW', fontfamily='Tahoma', fontweight='bold')
    # Edit layout, get the color, and save in pdf for better quality
    plt.tight_layout(rect=[0, 0.1, 1, 0.85])
    plt.savefig('PlotComparison2.pdf', facecolor=figure.get_facecolor())
    plt.show()

    return plt.show()


# Visualise only a, in order to have better understanding
def make_plots3(data1, data2, data3):
    figure = plt.figure(facecolor='lavender')
    plt.suptitle('Comparison Plots', fontfamily='Tahoma', fontsize=16, fontweight='bold')
    # We fit all the data sets in one picture
    ax1 = plt.subplot(2, 2, 1)
    data1.boxplot(column=['0.5', '0.8', '0.9', '0.95', '0.98'], showmeans=True,
                  medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax1.set_title('Gift10', fontstyle='oblique', fontweight='bold')
    ax1.set_ylabel('WRW', fontfamily='Tahoma', fontweight='bold')

    ax2 = plt.subplot(2, 2, 2)
    data2.boxplot(column=['0.5', '0.8', '0.9', '0.95', '0.98'], showmeans=True,
                  medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax2.set_title('Gift100', fontstyle='oblique', fontweight='bold')

    ax3 = plt.subplot(2, 2, 3)
    data3.boxplot(column=['0.5', '0.8', '0.9', '0.95', '0.98'], showmeans=True,
                  medianprops={'linestyle': None, 'linewidth': 0},
                  meanprops={"marker": "s", "markerfacecolor": "r", "markeredgecolor": "black"})
    ax3.set_title('Gift1000', fontstyle='oblique', fontweight='bold')
    ax3.set_ylabel('WRW', fontfamily='Tahoma', fontweight='bold')
    # Edit layout, get the color, and save in pdf for better quality
    plt.tight_layout(rect=[0, 0.1, 1, 0.85])
    plt.savefig('PlotComparison3.pdf', facecolor=figure.get_facecolor())
    plt.show()

    return plt.show()


"""
In an attempt to make SA 2ble movement even better, we reversed engineer the whole code
and modified entirely the trip creation function. 
Trip_cr2 is creating Trip IDS 100% randomly, allowing more and smaller trips to be created.
Smaller trips could mean less weariness
Three_way_suffix_2 makes the 2nd movement and SA_double_movement2 is the new function
"""


def weight_array2(dataset, distances):
    sleigh = 10

    x = dataset
    weight = np.sum(x[:, 4]) + sleigh
    if x.shape[0] != 0:
        d = distances[int(x[0, 0]), -1] * weight
        if x.shape[0] > 1:
            for i in range(1, x.shape[0]):
                weight -= x[i - 1, 4]
                d += distances[int(x[i - 1, 0]), int(x[i, 0])] * weight
        d += distances[int(x[-1, 0]), -1] * sleigh
    else:
        d = 0

    return d


def trip_cr2(dataset):  # New function creating trips
    trips = np.zeros([dataset.shape[0], 1])
    x = dataset
    trips[0, 0] = 1
    i = 1
    length = list(range(1, x.shape[0]))
    while i < x.shape[0]:  # We let the algorithm create as many trips as the problem size.
        trips[i, 0] = random.choice(length)
        i += 1
    ID = 1
    for trip in np.unique(trips):  # We sort the trips
        trips[trips[:, 0] == trip, 0] = ID
        ID += 1

    trips = sorted(trips)
    return trips


def three_way_suffix2(dataset):
    x = dataset
    trips = np.unique(x[:, 5])
    t1 = random.choice(trips)
    trips = trips[trips != t1]
    t2 = random.choice(trips)
    trips = trips[trips != t2]
    t3 = random.choice(trips)
    t1, t2, t3 = sorted([t1, t2, t3])
    x1 = x[x[:, 5] == t1]
    x2 = x[x[:, 5] == t2]
    x3 = x[x[:, 5] == t3]

    if x1.shape[0] >= 2:  # We can slice trip 1
        r1 = random.choice(range(1, x1.shape[0]))
        x1 = x1[r1:]
        x1[:, 5] = t3
        x = np.delete(x, slice(int(x1[0, 6]), int(x1[-1, 6]) + 1), axis=0)
        if x2.shape[0] >= 2:  # We can slice trip 2
            r2 = random.choice(range(1, x2.shape[0]))
            x2 = x2[r2:]
            x2[:, 5] = t1
            x = np.insert(x, int(x1[0, 6]), x2, axis=0)
            l1 = x2.shape[0] - x1.shape[0]
            x = np.delete(x, slice(int(x2[0, 6]) + l1, int(x2[-1, 6]) + l1 + 1), axis=0)
            if x3.shape[0] >= 2:  # We can slice trip3
                r3 = random.choice(range(1, x3.shape[0]))
                x3 = x3[r3:]
                x3[:, 5] = t2
                x = np.insert(x, int(x2[0, 6]) + l1, x3, axis=0)
                l2 = l1 + x3.shape[0] - x2.shape[0]
                x = np.delete(x, slice(int(x3[0, 6]) + l2, int(x3[-1, 6]) + l2 + 1), axis=0)
                x = np.insert(x, int(x3[0, 6]) + l2, x1, axis=0)
            else:  # No slicing for trip3
                x3[:, 5] = t2
                x = np.insert(x, int(x2[0, 6]) + l1, x3, axis=0)
                l2 = l1 + x3.shape[0] - x2.shape[0]
                x = np.delete(x, slice(int(x3[0, 6]) + l2, int(x3[-1, 6]) + l2 + 1), axis=0)
                x = np.insert(x, int(x3[0, 6]) + l2, x1, axis=0)
        else:  # We cannot slice trip 2
            if x3.shape[0] >= 2:  # We can slice trip 3
                x2[:, 5] = t1
                x = np.insert(x, int(x1[0, 6]), x2, axis=0)
                l1 = x2.shape[0] - x1.shape[0]
                x = np.delete(x, slice(int(x2[0, 6]) + l1, int(x2[-1, 6]) + l1 + 1), axis=0)
                r3 = random.choice(range(1, x3.shape[0]))
                x3 = x3[r3:]
                x3[:, 5] = t2
                x = np.insert(x, int(x2[0, 6]) + l1, x3, axis=0)
                l2 = l1 + x3.shape[0] - x2.shape[0]
                x = np.delete(x, slice(int(x3[0, 6]) + l2, int(x3[-1, 6]) + l2 + 1), axis=0)
                x = np.insert(x, int(x3[0, 6]) + l2, x1, axis=0)
            else:  # We cannot slice trip 3, so we merge trip 2,3 and recreate a new trip 3 from trip 1
                x = np.delete(x, slice(int(x3[0, 6]) - x1.shape[0], int(x3[-1, 6]) - x1.shape[0] + 1), axis=0)
                x = np.insert(x, int(x3[0, 6]) - x1.shape[0], x1, axis=0)
                x3[:, 5] = t2
                x = np.insert(x, int(x2[-1, 6]) - x1.shape[0], x3, axis=0)
    else:  # We cannot slice trip 1
        if x2.shape[0] >= 2:
            if x3.shape[0] < 2:  # We can slice trip2 only
                r2 = random.choice(range(1, x2.shape[0]))
                x2 = x2[r2:]
                x2[:, 5] = t1
                x1[:, 5] = t3
                x = np.delete(x, slice(int(x1[0, 6]), int(x1[-1, 6]) + 1), axis=0)
                x = np.insert(x, int(x1[0, 6]), x2, axis=0)
                l1 = x2.shape[0] - x1.shape[0]
                x = np.delete(x, slice(int(x2[0, 6]) + l1, int(x2[-1, 6]) + l1 + 1), axis=0)
                x = np.insert(x, int(x3[-1, 6]) + l1, x1, axis=0)
            else:  # We can slice both trips 2,3 but no trip 1
                r2 = random.choice(range(1, x2.shape[0]))
                r3 = random.choice(range(1, x3.shape[0]))
                x2 = x2[r2:]
                x3 = x3[r3:]
                x1[:, 5] = t3
                x2[:, 5] = t1
                x3[:, 5] = t2
                x = np.delete(x, slice(int(x1[0, 6]), int(x1[-1, 6]) + 1), axis=0)
                x = np.insert(x, int(x1[0, 6]), x2, axis=0)
                l1 = x2.shape[0] - x1.shape[0]
                x = np.delete(x, slice(int(x2[0, 6]) + l1, int(x2[-1, 6]) + 1 + l1), axis=0)
                x = np.insert(x, int(x2[0, 6]) + l1, x3, axis=0)
                l2 = l1 + x3.shape[0] - x2.shape[0]
                x = np.delete(x, slice(int(x3[0, 6]) + l2, int(x3[-1, 6]) + 1 + l2), axis=0)
                x = np.insert(x, int(x3[0, 6]) + l2, x1, axis=0)
        else:
            if x3.shape[0] >= 2:  # I we can slice only trip 3, we merge trips 1,2 and recreate 2 by slicing 3.
                r3 = random.choice(range(1, x3.shape[0]))
                x3 = x3[r3:]
                x2[:, 5] = t1
                x3[:, 5] = t2
                x = np.delete(x, slice(int(x2[0, 6]), int(x2[-1, 6]) + 1), axis=0)
                x = np.insert(x, int(x1[-1, 6]) + 1, x2, axis=0)
                x = np.delete(x, slice(int(x3[0, 6]), int(x3[-1, 6]) + 1), axis=0)
                x = np.insert(x, int(x2[0, 6]) + x2.shape[0], x3, axis=0)
            else:  # We cannot slice any trip, we merge them all in trip 1
                x = np.delete(x, slice(int(x2[0, 6]), int(x2[-1, 6]) + 1), axis=0)
                x = np.delete(x, slice(int(x3[0, 6]) - x2.shape[0], int(x3[-1, 6]) + 1 - x2.shape[0]), axis=0)
                x4 = np.concatenate((x2, x3))
                x4[:, 5] = t1
                x = np.insert(x, int(x1[-1, 6]), x4, axis=0)

    x[:, 6] = sorted(x[:, 6])
    return x, t1, t2, t3


def SA_double_movement2(data):
    x = np.array(pd.read_csv(data))
    distances = distances_cr(data)
    np.random.shuffle(x)
    best_route = trip_cr2(x)
    x = np.c_[x, best_route]
    x = np.c_[x, range(x.shape[0])]
    best = objective_f(x, distances)[0]
    top_best = best
    solutions = 1000 * x.shape[0]
    i = 0
    T = 1
    a = 0.95
    not_accepted = 0
    while i < solutions and not_accepted < 50000:

        # NM4: 1st swap between two gifts , one from trip k and one from trip l

        x1 = np.copy(x)
        k = np.random.randint(x.shape[0], size=1)
        other = x[x[:, 5] != x[k, 5]]
        l = int(np.random.choice(other[:, 6]))
        t1 = x[k, 5]
        t2 = x[l, 5]
        first= x[x[:, 5] == t1]
        second=x[x[:, 5] == t2]
        if first.shape[0]==1 and second.shape[0]==1 :
            x1 = np.delete(x1, int(second[0, 6]), axis=0)
            second[0, 5] = t1
            x1 = np.insert(x1, int(first[0, 6]) + 1, second, axis=0)
            x1[:, 6] = sorted(x1[:, 6])
        else:
            x1[k, :], x1[l, :] = x1[l, :], x1[k, :]
            x1[k, 5], x1[l, 5] = x1[l, 5], x1[k, 5]
            x1[k, 6], x1[l, 6] = x1[l, 6], x1[k, 6]
        trips3 = x[x[:, 5] == t1]
        trips4 = x[x[:, 5] == t2]
        wrw1 = weight_array2(trips3, distances) + weight_array2(trips4, distances)
        trip1 = x1[x1[:, 5] == t1]
        trip2 = x1[x1[:, 5] == t2]
        if np.sum(trip1[:, 4]) > 1000 or np.sum(trip2[:, 4]) > 1000:
            temp1, x1 = objective_f(x1, distances)
        else:
            wrw2 = weight_array2(trip1, distances) + weight_array2(trip2, distances)
            temp1 = best + (wrw2 - wrw1)
        x2, p1, p2, p3 = three_way_suffix2(x)
        prev1 = x[x[:, 5] == p1]
        prev2 = x[x[:, 5] == p2]
        prev3 = x[x[:, 5] == p3]
        wrw3 = weight_array2(prev1, distances) + weight_array2(prev2, distances) + weight_array2(prev3, distances)
        new1 = x2[x2[:, 5] == p1]
        new2 = x2[x2[:, 5] == p2]
        new3 = x2[x2[:, 5] == p3]
        wrw4 = weight_array2(new1, distances) + weight_array2(new2, distances) + weight_array2(new3, distances)
        temp2 = best + (wrw4 - wrw3)
        ID = 1
        for trips in np.unique(x[:, 5]):
            x[x[:, 5] == trips, 5] = ID
            ID += 1
        if temp2 >= temp1:
            best_move = temp1
            new_x = x1
        else:
            best_move = temp2
            new_x = x2
        value = (1 / T) * ((best - best_move) / best)

        if best_move < top_best:
            top_best = best_move
            x = new_x
            best_route = x[:, 5]
            best = best_move
            not_accepted = 0
        elif best_move < best or random.random() < np.exp(value):
            x = new_x
            best = best_move
            best_route = x[:, 5]
            not_accepted = 0
        else:
            not_accepted += 1
        if T > 0.05:
            T = a * T
        i += 1

    return top_best, sorted(x[:, 0])


"""Initialize our coursework"""

csv_creation([10,100,1000]) # First we create the csv files we are gonna deal with , size 10,100,100
files = ['gifts10.csv', 'gifts100.csv', 'gifts1000.csv'] # These are the files created by csv_creation



"""
Create the Data frames that we are gonna use for visualization
"""

random10 = np.reshape(r_seed(files[0],random_search),[30,1])
random100 = np.reshape(r_seed(files[1],random_search),[30,1])
random1000 = np.reshape(r_seed(files[2],random_search),[30,1])

annealing10 = np.reshape(r_seed(files[0],simulated_annealing),[30,1])
annealing100 = np.reshape(r_seed(files[1],simulated_annealing),[30,1])
annealing1000 = np.reshape(r_seed(files[2],simulated_annealing),[30,1])

SA2ble10 = np.reshape(r_seed(files[0],SA_double_movement),[30,1])
SA2ble100 = np.reshape(r_seed(files[1],SA_double_movement),[30,1])
SA2ble1000 = np.reshape(r_seed(files[2],SA_double_movement),[30,1])


gifts10 = pd.DataFrame(np.concatenate((random10, annealing10, SA2ble10), axis=1), columns=['RS', 'SA', 'SA2'])
gifts100 = pd.DataFrame(np.concatenate((random100, annealing100, SA2ble100), axis=1), columns=['RS', 'SA', 'SA2'])
gifts1000 = pd.DataFrame(np.concatenate((random1000, annealing1000, SA2ble1000), axis=1), column


# Now we create the data frames to visualize the varying a results

a_05_10 = np.reshape(r_seed(files[0],SA_double_movement,0.5),[30,1]
a_08_10 = np.reshape(r_seed(files[0],SA_double_movement,0.8),[30,1]
a_09_10 = np.reshape(r_seed(files[0],SA_double_movement,0.9),[30,1]
a_098_10 = np.reshape(r_seed(files[0],SA_double_movement,0.98),[30,1]

a_05_100 = np.reshape(r_seed(files[1],SA_double_movement,0.5),[30,1]
a_08_100 = np.reshape(r_seed(files[1],SA_double_movement,0.8),[30,1]
a_09_100 = np.reshape(r_seed(files[1],SA_double_movement,0.9),[30,1]
a_098_100 = np.reshape(r_seed(files[1],SA_double_movement,0.98),[30,1]

a_05_1000 = np.reshape(r_seed(files[2],SA_double_movement,0.5),[30,1]
a_08_1000 = np.reshape(r_seed(files[2],SA_double_movement,0.8),[30,1]
a_09_1000 = np.reshape(r_seed(files[2],SA_double_movement,0.9),[30,1]
a_098_1000 = np.reshape(r_seed(files[2],SA_double_movement,0.98),[30,1]

a_10 = pd.DataFrame(np.concatenate((random10,a_05_10, a_08_10, a_09_10,SA2ble10,a_098_10), axis=1), columns=['RS', '0.5', '0.8','0.9','0.95','0.98'])
a_1000 = pd.DataFrame(np.concatenate((random1000,a_05_1000, a_08_1000, a_09_1000,SA2ble1000,a_098_1000), axis=1), columns=['RS', '0.5', '0.8','0.9','0.95','0.98'])

a_10_only = pd.DataFrame(np.concatenate((a_05_10, a_08_10, a_09_10,SA2ble10,a_098_10), axis=1), columns=['0.5', '0.8','0.9','0.95','0.98'])
a_100_only = pd.DataFrame(np.concatenate((a_05_100, a_08_100, a_09_100,SA2ble100,a_098_100), axis=1), columns=['0.5', '0.8','0.9','0.95','0.98'])
a_1000_only = pd.DataFrame(np.concatenate((a_05_1000, a_08_1000, a_09_1000,SA2ble1000,a_098_1000), axis=1), columns=['0.5', '0.8','0.9','0.95','0.98'])

"""" Call functions for the plots"""
make_plots1(gifts10, gifts100, gifts1000)
make_plots2(a_10,a_100,a_1000)
make_plots3(a_10_only,a_10_only,a_10_only)


"""
In order to estimate the time needed to run 1 evaluation for the 100.000 
gifts dataset we used the code below
Example below is for the Simulated Annealing times
x is dataset size and y the time needed for 1 iteration
"""


x=np.array([10,100,1000,2000,5000,10000,20000])
y=np.array([0.0001,0.00014,0.00018,0.003,0.0062,0.047,0,45])

def estimate_time(x,y):
    x=x
    y=y
    # transforming the data to include another axis
    y = y[:, np.newaxis]
    x = x[:, np.newaxis]
    polynomial_features= PolynomialFeatures(degree=3)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    y_poly_pred
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    r2 = r2_score(y,y_poly_pred)
    print(rmse)
    print(r2)

    plt.scatter(x, y, s=10)
    # sort the values of x before line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(x,y_poly_pred), key=sort_axis)
    x, y_poly_pred = zip(*sorted_zip)
    plt.plot(x, y_poly_pred, color='m')
    plt.show()
    print(model.intercept_)
    #For retrieving the slope:
    print(model.coef_)
    print('Slope:' ,model.coef_)
    print('Intercept:', model.intercept_)
