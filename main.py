# ===========Fruits Classification Program==================================
#  *   K-Nearest Neighbors Algorithm
# =============================================================================
#  *   Created By Eric (Ida Bagus Dwi Putra Purnawa).
#  *   Github (https://github.com/EricCornetto).
# =============================================================================
#  *   Start Project Version 1.0 = 22 February 2019.
#  *   End Project Version 1.0 = 22 February 2019.
#  *   Update Project Version = Coming Soon.
#  *   GNU General Public License v3.0.
# =============================================================================
#             Python Machine Learning
# =============================================================================
import pandas as pd
import numpy as np
import math

# K-Nearest Neighbors Model
class KNearestNeighbor():
    # fitting data
    def fit(self,x_train,y_train,labels):
        self.x_train = x_train
        self.y_train = y_train
        self.labels = labels

    # find distance Neighbors to X
    def distance(self,inputData):
        len_data = len(self.x_train)
        distanceX = np.array([])
        for i in range(len_data):
            x = self.x_train[i] - inputData[0]
            y = self.y_train[i] - inputData[1]
            xSQR = np.square(x)
            ySQR = np.square(y)
            result = xSQR + ySQR
            resultSQRT = math.sqrt(result)
            distanceX = np.append(distanceX,resultSQRT)
        return distanceX

    # Predict the input data
    def predict(self,inputData):
        distanceX = self.distance(inputData)
        len_data = len(self.x_train)
        distanceXList = []
        for i in range(len_data):
            distanceXList.append(distanceX[i])
        kValues = []
        if len_data %2 == 0:
            for i in range(7):
                minValues = min(distanceXList)
                minIndex = distanceXList.index(minValues)
                kValues.append(self.labels[minIndex])
                distanceXList.remove(minValues)
        elif len_data %2 == 1:
            for i in range(2):
                minValues = min(distanceXList)
                minIndex = distanceXList.index(minValues)
                kValues.append(self.labels[minIndex])
                distanceXList.remove(minValues)
        return kValues

# Main Program
def main():
    # rawa dataset
    raw_dataset = pd.read_csv("Fruits_Data.csv")
    x_raw = raw_dataset.iloc[:,0]
    y_raw = raw_dataset.iloc[:,1]
    labels_raw = raw_dataset.iloc[:,2]

    # turn the raw data to array
    x = np.asarray(x_raw)
    y = np.asarray(y_raw)
    labels = np.asarray(labels_raw)

    # classifier
    classifier = KNearestNeighbor()
    classifier.fit(x,y,labels)

    # Input Data
    print()
    print("==================Fruits Classification Program======================")
    print("================== Created By Eric T. Cornetto=======================")
    x_input = float(input("Input Width Fruits : "))
    y_input = float(input("Input Height Fruits : "))
    pred = classifier.predict([x_input,y_input]) # predictions
    print("=====================================================================")
    print("K-Nearest : {}".format(pred))

if __name__ == "__main__":
    main()
