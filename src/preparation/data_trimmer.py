import os
from os.path import join
import pandas as pd
import random
import csv
import math

# The reason for this script is to read the records from the CSV files
# which store large amounts of 'BENIGN' records which we want to trim down to a manageable size (i.e. 25 - 30K).
# We utilize the Random class to pick an arbitrary count of 'BENIGN' network records we want to trim down to.
# This will help with the undersampling of the 'BENIGN' records and since we will run an oversampling on the 
# non 'BENIGN' records during pre-processing, we do not filter out these records at this stage.

# defined lower bound for generating a random number
LOWER_BOUND = 31000

# defined upper bound for generating a random number
UPPER_BOUND = 32000

masterOutputFilename = 'CICIDS2017_Train.csv'
rawDataPath = '../../data/processed/'
#
# Name: getRandomNumber
# 
# Description: Returns a random number between a lower bound and upper bound
#
def getRandomNumber(lowerLimit=LOWER_BOUND, upperLimit=UPPER_BOUND):
    return random.randint(
        lowerLimit,upperLimit)

#
# Name: processFile
# 
# Description: Provided a root folder and filename, we are to create a new file
# containing a limited amount of 'BENIGN' records in an output file for the preprocessing stage.
#  
def processFile(dataroot, filename, upperLimit):
    
    writeToPath = join(rawDataPath, masterOutputFilename)
    openFile = join(dataroot, filename)
    print("Processing File: {}".format(openFile))

    with open(writeToPath, mode = 'a') as outfile:
        print("Created Master Training File: {}".format(writeToPath))
        print("Start Writing to Master Training File...")
        with open(openFile, mode = 'r') as infile:
            reader = csv.reader(infile, delimiter=',')
            rownum = 0

            # get the header information
            header = next(reader)

            # loop through and print the header information
            for col in header:   
                outfile.write(col + ',')
            outfile.write("\n")

            # print the 'BENIGN' records up to the upper limit
            # and any other non 'BENIGN' records
            #
            for row in reader:
                targetCol = row[-1]
                if targetCol == "BENIGN" and rownum < upperLimit or targetCol != "BENIGN":

                    for col in row:
                        outfile.write(col + ',')
                    outfile.write("\n")

                rownum += 1
    print("Finished Processing File: {}\n".format(openFile))
#
# Name: main
#
# Description: Main driver for the program
#
def main():
    dataroot = '../../data/raw/'
    originalMondayFile = 'Monday-WorkingHours.pcap_ISCX.csv'
    originalTuesdayFile = 'Tuesday-WorkingHours.pcap_ISCX.csv'
    originalWednesdayFile = 'Wednesday-workingHours.pcap_ISCX.csv'
    originalThursdayMorningFile = 'Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv'
    originalThursdayEveningFile = 'Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv'
    originalFridayAfternoonDDos = 'Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv'
    originalFridayAfternoonPortScan = 'Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'

    processFile(dataroot, originalMondayFile, getRandomNumber())
    processFile(dataroot, originalTuesdayFile, getRandomNumber())
    processFile(dataroot, originalWednesdayFile, getRandomNumber())
    processFile(dataroot, originalThursdayMorningFile, getRandomNumber())
    processFile(dataroot, originalThursdayEveningFile, getRandomNumber())
    processFile(dataroot, originalFridayAfternoonDDos, getRandomNumber())
    processFile(dataroot, originalFridayAfternoonPortScan, getRandomNumber())

if __name__ == "__main__":
    main()