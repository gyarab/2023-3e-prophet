# this script servers as creator of training data
# note that u have to have a "data" folder relative to this script - the script reads from it and writes into it

import csv # lib for csv files access
from tqdm import tqdm # loop progress bar in terminal
import os # file paths


def fully_anotate(input_file, output_file):
    # <anotate technical factors>
    anotate_target_value(input_file, output_file)
    anotate_target_difference(input_file, output_file)
# returns an absolute path of file that is contained in data folder
# data folder has to be relative to this script
def get_absolute_path(input_file):
    input_file_path = os.path.join(os.path.dirname(__file__),'data', input_file)
    return input_file_path

# adds to every row (candle) an anotation of next row (candle) close value
def anotate_target_value(input_file, output_file):

    # Get the absolute path of the input CSV file
    input_file_path = get_absolute_path(input_file)
    output_file_path = get_absolute_path(output_file)

    # Reading the input CSV file
    with open(input_file_path, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Creating a new CSV file
    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Writing rows to the new CSV file
        for i in tqdm(range(len(data) - 1), desc="Processing rows"):
            # loads curent row
            current_row = data[i]
            # gets upcoming close value
            upcoming_close_value = float(data[i+1][4])  # 'close' value from upcoming set of 60
            # append at the end oh the row upcoming close valye
            current_row.append(upcoming_close_value) # add the 0 or 1 at the end of the file
            # Writing the concatenated row to the CSV file
            writer.writerow(current_row)

# adds to every row (candle) an anotation of next row (candle) close value difference
def anotate_target_difference(input_file, output_file):
    
    # Get the absolute path of the input CSV file
    input_file_path = get_absolute_path(input_file)
    output_file_path = get_absolute_path(output_file)

    # Reading the input CSV file
    with open(input_file_path, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Creating a new CSV file
    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        
        # Writing rows to the new CSV file
        for i in tqdm(range(len(data) - 1), desc="Processing rows"):
            # loads curent row
            current_row = data[i]
            # gets upcoming close value
            upcoming_close_value = float(data[i+1][4])  # 'close' value from upcoming set of 60
            # substract the current close valye with the upcoming close value
            close_diffrence = float(current_row[4]) - upcoming_close_value
            # rounds close_difference to three decimal digits (most data will be rounded to just two)
            rounded_close_difference = round (close_diffrence, 5)
            #append at the end oh the row upcoming close valye
            current_row.append(rounded_close_difference) # add the 0 or 1 at the end of the file
            # Writing the concatenated row to the CSV file
            writer.writerow(current_row)
    
# this function will create a new csv file that will put 0-59 row into one row and will add 0 or 1 at the end based upcoming close value
# the order example : 0 - 59 , 1 - 60 , 2 - 61 , 3 - 62 and so on
def create_new_csv(input_file, output_file):
    
    # Get the absolute path of the input CSV file
    input_file_path = get_absolute_path(input_file)
    output_file_path = get_absolute_path(output_file)
    # Reading the input CSV file
    with open(input_file_path, 'r') as infile:
        reader = csv.reader(infile)
        data = list(reader)

    # Creating a new CSV file
    with open(output_file_path, 'w', newline='') as outfile:
        writer = csv.writer(outfile)

        # Writing headers to the new CSV file
        #writer.writerow(data[0])

        # Writing rows to the new CSV file
        for i in tqdm(range(len(data) - 60), desc="Processing rows"):
            # Concatenate the rows into a single list
            concatenated_row = [item for sublist in data[i:i+60] for item in sublist]
            
            # Determine if to append 0 or 1 based on 'close' values
            current_close_value = float(data[i+59][4])  # 'close' value from current set of 60
            upcoming_close_value = float(data[i+60][4])  # 'close' value from upcoming set of 60

            # Append 0 or 1 to the concatenated row
            appended_value = 0 if current_close_value > upcoming_close_value else 1 # if upcoming value is same or bigger -> 1
            concatenated_row.append(appended_value) # add the 0 or 1 at the end of the file
            
            # Writing the concatenated row to the CSV file
            writer.writerow(concatenated_row)
            
if __name__ == '__main__':
    input_csv_file = 'test_target_v_BTCUSDT.csv'
    output_csv_file = 'test_target_v_BTCUSDT2.csv'
    anotate_target_difference(input_csv_file, output_csv_file)