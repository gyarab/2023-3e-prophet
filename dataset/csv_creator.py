# this script servers as creator of training data
# note that u have to have a "data" folder relative to this script - the script reads from it and writes into it

import csv # csv file manipulation
from tqdm import tqdm # progress bars, from the specific module import specific class
import os # file paths 

# this function will create a new csv file that will put 0-59 row into one row and will add 0 or 1 at the end based upcoming close value
# the order example : 0 - 59 , 1 - 60 , 2 - 61 , 3 - 62 and so on
def create_new_csv(input_file, output_file):
    
    # Get the absolute path of the input CSV file
    input_file_path = os.path.join(os.path.dirname(__file__),'data', input_file)
    output_file_path = os.path.join(os.path.dirname(__file__),'data',  output_file)
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
            current_close_value = float(data[i+59][2])  # 'close' value from current set of 60
            upcoming_close_value = float(data[i+60][2])  # 'close' value from upcoming set of 60

            # Append 0 or 1 to the concatenated row
            appended_value = 0 if current_close_value > upcoming_close_value else 1 # if upcoming value is same or bigger -> 1
            concatenated_row.append(appended_value) # add the 0 or 1 at the end of the file
            
            # Writing the concatenated row to the CSV file
            writer.writerow(concatenated_row)


if __name__ == '__main__':
    # Specify your input and output file paths
    input_csv_file = 'test_BTCUSDT.csv'
    output_csv_file = 'test_60BTCUSDT.csv'

    # Call the function to create the new CSV file
    create_new_csv(input_csv_file, output_csv_file)
