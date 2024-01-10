import csv
from tqdm import tqdm
import os

# adds to every row (candle) an anotation of next row (candle) close value
def anotate_target_value(input_file, output_file):

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
    input_file_path = os.path.join(os.path.dirname(__file__),'data', input_file)
    output_file_path = os.path.join(os.path.dirname(__file__),'data',  output_file)

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
            #append at the end oh the row upcoming close valye
            current_row.append(close_diffrence) # add the 0 or 1 at the end of the file
            # Writing the concatenated row to the CSV file
            writer.writerow(current_row)
    

if __name__ == '__main__':
    input_csv_file = 'test_target_v_BTCUSDT.csv'
    output_csv_file = 'test_target_v_BTCUSDT2.csv'
    anotate_target_difference(input_csv_file, output_csv_file)