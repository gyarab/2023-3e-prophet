import csv
import tqdm
import os

#adds to every row (candle) an anotation of next row (candle) close value
def anotate_target_value(input_file, output_file):

    # Get the absolute path of the input CSV file
    input_file_path = os.path.join(os.path.dirname(__file__),'data', input_file)
    output_file_path = os.path.join(os.path.dirname(__file__),'data',  output_file)
    

#adds to every row (candle) an anotation of next row (candle) close value difference
def anotate_target_difference():
    pass

if __name__ == '__name__':
    input_csv_file = 'BTCUSDT.csv'
    output_csv_file = 'target_v_BTCUSDT.csv'
    anotate_target_value(input_csv_file, output_csv_file)