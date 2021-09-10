import os, csv
from tqdm import tqdm, trange
import argparse

# file paths
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='input folder')
parser.add_argument('--output', type=str, required=True, help='output folder')
args = parser.parse_args()

data_dir_input = args.input
data_dir_out = args.output

train_0 = os.path.join(data_dir_input ,"train.0")
train_1 = os.path.join(data_dir_input,"train.1")
test_0 = os.path.join(data_dir_input,"test.0")
test_1 = os.path.join(data_dir_input,"test.1")
dev_0 = os.path.join(data_dir_input,"dev.0")
dev_1 = os.path.join(data_dir_input,"dev.1")

train_out = os.path.join(data_dir_out,"train.csv")
dev_out = os.path.join(data_dir_out,"dev.csv")
test_out = os.path.join(data_dir_out,"test.csv")

def create_classification_file(input_file_path_list, input_file_label_list, output_file_path):
    """
    Create a csv file combining training data for BERT classification training.
    input_file_path_list : Path of the input files
    input_file_label_list : Correspoding labels for input_files
    output_file_path : csv file path
    """
    with open(output_file_path, "w") as out_fp:
        writer = csv.writer(out_fp, delimiter="\t")
        for i, file in enumerate(tqdm(input_file_path_list)):
            with open(file) as fp:
                for line in fp:
                    writer.writerow([line.strip(),input_file_label_list[i]])

create_classification_file([train_0,train_1],[0,1], train_out)
create_classification_file([test_0,test_1],[0,1], test_out)
create_classification_file([dev_0,dev_1],[0,1], dev_out)