import os, csv
from tqdm import tqdm, trange
import argparse

# file paths

data_dir = "./data/"
dataset = "gyafc" # yelp/gyafc/cs
train_0 = os.path.join(data_dir ,"./{}/train.0".format(dataset))
train_1 = os.path.join(data_dir,"./{}/train.1".format(dataset))
test_0 = os.path.join(data_dir,"./{}/test.0".format(dataset))
test_1 = os.path.join(data_dir,"./{}/test.1".format(dataset))
dev_0 = os.path.join(data_dir,"./{}/dev.0".format(dataset))
dev_1 = os.path.join(data_dir,"./{}/dev.1".format(dataset))


train_out = os.path.join(data_dir,"{}/train.csv".format(dataset))
dev_out = os.path.join(data_dir,"{}/dev.csv".format(dataset))
test_out = os.path.join(data_dir,"{}/test.csv".format(dataset))

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