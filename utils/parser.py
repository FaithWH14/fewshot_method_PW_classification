import argparse
import os

with open(os.getcwd() + os.sep + "config.txt", "r") as f:
    dir = [line.rstrip() for line in f]
    input_dir = dir[0].split("=")[1] 
    input_dir1 = dir[1].split("=")[1]
def get_arg():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d1', '--datapath', type=str, help='PW support data_path', default= input_dir)
    argparser.add_argument('-d2', '--datapath1', type=str, help='PW query data_path', default= input_dir)
    argparser.add_argument('-q', '--val_support_num', type=int, default=2)
    argparser.add_argument('-s', '--support_num', type=int, default=5)

    args = argparser.parse_args()
    return args