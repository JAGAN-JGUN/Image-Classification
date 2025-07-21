import splitfolders
import sys

if(len(sys.argv) == 2):
    splitfolders.ratio(sys.argv[1],output=".",seed=1337,ratio=(.9,.1),group_prefix=None,move=False)
else:
    print("Usage: python Split.py <path_to_dataset>")
    print("Example: python Split.py ./Data")
    print("This will split the dataset into train and validation sets with a 90/10 ratio.")