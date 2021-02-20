import os
import time

with open("./data/download.txt", "r") as f:
    txt = f.read()

txt = [t.strip() for t in txt.split(" ")]

with open("./data/download.txt", "w") as f:
    f.writelines([t + "\n" for t in txt])

os.system("wget -w 3 -i ./data/download.txt -P ./data")
