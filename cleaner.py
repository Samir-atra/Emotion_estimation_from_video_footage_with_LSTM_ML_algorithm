import os
import time

# a loop to delete the files in the checkpoints directory and leave the last two, to save the disk space from the unneccessary models saved by 
while True:
    oo = os.listdir('/home/samer/Desktop/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/ckpt/')
    if len(oo) > 2:
        for i in sorted(oo)[0:-2]:
            os.remove('/home/samer/Desktop/Big data Small Data/BDSD/Minor_project/BDSD_Minor_Project/ckpt/'+str(i))
    time.sleep(600)        
