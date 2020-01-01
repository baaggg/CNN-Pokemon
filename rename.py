import os
import time

path = "C:/Users/brand/Desktop/Python/CNN-Pokemon/pokemon/bulbasaur/"

start_time = time.time()
i = 1
for filename in os.listdir(path):
    type = filename.split('.')[1]
    dest = 'bulb' + str(i) + '.' + type
    source = path + filename
    dest = path + dest

    os.rename(source, dest)
    i += 1

num_files = len(os.listdir(path))
duration = round((time.time() - start_time), 3)
print(num_files, 'files were renamed in', duration, 'seconds.')
