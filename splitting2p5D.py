import sys
from collections import defaultdict
with open(sys.argv[1],"r") as f:
    content = f.readlines()
header,*data = content
slice_i = header.split(",").index("slice_in_nodule")

files = defaultdict(list)

print("Splitting the dataset...")
for i in data:
    files[eval(f"[{i.replace(',,',',0,')[:-1]}]")[slice_i]].append(i)
print("Writing files...")
for i in files:
    with open(f"{sys.argv[1][:-4]}{i}.csv","w") as f:
        f.write(header)
        f.write("".join(files[i]))

