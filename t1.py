
"""
import numpy as np

t = np.load("dataset2d_ri_feats.npy")


for i in t:
    print(i)
    break
"""

with open("data/lndb_sphere_r6_hu.csv","r") as f:
    header0, *content0 = f.readlines()

header0 = header0[:-1]
header0 += ","+",".join([f"Feature{i}" for i in range(2048)])+"\n"

l = []

import numpy as np

t = np.load("dataset2d_ri_feats.npy")

for i,j in zip(content0,t[15::32]):
    i = i[:-1]
    i+=","+",".join(map(str,j))+"\n"
    l.append(i)

with open("dataset2allFeat.csv","w") as f:
    f.write(header0)
    f.write("".join(l))

"""
with open("lndb_circle_r6_hu.csv","r") as f:
    header0, *content0 = f.readlines()

header0 = header0[:-1].split(",")
end_i = header0.index("fold")
header0 = header0[:end_i+1]
header0= ",".join(header0)+","+",".join([f"Feature{i}" for i in range(2048)])+"\n"

l = []

import numpy as np

t = np.load("dataset2d_ri_feats.npy")

p = "\""
for i,j in zip(content0,t[15::32]):
    i = i[:-1]
    i = [j for k,i in enumerate(i.split("\"")) for j in ([p+i+p] if k%2 else i.strip(",").split(","))]
    i = ",".join(i[:end_i+1])+","+",".join(map(str,j))+"\n"
    l.append(i)


with open("dataset3allFeat.csv","w") as f:
    f.write(header0)
    f.write("".join(l))
"""
