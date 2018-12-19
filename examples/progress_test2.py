
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.api.types import CategoricalDtype
from plotnine import ggplot, aes,\
    geom_point, geom_line,\
    scale_color_discrete, scale_shape_discrete, scale_color_brewer,\
    xlab, ylab, theme, element_rect, facet_grid, ggtitle


to_digit = re.compile("tensor\(([\d\.]*),")
remove_nl = re.compile("\n")

# Test 2
test = "test2"

#
# baseline_file = open(test+"/baseline_full_test.log", "r")
# baseline_arr = baseline_file.readlines()
# baseline_arr2 = []
# for i, val in enumerate(baseline_arr):
#     filteredstr = re.sub("\n", "", val)
#     filteredstr = re.sub("\,\s", ",", filteredstr)
#     baseline_arr2.append(filteredstr.split(" "))
#     baseline_arr2[i][1] = re.match("tensor\(([\d\.]*),", baseline_arr2[i][1]).group(1)
#     baseline_arr2[i][2] = re.match("tensor\(([\d\.]*),", baseline_arr2[i][2]).group(1)
#
# # [time, baseline_err, madry_err, robust_err]
# baseline_arr2 = np.array(baseline_arr2, dtype=float)

# madry_file = open(test+"/madry_full_test.log", "r")
# madry_arr = madry_file.readlines()
# madry_arr2 = []
# for i, val in enumerate(madry_arr):
#     filteredstr = re.sub("\n", "", val)
#     filteredstr = re.sub("\,\s", ",", filteredstr)
#     madry_arr2.append(filteredstr.split(" "))
#     madry_arr2[i][1] = re.match("tensor\(([\d\.]*),", madry_arr2[i][1]).group(1)
#     madry_arr2[i][2] = re.match("tensor\(([\d\.]*),", madry_arr2[i][2]).group(1)
#
# # [time, madry_err, madry_err, robust_err]
# madry_arr2 = np.array(madry_arr2, dtype=float)


robust_file = open(test+"/robust_full_test.log", "r")
robust_arr = robust_file.readlines()
robust_arr2 = []
for i, val in enumerate(robust_arr):
    filteredstr = re.sub("\n", "", val)
    filteredstr = re.sub("\,\s", ",", filteredstr)
    robust_arr2.append(filteredstr.split(" "))
    robust_arr2[i][1] = re.match("tensor\(([\d\.]*),", robust_arr2[i][1]).group(1)
    robust_arr2[i][2] = re.match("tensor\(([\d\.]*),", robust_arr2[i][2]).group(1)

# [time, robust_err, madry_err, robust_err]
robust_arr2 = np.array(robust_arr2, dtype=float)


mix_file = open(test+"/mix_full_test.log", "r")
mix_arr = mix_file.readlines()
mix_arr2 = []
for i, val in enumerate(mix_arr):
    filteredstr = re.sub("\n", "", val)
    filteredstr = re.sub("\,\s", ",", filteredstr)
    mix_arr2.append(filteredstr.split(" "))
    mix_arr2[i][1] = re.match("tensor\(([\d\.]*),", mix_arr2[i][1]).group(1)
    mix_arr2[i][2] = re.match("tensor\(([\d\.]*),", mix_arr2[i][2]).group(1)

# [time, mix_err, madry_err, robust_err]
mix_arr2 = np.array(mix_arr2, dtype=float)




# plt.plot(robust_arr2[:,0], robust_arr2[:,1], label='robust')
# plt.plot(mix_arr2[:,0], mix_arr2[:,1], label='mix')
#
# plt.xlabel('Time (seconds)')
# plt.ylabel('Error')
# plt.title("Baseline Error")
# plt.legend()
# plt.savefig(test+"/baselineErr.png")
# plt.clf()


# plt.plot(robust_arr2[:,0], robust_arr2[:,2], label='robust')
# plt.plot(mix_arr2[:,0], mix_arr2[:,2], label='mix')
#
# plt.xlabel('Time (seconds)')
# plt.ylabel('Error')
# plt.title("Madry (PGD) Error")
# plt.legend()
# plt.savefig(test+"/madryErr.png")
# plt.clf()


# plt.plot(robust_arr2[:,0], robust_arr2[:,3], label='robust')
# plt.plot(mix_arr2[:,0], mix_arr2[:,3], label='mix')
#
# plt.xlabel('Time (seconds)')
# plt.ylabel('Error')
# plt.title("Robust Error")
# plt.legend()
# plt.savefig(test+"/robustErr.png")
# plt.clf()


df = pd.DataFrame([], columns=["method", "x", "y"])
df = df.append(pd.DataFrame(
    np.transpose([np.repeat("Provable", robust_arr2.shape[0]),
     robust_arr2[:,0],
     robust_arr2[:,1]]), columns=["method", "x", "y"]))
df = df.append(pd.DataFrame(
    np.transpose([np.repeat("Mix", mix_arr2.shape[0]),
    mix_arr2[:,0],
    mix_arr2[:,1]]), columns=["method", "x", "y"]))
df['x'] = pd.to_numeric((df['x']))
df['y'] = pd.to_numeric((df['y']))
p = (
    ggplot(df)
    + aes(x='x', y='y', shape='method', color='method')
    # + geom_point(size=4, stroke=0)
    + geom_line(size=1)
    + scale_shape_discrete(name='Method')
    + scale_color_brewer(
        type='qual',
        palette=2,
        name='Method')
    + xlab('Training Time (seconds)')
    + ylab('Error')
    + theme(
        aspect_ratio=0.8,
    )
    + ggtitle("Baseline Error")
)
p.save(test+"/baselineErr.png", verbose=False)



df = pd.DataFrame([], columns=["method", "x", "y"])
df = df.append(pd.DataFrame(
    np.transpose([np.repeat("Provable", robust_arr2.shape[0]),
     robust_arr2[:,0],
     robust_arr2[:,2]]), columns=["method", "x", "y"]))
df = df.append(pd.DataFrame(
    np.transpose([np.repeat("Mix", mix_arr2.shape[0]),
    mix_arr2[:,0],
    mix_arr2[:,2]]), columns=["method", "x", "y"]))
df['x'] = pd.to_numeric((df['x']))
df['y'] = pd.to_numeric((df['y']))
p = (
    ggplot(df)
    + aes(x='x', y='y', shape='method', color='method')
    # + geom_point(size=4, stroke=0)
    + geom_line(size=1)
    + scale_shape_discrete(name='Method')
    + scale_color_brewer(
        type='qual',
        palette=2,
        name='Method')
    + xlab('Training Time (seconds)')
    + ylab('Adversarial Error')
    + theme(
        aspect_ratio=0.8,
    )
    + ggtitle("PGD Error")
)
p.save(test + "/madryErr.png", verbose=False)




df = pd.DataFrame([], columns=["method", "x", "y"])
df = df.append(pd.DataFrame(
    np.transpose([np.repeat("Provable", robust_arr2.shape[0]),
     robust_arr2[:,0],
     robust_arr2[:,3]]), columns=["method", "x", "y"]))
df = df.append(pd.DataFrame(
    np.transpose([np.repeat("Mix", mix_arr2.shape[0]),
    mix_arr2[:,0],
    mix_arr2[:,3]]), columns=["method", "x", "y"]))
df['x'] = pd.to_numeric((df['x']))
df['y'] = pd.to_numeric((df['y']))
p = (
    ggplot(df)
    + aes(x='x', y='y', shape='method', color='method')
    + geom_point(size=4, stroke=0)
    + geom_line(size=1)
    + scale_shape_discrete(name='Method')
    + scale_color_brewer(
        type='qual',
        palette=2,
        name='Method')
    + xlab('Training Time (seconds)')
    + ylab('Adversarial Error')
    + theme(
        aspect_ratio=0.8,
    )
    + ggtitle("Robust Error")
)
p.save(test + "/robustErr.png", verbose=False)
