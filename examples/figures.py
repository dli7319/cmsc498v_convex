import re
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from pandas.api.types import CategoricalDtype
from plotnine import ggplot, aes,\
    geom_point, geom_line,\
    scale_color_discrete, scale_shape_discrete, scale_color_brewer,\
    xlab, ylab, theme, element_rect, facet_grid, ggtitle


to_digit = re.compile("tensor\(([\d\.]*),")
remove_nl = re.compile("\n")

# Test 1

baseline_file = open("test1/baseline_full_test.log", "r")
baseline_arr = baseline_file.readlines()
baseline_arr2 = []
for i, val in enumerate(baseline_arr):
    filteredstr = re.sub("\n", "", val)
    filteredstr = re.sub("\,\s", ",", filteredstr)
    baseline_arr2.append(filteredstr.split(" "))
    baseline_arr2[i][1] = re.match("tensor\(([\d\.]*),", baseline_arr2[i][1]).group(1)
    baseline_arr2[i][2] = re.match("tensor\(([\d\.]*),", baseline_arr2[i][2]).group(1)

# [time, baseline_err, madry_err, robust_err]
baseline_arr2 = np.array(baseline_arr2, dtype=float)

madry_file = open("test1/madry_full_test.log", "r")
madry_arr = madry_file.readlines()
madry_arr2 = []
for i, val in enumerate(madry_arr):
    filteredstr = re.sub("\n", "", val)
    filteredstr = re.sub("\,\s", ",", filteredstr)
    madry_arr2.append(filteredstr.split(" "))
    madry_arr2[i][1] = re.match("tensor\(([\d\.]*),", madry_arr2[i][1]).group(1)
    madry_arr2[i][2] = re.match("tensor\(([\d\.]*),", madry_arr2[i][2]).group(1)

# [time, madry_err, madry_err, robust_err]
madry_arr2 = np.array(madry_arr2, dtype=float)


robust_file = open("test1/robust_full_test.log", "r")
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


mix_file = open("test1/mix_full_test.log", "r")
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

# plt.plot(baseline_arr2[:,0], baseline_arr2[:,1], label='baseline')
# plt.plot(madry_arr2[:,0], madry_arr2[:,1], label='madry')
# plt.plot(robust_arr2[:,0], robust_arr2[:,1], label='robust')
# plt.plot(mix_arr2[:,0], mix_arr2[:,1], label='mix')
# print(baseline_arr2[:,0])
# print(madry_arr2[:,0])

# plt.xlabel('Time (seconds)')
# plt.ylabel('Error')
# plt.title("Baseline Error")
# plt.legend()
# plt.savefig("test1/baselineErr.png")
# plt.clf()


# plt.plot(baseline_arr2[:,0], baseline_arr2[:,2], label='baseline')
# plt.plot(madry_arr2[:,0], madry_arr2[:,2], label='madry')
# plt.plot(robust_arr2[:,0], robust_arr2[:,2], label='robust')
# plt.plot(mix_arr2[:,0], mix_arr2[:,2], label='mix')
# print(baseline_arr2[:,0])
# print(madry_arr2[:,0])

# plt.xlabel('Time (seconds)')
# plt.ylabel('Error')
# plt.title("Madry (PGD) Error")
# plt.legend()
# plt.savefig("test1/madryErr.png")
# plt.clf()


# plt.plot(baseline_arr2[:,0], baseline_arr2[:,3], label='baseline')
# plt.plot(madry_arr2[:,0], madry_arr2[:,3], label='madry')
# plt.plot(robust_arr2[:,0], robust_arr2[:,3], label='robust')
# plt.plot(mix_arr2[:,0], mix_arr2[:,3], label='mix')
# print(baseline_arr2[:,0])
# print(madry_arr2[:,0])

# plt.xlabel('Time (seconds)')
# plt.ylabel('Error')
# plt.title("Robust Error")
# plt.legend()
# plt.savefig("test1/robustErr.png")
# plt.clf()

x = {}
y = {}
method = {}

index = 0
for element1, element2 in zip(baseline_arr2[:,0], baseline_arr2[:,1]):    
    x[str(index)] = element1    
    y[str(index)] = element2   
    method[str(index)] = "Baseline"
    index = index + 1

for element1, element2 in zip(madry_arr2[:,0], madry_arr2[:,1]):    
    x[str(index)] = element1    
    y[str(index)] = element2    
    method[str(index)] = "PGD"
    index = index + 1

for element1, element2 in zip(robust_arr2[:,0], robust_arr2[:,1]):    
    x[str(index)] = element1    
    y[str(index)] = element2    
    method[str(index)] = "Provable"
    index = index + 1

for element1, element2 in zip(mix_arr2[:,0], mix_arr2[:,1]):    
    x[str(index)] = element1    
    y[str(index)] = element2    
    method[str(index)] = "Mix"
    index = index + 1

jsonFile = {}
jsonFile['method'] = method
jsonFile['x'] = x
jsonFile['y'] = y

def baseline_error():    
    al_baselines_len = len(jsonFile['x'])    
    df = pd.DataFrame(jsonFile)

    method_type = CategoricalDtype(
        categories=['Baseline', 'PGD', 'Provable', 'Mix'], ordered=True)
    df['method'] = df['method'].astype(method_type)

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
                # manually position legend, not used anymore
                # legend_position=(0.70, 0.65),
                # legend_background=element_rect(fill=(0, 0, 0, 0)),
                aspect_ratio=0.8,
            )         
            + ggtitle("Baseline Error")   
    )
    fig_dir = '.'
    p.save(os.path.join(fig_dir, 'baselineErr.pdf'))

if __name__ == '__main__':
    baseline_error()
    # madry_error()
    # robust_error()    