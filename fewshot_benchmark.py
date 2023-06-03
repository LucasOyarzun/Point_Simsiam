import numpy as np

### PointSimsiam transformer_linear_mask_05_08
## 5w10s
# results = [96, 100, 95, 92, 99, 97, 98, 94, 96, 92]
### PointSimsiam transformer_linear_mask_05_08
## 5w20s
results = [96, 99, 99, 98, 97, 99, 95, 95]
mean = np.mean(results)
std = np.std(results)
print("mean: ", mean)
print("std: ", std)
