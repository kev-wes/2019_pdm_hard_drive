import pandas as pd
import numpy as np
import math
from scipy.spatial import distance

### Step 1: Alternative matrix and criteria weights

# Possible fleet_type = ['Identical', 'Homogeneous', 'Heterogeneous']
fleet_type = ['Homogeneous']
# Possible fleet_feature_type = ['Numerical', 'Categorical', 'Semantics', 'None']
fleet_feature_type = ['Categorical', 'None']
# Possible output_type = ['Point-estimate', 'Interval', 'Distribution']
output_type = ['Point-estimate', 'Interval', 'Distribution']
# Alternative/Criteria matrix
data = pd.read_csv("Results.csv", sep = ";")
# Filter on elimination criteria
data = data[(data['Fleet Type'].isin(fleet_type)) &
            (data['Output Type'].isin(output_type)) &
            (data['Fleet Feature Type'].isin(fleet_feature_type))].copy().reset_index(drop=True)
A = data.iloc[:,4:13].values.astype(float)

# Criteria weight
#-------- Adjust weight according to use case --------#
w = np.array([1, 4, 0, 10, 4, 4, 0, 0, 3]).astype(float)
#-----------------------------------------------------#

# Normalize criteria
w = w / w.sum()

### Step 2: Normalize matrix
A_numrows = len(A)
A_numcols = len(A[0])

N = A.copy()
for j in range(0, A_numcols):
    A_column_length = 0
    # Calculate column vector length
    for i in range(0, A_numrows):
        A_column_length = (A[i][j]) ** 2 + A_column_length
    A_column_length = math.sqrt(A_column_length)
    # Divide each element by its column vector length
    for i in range(0, A_numrows):
        N[i][j] = (A[i][j]) / A_column_length

### Step 3: Weigh matrix
W = N.copy()
for j in range(0, A_numcols):
    for i in range(0, A_numrows):
        W[i][j]= w[j]* N[i][j]

### Step 4: Compute PIS and NIS
pis = np.max(W, axis = 0)
nis = np.min(W, axis = 0)

### Step 5: Compute distances from each alternative to PIS and NIS
dist_mat = np.zeros((A_numrows, 2))
for i in range(0, A_numrows):
    dist_mat[i, 0] = distance.euclidean(W[i, :], pis)
    dist_mat[i, 1] = distance.euclidean(W[i, :], nis)

### Step 6 Calculate similarity to PIS and NIS
similarity_mat = np.zeros((A_numrows, 1))
for i in range(0, A_numrows):
    similarity_mat[i] = dist_mat[i, 1]/(dist_mat[i, 1] + dist_mat[i, 0])

result = pd.concat([data.iloc[:,0], pd.DataFrame(similarity_mat)], axis=1)
result.columns = ['Alternative', 'Score']
print(result.sort_values(by=['Score'], ascending = False))