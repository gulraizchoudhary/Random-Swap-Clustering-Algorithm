# Random-Swap-Clustering-Algorithm
Random swap is a kmeans variant that does not stuck in local optima and finds the optimal partitions by removing a random centroid and adding a random centroid. This strategy work pretty well. You just need to book keep the successful swaps where distortion value reduced further. 

## Usage

```python
#
from rs import rs
import numpy as np
from sklearn.cluster import KMeans

#random 2D data set
X=np.random.rand(1000,2)

# number of centroids
k=100
swaps = 1000

for i in range(5):
    randomSwap = rs(n_clusters=k)
    randomSwap.fit(X, swaps)

    km = KMeans(n_clusters=k, init='random').fit(X)
    
    # relative SSE improvement of random swap over kmeans
    imp = 1 - randomSwap.inertia_/km.inertia_
    print(f"SSE improvement over k-means: {imp:.2%}")
#Output
SSE improvement over k-means: 6.31%
SSE improvement over k-means: 6.89%
SSE improvement over k-means: 5.23%
SSE improvement over k-means: 5.11%
SSE improvement over k-means: 5.42%

## License
[MIT](https://choosealicense.com/licenses/mit/)
