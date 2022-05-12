# Random Swap Clustering Algorithm
Random swap is a kmeans variant that does not stuck in local optima and finds the optimal partitions by removing a random centroid and adding a random centroid. This strategy work pretty well. You just need to book keep the successful swaps where distortion value reduced further. 

## Example 1: Random swap relative SSE improvements over kmeans using random dataset 

```python

from rs import rs
import numpy as np
from sklearn.cluster import KMeans

#random 2D data set
X=np.random.rand(1000,2)

# number of centroids
k=100
swaps = 1000

for i in range(5):
    randomSwap = rs(n_clusters=k).fit(X, swaps)

    km = KMeans(n_clusters=k, init='random').fit(X)
    
    # relative SSE improvement of random swap over kmeans
    imp = 1 - randomSwap.inertia_/km.inertia_
    print(f"SSE improvement over k-means: {imp:.2%}")
```
## Output
```bash
SSE improvement over k-means: 4.51%
SSE improvement over k-means: 3.10%
SSE improvement over k-means: 3.73%
SSE improvement over k-means: 1.93%
SSE improvement over k-means: 5.79%
```

## Example 2: Random swap relative SSE improvements over kmeans++ using random dataset
```python

from rs import rs
import numpy as np
from sklearn.cluster import KMeans

#random 2D data set
X=np.random.rand(1000,2)

# number of centroids
k=100
swaps = 20000

for i in range(5):
    randomSwap = rs(n_clusters=k).fit(X, swaps)

    km = KMeans(n_clusters=k, init='random').fit(X)
    
    # relative SSE improvement of random swap over kmeans
    imp = 1 - randomSwap.inertia_/km.inertia_
    print(f"SSE improvement over k-means: {imp:.2%}")
```
## Output
```bash
SSE improvement over k-means++: 1.69%
SSE improvement over k-means++: -1.98%
SSE improvement over k-means++: 0.42%
SSE improvement over k-means++: 1.18%
SSE improvement over k-means++: -1.18%
```


## Acknowledgements
Credit goes to Pasi Fränt, You may consider to read his paper for more understanding "Efciency of random swap clustering", also credit to the scikit-learn team for their excellent sklearn.cluster.KMeans class.

## License
[MIT](https://choosealicense.com/licenses/mit/)
