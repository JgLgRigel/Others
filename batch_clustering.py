# Author :      Guillermo López
# Date :        February 1st, 2021
# Description:  Useful to build clusters in large datasets. It uses a batch training methodology.

def batch_clustering(db, alpha=1e-1, b=1e2, k=3):
    n = len(db)
    batch = np.around((np.random.rand(n) * (b-1)) + 1)
    for i in np.arange(1,(b+1)):
        db2 = db[batch == i].copy()
        df = normalization(transformations(db2))
        if i == 1:
            # Forgy initialization
            centroids = df.loc[np.random.choice(range(df.shape[0]), replace=False, size=k)]
            centroids.reset_index(inplace=True,drop=True)
        dist = calculating_distance(df,centroids)
        cluster = dist.argmin(0)
        tmp = centroids.copy()
        for j in np.arange(0,k):
            tmp.loc[j] = df[cluster==j].mean()
        tmp = tmp - centroids
        centroids = centroids + (tmp * alpha)
        if i == b:
            dist = calculating_distance(df,centroids)
            total_distance = sum(dist.min(0))
            cluster = dist.argmin(0)
    return(total_distance, cluster, centroids)
