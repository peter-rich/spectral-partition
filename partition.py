import sys
import numpy as np
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from objective import objective

def main():
    filename = sys.argv[1]
    E = []
    adjacency = []
    nofVertices = 0
    nofEdges = 0
    k = 2

    nofVertices = 200
    adjacency = np.zeros((nofVertices, nofVertices), dtype=int)
        
    if not filename:
        print("no filename")
        return
    try:
        #data = pd.ExcelFile("graphs_processed/"+filename)
        
        f = open("graphs_processed/"+filename, "r")
        #print(data[0])
        for line in f:
            meta = line.split(" ")
            u = int(meta[0])
            v = int(meta[1])
            u, v = map(lambda x: int(x), line.split(" "))
            adjacency[u][v] = 1
            adjacency[v][u] = 1
            E.append((u, v))
    except:
        print(f"no such file {filename}")
        return
    import pymetis

    # Calculate the pmetis partition
    '''
   adj = []
    for i in range(0, len(adjacency)):
        tmp = []
        for j in range(0, len(adjacency[i])):
            if (adjacency[i][j] == 1):
                tmp.append(adjacency[i][j])
        adj.append(np.array(tmp))
    n_cuts, membership = pymetis.part_graph(2, adjacency=adj) 

    #import csv
    #file = open('res_bar.csv', 'w', newline ='')
    with file:
        # using csv.writer method from CSV package
        write = csv.writer(file)
        mem = []
        for kk in range(0, len(membership)):
            mem.append([membership[kk]])
        write.writerows(mem)

    print("####ALL")
    print(n_cuts, membership) 
    '''
    # Calculate degrees
    degree = np.array(list(map(lambda x: np.sum(x), adjacency)), dtype=float)
    D = np.diag(np.sqrt(1 / degree))
    DD = np.diag(degree)

    # Compute clusterings
    #kmeans_fiedler
    #spec = fiedler(adjacency, D, k)
    #spec = spectral(adjacency, D, k)
    spec = ogSpectral(adjacency, DD, k)

    print("####Break here####")
    # Calculate scores
    # Kmeans partition
    # print("Kmeans Clustering:", kmeans_fiedler)
    # kmeans_fiedler_score = objective(kmeans_fiedler, E, k)
    
    # Spectral partition with normalized Laplacian matrix
    print("Spectral vector:", spec)
    len_1, len_2 = 0, 0
    adj1 = np.zeros((nofVertices, nofVertices), dtype=int)
    adj2 = np.zeros((nofVertices, nofVertices), dtype=int)
    for i in range (0, len(spec)):
        if spec[i] == 0:
            len_1 += 1
            for j in range(0, nofVertices):
                if adjacency[i][j] == 1 and spec[adjacency[i][j]] == 0:
                    adj1[i][j] = 1
        else:
            len_2 += 1
            for j in range(0, nofVertices):
                if adjacency[i][j] == 1 and spec[adjacency[i][j]] == 1:
                    adj2[i][j] = 1
    print("Length of 1:", len_1)
    print("Length of 2:", len_2) 

    spec_score = objective(spec, E, k)
    # Unnormalized Laplacian matrix without the first eigenvector partition.
    #print("unnormalized spectral vector:", og_spec)
    #og_spec_score = objective(og_spec, E, k)

    #print("kmeans fiedler:", kmeans_fiedler_score)
    print("spec:", spec_score)
    print("Adj1:", adj1)
    print("Adj2:", adj2)
    #print("og spec:", og_spec_score)

    #writeRes("fiedler", filename, nofVertices, nofEdges, k, kmeans_fiedler)
    #writeRes("normalized-spectral", filename, nofVertices, nofEdges, k, spec)
    #writeRes("without-first-eigvec", filename, nofVertices, nofEdges, k, og_spec)

# Spectral clustering with Fiedler vector
def fiedler(A, D, k):
    UL = D - A
    # Get eigenvalues v and eigenvectors w
    v, w = np.linalg.eig(UL)
    # Sort eigenvalues
    idx = v.argsort()[::1]
    x2 = w[idx[:2][1]].T
    fiedler = x2.reshape(-1,1)
    res = KMeans(n_clusters=k).fit_predict(fiedler)
    return res


# Basic normalized spectral clustering
def spectral(A, D, k):
    L = np.identity(A.shape[0]) - D @ A @ D # @ is matrix multiplication operation
    V, eig = eigsh(L, 2)
    
    U = eig
    U_rowsums = U.sum(axis=1)
    U = U / U_rowsums[:, np.newaxis]

    res = KMeans(n_clusters=k).fit_predict(U)
    return res

# Spectral clustering without the first eigenvector
def ogSpectral(A, D, k):
    res = []
    L = D - A
    w, eig = eigsh(L, 2, which="SA")
    #print("eig", eig)
    # Drop the first eigenvector
    U = eig.T[1::].T
    for i in range(0, len(U)):
        if eig[i][1] < 0:
            res.append(0)
        else:
            res.append(1)
    return res

def writeRes(alg, name, nofV, nofE, k, clustering):
    try:
        f = open(f"../results/{alg}/{name}", "w")
        f.write(f"# {name} {nofV} {nofE} {k}\n")
        for v, c in enumerate(clustering, start=1):
            f.write(f"{v} {c+1}\n")
        f.close()
    except Exception as e:
        print(e)
        print("write failed")

main()
