import sys
import pandas as pd
import numpy as np
from collections import Counter
import networkx as nx

BINS = [(0, 0),
        (0, 0.02),
        (0.02, 0.06),
        (0.08, 0.10),
        (0.10, 0.12),
        (0.12, 0.15),
        (0.15, 1)]

def discretize(PropIBD):
    for index, (r1, r2) in enumerate(BINS):
        if r1 < PropIBD <= r2:
            return index

class KING:
    def __init__(self, kingFile, binProbs, samplesList=None, maxPropIBD=0.1):

        self.kingDf = pd.read_csv(kingFile, delim_whitespace=True)

        self.kingDf["bin"] = self.kingDf["PropIBD"].apply(discretize)

        if not samplesList:
            samplesList = list(set(self.kingDf.ID1) | set(self.kingDf.ID2))

        n_bins = len(binProbs)

        binArray = np.empty((len(samplesList), n_bins), dtype=object)
        binCounts = np.zeros((len(samplesList), n_bins))
        kinArray = np.zeros((len(samplesList), len(samplesList)))
        for i in range(len(samplesList)):
            for j in range(n_bins):
                binArray[i, j] = []

        for _, row in self.kingDf.iterrows():
            id1, id2, pairBin = row["ID1"], row["ID2"], row["bin"]
            idx1, idx2 = samplesList.index(id1), samplesList.index(id2)
            binArray[idx1, pairBin].append(idx2)
            binArray[idx2, pairBin].append(idx1)
            binCounts[idx1, pairBin] += 1
            binCounts[idx2, pairBin] += 1
            kinArray[idx1, idx2] = row["PropIBD"]
            kinArray[idx2, idx1] = row["PropIBD"]

        self.binArray = binArray # n_samples x n_bins
        self.kinArray = kinArray # n_samples x n_samples
        self.binCounts = binCounts # n_samples x n_bins
        self.binProbs = binProbs
        self.samples = samplesList
        self.n_bins = n_bins
        self.n_samples = len(samplesList)
        self.orig_n = self.n_samples

    def addChild(self, childId, parent1Idx, parent2Idx):

        # Step 1: get kinship to other relatives in the dataset
        k = self.kinArray[[parent1Idx, parent2Idx], :].sum(axis=0) / 2

        # Step 2: add row/column to kinship array
        m = self.n_samples
        tmp = np.zeros((m+1, m+1), dtype=self.kinArray.dtype)
        tmp[:m, :m] = self.kinArray
        tmp[:, m] = np.append(k, 0); tmp[m, :] = np.append(k, 0)
        self.kinArray = tmp

        # Step 2: discretize and update the bin counts
        kDisc = [discretize(i) for i in k]
        discCounts = Counter(kDisc)
        binCountRow = [0 for _ in range(self.n_bins)]
        for d, count in discCounts.items():
            binCountRow[d] = count
        self.binCounts = np.append(self.binCounts, np.array([binCountRow]), axis=0)

        # Step 3: add new to binArray
        row = [list() for _ in range(self.n_bins)]
        for index, d in enumerate(kDisc):
            row[d].append(index)
        self.binArray = np.append(self.binArray, np.array([row]), axis=0)

        self.samples.append(childId)
        self.n_samples += 1


    def randomCouple(self):
        randBin = -1
        while randBin < 0:
            tmp = np.random.choice(np.arange(self.n_bins), 1, p=self.binProbs)[0]

            randBin = tmp if self.binCounts[:, tmp].sum() > 0 else -1
            
            m = np.random.choice(np.where(self.binCounts[:,randBin]>0)[0], 1)[0]
    
        return m, np.random.choice(self.binArray[m, randBin])

    def randomFounder(self, idx):
        assert idx >= self.orig_n

        randBin = -1
        while randBin < 0:
            tmp = np.random.choice(np.arange(self.n_bins), 1, p=self.binProbs)[0]

            randBin = tmp if self.binCounts[idx, tmp].sum() > 0 else -1

            if randBin < 0:
                continue

            m = np.random.choice(self.binArray[idx, randBin], 1)[0]

            if m >= self.orig_n:
                randBin = -1
        
        return m













    def accept_couple(self):
        return True

    def get_random(self, n):
        return np.random.choice(np.arange(10000), 2)

"""
General rule for choosing a spouse: (1) choose from distribution of their relatedness to you
and (2) make sure they are not overly related to other founders
"""
class FamGraph:
    def __init__(self, famDf):

        self.g = nx.DiGraph()

        for p in ["parent1", "parent2"]:
            self.g.add_edges_from(famDf[[p, "id1"]].values)

        self.founders = famDf[famDf.apply(lambda x: x.parent1 == "0" and x.parent2 == "0", axis=1)]["id1"].values

        self.couplesDf = famDf[~famDf["id1"].isin(self.founders)][["parent1", "parent2"]]

        self.couplesDf["founder1"] = self.couplesDf["parent1"].isin(self.founders)
        self.couplesDf["founder2"] = self.couplesDf["parent2"].isin(self.founders)

        

def readFam(famFile):
    famDf = pd.read_csv(famFile, delim_whitespace=True, dtype=str, header=None)

    return famDf[famDf[0]==famDf.at[0,0]].rename({1: "id1", 2: "parent1", 3: "parent2"}, axis=1)



if __name__ == "__main__":

    famFile = sys.argv[1]

    famDf = readFam(famFile)

    famG = FamGraph(famDf)

    binProbs = [0.6, 0.2, 0.1, 0.04, 0.03, 0.02, 0.01]

    print(sum(binProbs))

    king = KING("../../PedigreeSimulations/QC_manuscript_sims/king.seg",
                binProbs)

    king.addChild("A", 40, 90)

    m = king.randomFounder(6049)

    import pdb; pdb.set_trace()



