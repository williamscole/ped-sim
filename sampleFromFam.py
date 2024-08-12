import sys
import pandas as pd
import numpy as np
import itertools as it
from collections import Counter
import networkx as nx
import argparse

class Bins:
    def __init__(self, bins, binProbs):
        self.bins = [(bins[i], bins[i+1]) for i in range(len(bins)-1)]
        self.n_bins = len(self.bins)
        self.binProbs = binProbs

    def discretize(self, PropIBD):
        assert 0 <= PropIBD <= 1
        for index, (r1, r2) in enumerate(self.bins):
            if r1 < PropIBD <= r2:
                return index
            
    def randomBin(self):
        return np.random.choice(np.arange(self.n_bins), 1, p=self.binProbs)[0]


def extractId(pedSimId):
    tmp = pedSimId.split("_")[1].split("-")
    spouse = tmp[2][0]
    return [int(tmp[0][1:]), int(tmp[1][1:]), int(spouse=="i"), int(tmp[2][1:])]
        
class KING:
    def __init__(self, kingFile, bins, samplesList, samplesToAdd=None, maxPropIBD=0.1):

        self.kingDf = pd.read_csv(kingFile, delim_whitespace=True)

        self.kingDf["bin"] = self.kingDf["PropIBD"].apply(bins.discretize)
        
        self.orig_n = len(samplesList)
        self.n_samples = self.orig_n + len(samplesToAdd)
        self.simSamples = samplesToAdd
        samplesList += samplesToAdd


        n_bins = bins.n_bins

        binArray = np.empty((self.n_samples, n_bins), dtype=object)
        binCounts = np.zeros((self.n_samples, n_bins))
        kinArray = np.zeros((self.n_samples, len(samplesList)))
        for i in range(self.n_samples):
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
        self.samples = samplesList
        self.unrelated = np.where(kinArray==0)
        self.n_unrelated = len(self.unrelated[0])
        self.n_bins = n_bins
        self.maxPropIBD = maxPropIBD

        self.nth_founder = 0
        self.founderColumn = [[] for _ in self.samples]

        self.bins = bins

    def addChild(self, childId, parent1Idx, parent2Idx):
        assert childId in self.samples

        childIdx = self.samples.index(childId)

        # Step 1: get kinship to other relatives in the dataset
        k = self.kinArray[[parent1Idx, parent2Idx], :].sum(axis=0) / 2

        # Step 2: add row/column to kinship array
        self.kinArray[:, childIdx] = k; self.kinArray[childIdx, :] = k

        # Step 2: discretize and update the bin counts
        kDisc = [self.bins.discretize(i) for i in k]
        discCounts = Counter(kDisc)
        binCountRow = [0 for _ in range(self.n_bins)]
        for d, count in discCounts.items():
            binCountRow[d] = count
        self.binCounts[childIdx, :] = binCountRow

        # Step 3: add new to binArray
        for index, d in enumerate(kDisc):
            self.binArray[childIdx, d].append(index)

        return childIdx

    def randomCouple(self, checkAgainstFounders=None):
        randBin = -1
        while randBin < 0:
            tmp = self.bins.randomBin()

            if tmp == 0:
                i = -1
                while i < 0:
                    i = np.random.choice(np.arange(self.n_unrelated))
                    m, n = self.unrelated[0][i], self.unrelated[1][i]
                    if m == n:
                        i = -1
                    if m > self.orig_n or n > self.orig_n:
                        i = -1
                return self.unrelated[0][i], self.unrelated[1][i]

            randBin = tmp if self.binCounts[:, tmp].sum() > 0 else -1

            m = np.random.choice(np.where(self.binCounts[:,randBin]>0)[0], 1)[0]

            if checkAgainstFounders:
                for n in checkAgainstFounders:
                    if self.kinArray[m, n] > self.maxPropIBD:
                        randBin = -1
                        break
    
        return m, np.random.choice(self.binArray[m, randBin])

    def randomFounder(self, idx, checkAgainstFounders=None):
        # assert idx >= self.orig_n

        randBin = -1
        while randBin < 0:
            tmp = self.bins.randomBin()

            randBin = tmp if self.binCounts[idx, tmp].sum() > 0 else -1

            if randBin < 0:
                continue

            m = np.random.choice(self.binArray[idx, randBin], 1)[0]

            if m >= self.orig_n:
                randBin = -1

            if checkAgainstFounders and randBin > 0:
                for n in checkAgainstFounders:
                    if self.kinArray[m, n] > self.maxPropIBD:
                        randBin = -1
                        break
        
        return m
    
    def addFounders(self, idxList):
        for idx in idxList:
            self.founderColumn[idx].append(self.nth_founder)
            self.nth_founder += 2

    def printSamples(self):
        for sample, founders in zip(self.samples[:self.orig_n], self.founderColumn):
            print(f"{sample} {' '.join([str(i) for i in founders])}")



"""
General rule for choosing a spouse: (1) choose from distribution of their relatedness to you
and (2) make sure they are not overly related to other founders
"""
class FamGraph:
    def __init__(self, famDf):

        self.founders = famDf[famDf.apply(lambda x: x.parent1 == "0" and x.parent2 == "0", axis=1)]["id1"].values

        self.g = nx.DiGraph()
        self.g.add_nodes_from(famDf.id1.values)

        for node in self.g.nodes():
            self.g.nodes[node]["spouse"] = set()
            self.g.nodes[node]["founder"] = node in self.founders
            self.g.nodes[node]["idx"] = -1

        for p in ["parent1", "parent2"]:
            self.g.add_edges_from([i for i in famDf[[p, "id1"]].values if i[0] != "0"])

        self.couplesDf = famDf[~famDf["id1"].isin(self.founders)][["parent1", "parent2"]]

        self.couplesDf["founder1"] = self.couplesDf["parent1"].isin(self.founders)
        self.couplesDf["founder2"] = self.couplesDf["parent2"].isin(self.founders)

        for _, row in self.couplesDf.iterrows():
            self.g.nodes[row["parent1"]]["spouse"] |= {row["parent2"]}
            self.g.nodes[row["parent2"]]["spouse"] |= {row["parent1"]}

    def addSpouses(self, g, founders):
        for node in g.nodes():
            data = g.nodes[node]
            if data["founder"]: # Node is a founder
                for spouse in data["spouse"]: # Iterate through the spouses
                    idx = g.nodes[node]["idx"]
                    spouse_idx = g.nodes[spouse]["idx"]
                    if idx == -1 and spouse_idx == -1 and g.nodes[spouse]["founder"]:
                        n1, n2 = king.randomCouple(founders)
                        founders += [n1, n2]
                        g.nodes[node]["idx"] = n1; g.nodes[spouse]["idx"] = n2
                    if idx == -1 and spouse_idx > -1:
                        n1 = king.randomFounder(spouse_idx, founders)
                        g.nodes[node]["idx"] = n1
        return g, founders


    def findFounders(self, king):
        founders = []
        return_g = self.g.copy()

        return_g, founders = self.addSpouses(return_g, founders)

        # Get non-founder nodes
        needsIndex = [node for node, data in return_g.nodes(data=True) if not data["founder"]]
        while len(needsIndex) > 0:
            for node in needsIndex:
                parents = [i for i in return_g.predecessors(node)]
                id1x, id2x = [return_g.nodes[i]["idx"] for i in parents]
                # We have parents!
                if id1x > -1 and id2x > -1:
                    idx = king.addChild(node, id1x, id2x)
                    needsIndex.remove(node)
                    return_g.nodes[node]["idx"] = idx
                    return_g, founders = self.addSpouses(return_g, founders)
                    break
    


        couples = set()
        for node, data in return_g.nodes(data=True):
            for spouse in data["spouse"]:
                couples |= {tuple(sorted([data["idx"], return_g.nodes[spouse]["idx"]]))}

        tmp = pd.DataFrame(return_g.nodes(data=True))
        tmp = tmp[tmp[1].apply(lambda x: x["founder"])].reset_index(drop=True)
        tmp["idx"] = tmp[1].apply(lambda x: x.get("idx", -1))
        decomposedIds = pd.DataFrame(tmp[0].apply(extractId).values.tolist(),
                                    columns=["g", "b", "ind1", "ind2"])
        tmp = pd.concat([tmp, decomposedIds], axis=1)
        tmp["id1"] = tmp["idx"].apply(lambda x: king.samples[x])

        tmp = tmp.sort_values(["g", "b", "ind1", "ind2"])
        king.addFounders(tmp["idx"].values)                    


def readFam(famFile):
    famDf = pd.read_csv(famFile, delim_whitespace=True, dtype=str, header=None)

    return famDf[famDf[0]==famDf.at[0,0]].rename({1: "id1", 2: "parent1", 3: "parent2"}, axis=1)

def readVCF(vcfFile):
    with open(vcfFile, "r") as vcf:
        prevLine = None
        for line in vcf:
            if line[0] != "#":
                break
            prevLine = line
    return prevLine.split()[9:]

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fam', nargs='*', type=str, required=True,
                        help="PLINK-formatted .fam files. Can accept more than one, just separate by a space.") 
    
    parser.add_argument("--vcf", required=True, help="The VCF to be used as input. The program will open it to get the list of samples.")

    parser.add_argument("--king", required=True, help="KING .seg file, generated by running --ibdseg option of KING.")

    parser.add_argument("--bins", required=False, help="Define the PropIBD bins of the sampling step. Supplying the argument -1, 0, 0.05, 0.1, 1 would create the bins (-1, 0], (0, 0.05], (0.05, 0.1], (0.1, 1]",
                        nargs="*",
                        type=float,
                        default=[-1, 0, 0.04, 0.08, 0.12, 0.25, 1])
    
    parser.add_argument("--bin_probs", nargs="*", type=float,
                        default=[0.6, 0.25, 0.10, 0.03, 0.01, 0.02],
                        help="The corresponding probability of each bin in --bins.")
    
    parser.add_argument("--max_PropIBD", required=False, type=float, default=0.1,
                        help="Maximum PropIBD from KING allowed between founders in a family.")

    parser.add_argument("--n_simulations", required=True, type=int,
                        help="The number of simulations that you wish to perform on a given genealogy. Must provide this for each .fam file provided.",
                        nargs="*")
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()

    assert round(sum(args.binProbability), 8) == 1
    assert len(args.bin_probs) == (len(args.bins) - 1)
    assert len(args.n_simulations) == len(args.fam)

    bins = Bins(bins=args.bins, binProbs=args.bin_probs)

    samples = readVCF(args.vcf)
    founderColumn = [[] for _ in samples]
    nth_founder = 0


    for iter_num, famFile in zip(args.n_simulations, args.fam):

        famDf = readFam(famFile)

        samplesToAdd = famDf[famDf.apply(lambda x: x.parent1 != "0" and x.parent2 != "0", axis=1)]["id1"].values.tolist()

        famG = FamGraph(famDf)


        king = KING(kingFile=args.king,
                    bins=bins,
                    samplesList=samples,
                    samplesToAdd=samplesToAdd,
                    maxPropIBD=args.maxPropIBD)
        
        king.founderColumn = founderColumn
        king.nth_founder = nth_founder
        
        for _ in range(iter_num):
            famG.findFounders(king)

        founderColumn = king.founderColumn
        nth_founder = king.nth_founder

        