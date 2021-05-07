"""Checks for projectivity. Also added to the UD parser in udify/dataset_readers"""
import sys
from collections import defaultdict
import numpy as np


class ProjectiveChecker:
    def __init__(self, heads):
        self.heads = heads
        self.the_heads = [(i + 1, h) for i, h in enumerate(heads)]
        self.children = defaultdict(list)
        for i, j in self.the_heads:
            self.children[j].append(i)

    def is_ancestor(self, ancestor, j):
        if j in self.children[ancestor]:
            return True
        for c in self.children[ancestor]:
            if self.is_ancestor(c, j):
                return True
        return False

    def is_ancestor_wrap(self, ancestor, j):
        if ancestor == j:
            return True
        return self.is_ancestor(ancestor, j)

    def check_projective(self):
        proj = True
        for i, j in self.the_heads:
            if j == 0:
                continue
            if np.abs(i - j) == 1:
                continue
            if j > i:
                for check in range(i, j):
                    if not self.is_ancestor_wrap(j, check):
                        proj = False
                        break
            if i > j:
                for check in range(j, i):
                    if not self.is_ancestor_wrap(j, check):
                        proj = False
                        break
        return proj


def TestProjectivity():
    """The only test in this whole project"""
    projective = ProjectiveChecker([2, 5, 5, 5, 0, 8, 8, 5, 5])
    nonprojective = ProjectiveChecker([2, 0, 4, 2, 0, 7, 4, 9, 7])

    assert projective.check_projective() == True
    assert nonprojective.check_projective() == False


if __name__ == "__main__":
    TestProjectivity()
    
    nonprojective = defaultdict()
    total = defaultdict()

    with open("PROJECTIVITY.txt", "r") as f:
        content = f.read()
        langs = content.split("\n\n")
        for lang in langs:
            data = lang.split("\n")
            nonprojective[data[0]] = sum([int(z) for z in data[1:]])

    with open("TOTAL.txt", "r") as f:
        for line in f.readlines():
            line = line.split()
            lang = line[0]
            total[lang] = sum([int(z) for z in line[1:]])

    normalized = defaultdict()
    n_sorted = sorted(nonprojective.items(), key=lambda x: x[1] / total[x[0]])
    for lan, _ in n_sorted:
        normalized[lan] = nonprojective[lan] / total[lan]
        print(lan, round(nonprojective[lan] / total[lan], 3))
    print(normalized)
