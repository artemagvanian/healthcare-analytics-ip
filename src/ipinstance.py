from dataclasses import dataclass

import numpy as np
from docplex.mp.model import Model


@dataclass(frozen=True)
class IPConfig:
    numTests: int  # number of tests
    numDiseases: int  # number of diseases
    costOfTest: np.ndarray  # [numTests] the cost of each test
    A: np.ndarray  # [numTests][numDiseases] 0/1 matrix if test is positive for disease


#  * File Format
#  * #Tests (i.e., n)
#  * #Diseases (i.e., m)
#  * Cost_1 Cost_2 . . . Cost_n
#  * A(1,1) A(1,2) . . . A(1, m)
#  * A(2,1) A(2,2) . . . A(2, m)
#  * . . . . . . . . . . . . . .
#  * A(n,1) A(n,2) . . . A(n, m)

def data_parse(filename: str):
    try:
        with open(filename, "r") as fl:
            numTests = int(fl.readline().strip())  # n
            numDiseases = int(fl.readline().strip())  # m

            costOfTest = np.array([float(i) for i in fl.readline().strip().split()])

            A = np.zeros((numTests, numDiseases), dtype=int)
            for i in range(0, numTests):
                A[i, :] = np.array([int(i) for i in fl.readline().strip().split()])
            return numTests, numDiseases, costOfTest, A
    except Exception as e:
        print(f"Error reading instance file. File format may be incorrect.{e}")
        exit(1)


class IPInstance:
    def __init__(self, filename: str) -> None:
        numT, numD, cst, A = data_parse(filename)
        self.numTests = numT
        self.numDiseases = numD
        self.costOfTest = cst
        self.A = A
        self.model = Model()  # CPLEX solver
        self.tests = []  # to hold variables

    def solve(self):
        for i in range(self.numTests):
            self.tests.append(self.model.binary_var(f"T{i}"))

        for i in range(self.numDiseases):
            for j in range(i + 1, self.numDiseases):
                equals = [self.tests[k] * self.A[k][i] == self.tests[k] * self.A[k][j] for k in range(self.numTests)]
                self.model.add_constraint(self.model.logical_not(self.model.logical_and(*equals)))

        cost = self.model.sum([self.costOfTest[i] * self.tests[i] for i in range(self.numTests)])
        self.model.minimize(cost)

        self.model.solve()

        return self.model.objective_value

    def __str__(self):
        out = ""
        out += f"Number of test: {self.numTests}\n"
        out += f"Number of diseases: {self.numDiseases}\n"
        cst_str = " ".join([str(i) for i in self.costOfTest])
        out += f"Cost of tests: {cst_str}\n"
        A_str = "\n".join([" ".join([str(j) for j in self.A[i]]) for i in range(0, self.A.shape[0])])
        out += f"A:\n{A_str}"
        return out
