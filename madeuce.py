from math import factorial
import numpy as np

def comb(n,r):
    return factorial(n) / (factorial(r) * factorial(n-r))

dt = 10**-5

class Component:
    def __init__(self, prob, z) -> None:
        if prob == None and z == None:
            raise ValueError('Either probability or failure rate is required')

        self._z = z if callable(z) else lambda _: z

        if prob != None:
            self._prob = prob if callable(prob) else lambda _: prob
        else:
            self._prob = lambda t: np.exp(-self._z(t) * t)

    def R(self, t):
        return self._prob(t)

    def Z(self, t):
        return  self._z(t)


class SeriesStructure:
    def __init__(self, components) -> None:
        self._components = components

    def R(self, t):
        return np.prod([component.R(t) for component in self._components])

    def Z(self, t):
        return  sum([component.Z(t) for component in self._components])

class ParallelStructure:
    def __init__(self, components) -> None:
        self._components = components

    def R(self, t):
        return 1 - np.prod([1-component.R(t) for component in self._components])

    def Z(self, t):
        return  -((self.R(t+dt) - self.R(t)) / dt) / self.R(t)

class KNStructure:
    def __init__(self,k,n,z,prob) -> None:
        if prob == None and z == None:
            raise ValueError('Either probability or failure rate is required')

        self._k = k
        self._n = n
        self._z = z

        if prob != None:
            self._indiv_prob = prob if callable(prob) else lambda _: prob
        else:
            self._indiv_prob = lambda t: np.exp(-self._z * t)

    def R(self,t):
        return sum([comb(self._n,y) * (self._indiv_prob(t))**y * (1-self._indiv_prob(t))**(self._n-y) for y in range(self._k,self._n+1)])

    def Z(self, t):
        return  -((self.R(t+dt) - self.R(t)) / dt) / self.R(t)

