from os import stat
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt


class Markov_Chain:
    """Class to model markov chain in python."""

    def __init__(
        self: "Markov_Chain",
        web_net: dict = None,
        max: int = 100,
        threshold: float = 0.001,
        alpha: float = 0.001,
        random_restart: bool = True,
        show_change: bool = False,
    ) -> None:
        """Constructor

        Args:
            self (Markov_Chain): mandatory self object.
            web_net (dict, optional): initial or pre-defined web betwork. Defaults to None.
            max (int, optional): maximum number of itterations. Defaults to 100.
            threshold (float, optional): error threshold. Defaults to 0.001.
            alpha (float, optional): some constant for random restart. Defaults to 0.001.
            random_restart (bool, optional): to start randomly if user get struck at some web pages. Defaults to True.
            show_change (bool, optional): show how the rank vector change. Defaults to False.
        """
        if web_net:
            self.web_net = web_net
        else:
            self.web_net = {}

        self.max = max
        self.threshold = threshold
        self.alpha = alpha
        self.random_restart = random_restart
        self.show_change = show_change
        self.change = []

    def transition_matrix(self: "Markov_Chain") -> None:
        """Generate the transition matrix with the uniform probability
        to each outwards edge a node has.

        Args:
            self (Markov_Chain): _description_
        """
        self.P = np.zeros(
            shape=(
                len(self.web_net),
                len(self.web_net),
            ),
            dtype=float,
        )
        for _from, neighbours in self.web_net.items():
            p = 1 / len(neighbours)
            for _to in neighbours:
                self.P[_from][_to] = p

    def rank_vector(self: "Markov_Chain") -> None:
        """rank vector that contains the rank as the web-pages

        Args:
            self (Markov_Chain): mandatory self object.
        """
        self.N = len(self.web_net)
        uniform_probability = 1 / self.N
        self.rank = np.array([uniform_probability for _ in range(self.N)], dtype=float)

    def simulate(self: "Markov_Chain") -> None:
        """starts the simulation by performing repetative matrix product following the
        relation rank_vector_{n+1} = rank_vector_{n}*Transition_probablity_matrix

        Args:
            self (Markov_Chain): mandatory self object.
        """
        self.rank_vector()
        self.transition_matrix()
        dist = 1
        _max = self.max
        if self.random_restart:
            P = (1 - self.alpha) * self.P + (self.alpha / self.N) * np.ones(
                shape=(self.N, self.N), dtype=float
            )
        else:
            P = self.P

        if self.show_change:
            print("Ranking at the start of simulation:", self.rank)

        while dist > self.threshold and _max:
            self.change.append(self.rank)
            old = self.rank
            self.rank = np.matmul(self.rank, P)
            dist = np.linalg.norm(old - self.rank)
            _max -= 1
        self.max -= _max
        if self.show_change:
            print("Ranking at the end of simulation:", self.rank)
            print("It took", self.max, "steps to complete.")

    def top(self: "Markov_Chain", n: int = 5) -> Optional[int]:
        """Returns top n pages ranked by the algorithm.

        Args:
            self (Markov_Chain): mandatory self object.
            n (int, optional): number of top results to show. Defaults to 5.

        Returns:
            Optional[int]: list of ranked pages from top rank at 0 to low rank towards right.
        """

        self.simulate()
        if n > self.N:
            n = self.N
        self.rank_pages = [i for i in enumerate(self.rank)]
        self.rank_pages = sorted(self.rank_pages, key=lambda k: k[1], reverse=True)
        return [i[0] for i in self.rank_pages[:n]]

    def graph(self: "Markov_Chain") -> None:
        self.change = np.transpose(self.change)
        for i in range(self.N):
            plt.plot(list(range(self.max)), self.change[i], label=str(i))
        plt.xlabel("Number of steps")
        plt.ylabel("Rank at each step")
        plt.title("Markov Chain Algorithm")
        plt.legend()
        plt.show()

    def add_state(self: "Markov_Chain", _from: int, _to: int) -> None:
        """add state to the graph state

        Args:
            _to (int): intial node of the state
            _from (int): terminating node of the state
            transition_probablity (float): transition probability of the state
        """
        self.web_net[_from] = self.web_net.get(_from, set()) | set([_to])

    def __add__(self: "Markov_Chain", state: tuple[int, int]) -> "Markov_Chain":
        """special method to add states as tuples

        Args:
            state (tuple[int, int]): state tuple (_from, _to)

        Returns:
            Markov_State: class object
        """
        self.add_state(state[0], state[1])
        return self


web_network = {
    0: set([2, 3]),
    1: set([2]),
    2: set([0, 1, 3]),
    3: set([0, 4]),
    4: set([0]),
}
www = Markov_Chain(web_network, random_restart=False, show_change=False, threshold=0)
www += (4, 1)
# www += (5, 2)
# www += (0, 5)
# www += (6, 1)
# www += (2, 6)
# www += (1, 6)
n = 5
pg = www.top(n)
www.graph()
