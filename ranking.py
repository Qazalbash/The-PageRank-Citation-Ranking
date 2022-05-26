from typing import Optional

import numpy as np


class Markov_Chain:
    """Class to model markov chain in python."""

    def __init__(self: "Markov_Chain", web_net: dict = None) -> None:
        """Costructor

        Args:
            self (Markov_Chain): mandatory self object.
            web_net (dict, optional): initial or pre-defined web-network. Defaults to None.
        """
        if web_net:
            self.web_net = web_net
        else:
            self.web_net = {}

        self.threshold = 0.001
        self.max = 100
        self.alpha = 0.15
        self.random_restart = True
        self.show_change = False

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
            old = self.rank
            self.rank = np.matmul(self.rank, P)
            dist = np.linalg.norm(old - self.rank)
            _max -= 1
        if self.show_change:
            print("Ranking at the end of simulation:", self.rank)
            print("It took", max, "steps to complete.")

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
        self.web_net[state[0]] = self.web_net.get(state[0], set()) | set([state[1]])
        return self


web_network = {
    0: set([2, 3]),
    1: set([2]),
    2: set([0, 1, 3]),
    3: set([0, 4]),
    4: set([0]),
}
www = Markov_Chain(web_network)
www += (4, 1)
n = 5
pg = www.top(n)
print(pg)
