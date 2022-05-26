import numpy as np

# from graph import Graph

# original paper
# http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf

# video explaination
# https://www.youtube.com/watch?v=JGQe4kiPnrU


class Markov_Chain:
    """Class to model markov state in python.

    State tranistion incomplete.
    error threshold <= 0.001
    pi_{n+1} = (P^T) * pi_{n}, where P is the transition probability matrix.
    """

    def __init__(self: "Markov_Chain", web_net: dict = None) -> None:
        """Constructor

        Args:
            web_net (dict, optional): Initial or pre-defined web-network. Defaults to None.
        """
        if web_net:
            self.web_net = web_net
        else:
            self.web_net = {}

    def transition_matrix(self: "Markov_Chain") -> None:
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
        size = len(self.web_net)
        uniform_probability = 1 / size
        self.rank = np.array([uniform_probability for _ in range(size)], dtype=float)

    def simulate(
        self: "Markov_Chain",
        threshold: float = 0.001,
        max: int = 100,
    ) -> None:
        self.transition_matrix()
        self.rank_vector()
        print(self.rank)
        dist = 1
        while dist > threshold and max:
            old = self.rank
            self.rank = np.matmul(self.rank, self.P)
            dist = np.linalg.norm(old - self.rank)
            max -= 1
        print(max)
        print(self.rank)

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
    4: set([0, 1]),
}

www = Markov_Chain(web_network)

www.simulate()
