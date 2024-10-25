""" Utils to analyse a give network """

import numpy as np
import networkx as nx

from cr_constraints import *
from yield_mc_simulation import *


class FrequencyGraph(nx.DiGraph):

    def __init__(self, edges, frequencies, anharmonicity, f_drive=None, cz=False):
        # init the Digraph
        nx.DiGraph.__init__(self)

        # add the nodes to the graph:
        self.add_edges_from(edges)

        # add the frequencies. We cannot enumerate the nodes as add_edge changes the order
        for k in range(len(self.nodes)):
            self.nodes[k]['freq'] = frequencies[k]
            self.nodes[k]['a'] = anharmonicity[k]

        # add the driving
        for e, fd in zip(edges, f_drive):
            self.edges[e]['drive'] = fd

        self.cz = cz

        if not self.cz:
            self.check_cr()

    def plot(self, fig=None, ax=None):
        """ Draw the Graph """
        nx.draw(self, with_labels=True, font_weight='bold')

    def check_cr(self):
        """ check if the frequency drives are compatible with a CR drive type"""
        for edge in self.edges:

            # defined a boolean that check the CR compatibility
            b = self.edges[edge]['drive'] == self.nodes[edge[1]]['freq']

            if not b:
                print(f"The edge {edge} is not a CR edge "+str(self.edges[edge]['drive'])+" "+str(self.nodes[edge[1]]['freq']))
                return False

        print("The drive frequency are CR compatible")
        return True

    def check_constraint(self, thresholds: np.array, verbose=1, qutrit=False):
        """
        Check all the constraint on the graph of solution
        Args:
            thresholds (np.array): threshold associated with the differents constraints.
            verbose (int): if verbose > 0, will print where the constraints are not satisfied
        Returns:
            number_of_error (int) number of non satisfied constraints
        """
        constraints = [type_A1, type_A2, type3, type4, type5, type6, type7]
        if qutrit:
            constraints.extend([type1b, type2b, type2c,
                                type3b, type5b, type6b, type6c])
            print(thresholds)
            thresholds = np.concatenate(
                (thresholds, thresholds[np.ix_([0, 1, 1, 2, 4, 5, 5])]))
        res = 0

        for c, d in zip(constraints, thresholds):

            error = list(c(self, d).keys())
            if error != []:

                if verbose > 0:
                    print(f"{c.__name__:<8} (min= {d:.03f} GHz) on {error}")
                res += 1
        return res

    @property
    def freqs(self):
        """
        return the frequencies in an array ordered by the nodes
        """
        return np.array([self.nodes[n]['freq'] for n in range(len(self.nodes))],
                        dtype=np.float32)

    @property
    def anharmonicity(self):
        """
        return the anharmonicity in an array ordered by the nodes
        """
        return np.array([self.nodes[n]['a'] for n in range(len(self.nodes))],
                        dtype=np.float32)

    @property
    def oriented_edge_index(self):
        """ Return a list of the oriented edges as tuples """
        return [(i, j) for i, j in self.edges]

    @property
    def unoriented_edge_index(self):
        """ Return a list of the oriented edges as tuples """
        return [(i, j) for i, j in self.edges] + [(j, i) for i, j in self.edges]

    @property
    def cr_neighbhors(self):
        """ Return the tuple (i,j,k) for the neighbhors of the control and target j """
        idx_neighbhors = []
        for i, j in self.edges:
            control_neighbhors = list(nx.all_neighbors(self, i))
            control_neighbhors.remove(j)  # remnoving the target
            for k in control_neighbhors:
                idx_neighbhors.append((i, j, k))

            if self.cz:
                # neighbhors of the target
                control_neighbhors = list(nx.all_neighbors(self, j))
                control_neighbhors.remove(i)  # remnoving the target
                for k in control_neighbhors:
                    idx_neighbhors.append((i, j, k))

        return idx_neighbhors

    def get_collision(self,
                      thresholds: np.array,
                      sigma: float = 0.05,
                      Nsamples: int = 10000,
                      qutrit: bool = False,
                      cstr=None
                      ):
        """
        Calculate the yield of the FrequencyGraph for a given dispersion in frequency.
        Args:
            thresholds (np.array): array of the threshold for each constraint
            sigma (float): dispersion of frequency in GHz
            Nsample (int): number of sample for the MC yield simulation
        """

        # define the target frequencies and alpha
        target_frequencies = self.freqs
        target_alpha = self.anharmonicity

        # get a frequency distribution
        freqs_distribution = generate_random_sample(target_frequencies,
                                                    sigma=sigma,
                                                    Nsamples=Nsamples)
        alpha_distribution = generate_random_sample(target_alpha,
                                                    sigma=1e-5,
                                                    Nsamples=Nsamples)

        # construct the list of boolean function and index to apply it
        idx_list, expr_list, constraints = construct_constraint_function(
            self,
            freqs_distribution,
            alpha_distribution,
            thresholds,
            qutrit=qutrit,
            cstr=cstr)

        # Count the number of collisions

        c = []
        for idx, expr in zip(idx_list, expr_list):
            for i in idx:
                c.append(expr(*i))

        # counting the tiime where all the conditions are no validated
        return np.sum(~np.array(c), axis=0), c, idx_list, constraints
