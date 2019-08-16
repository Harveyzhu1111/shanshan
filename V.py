from operator import itemgetter


class Viterbi:
    def __init__(self, hmm, emission_probability, priors=None, constraint_length=10, candidate_states=None,
                 smallV=0.00000000001):
        """ Sets the stage for running the viterbi algorithm.
        hmm -- a map { state_id : [(next_state1, probability), (next_state2, probability)]}
        priors -- a map { state_id : probability } where the sum of probabilities=1
        emission_probability -- a function(state_id, observation) -> [0..1]
        constraint_length -- how many steps into the past to consider
        candidate_states -- a function f(obs) that returns a set of state ids given an observation
        """
        self.hmm = hmm
        self.emission_probability = emission_probability
        self.constraint_length = constraint_length
        if candidate_states:
            self.candidate_states = candidate_states
        else:
            self.candidate_states = lambda obs: self.hmm.keys()

        if not priors:
            self.priors = {}
            for state in self.hmm:
                self.priors[state] = 1.0 / len(self.hmm)
        else:
            self.priors = priors
        # set up the 'incoming' reverse index: for each state, what states contribute to its probability?
        self.incoming = {}
        for from_state in hmm:
            for to_state, probability in hmm[from_state]:
                if not to_state in self.incoming:
                    self.incoming[to_state] = {}
                self.incoming[to_state][from_state] = probability
        self.smallV = smallV

    def step(self, obs, V, path):
        """ performs viterbi matching. updates matrix V based on a single observation """

        # if no priors are specified, make them uniform 
        if V is None:
            V = dict(self.priors)
        new_V = {}
        new_path = {}
        # states that the current observation could in some way support
        state_eps = [(candi_state, self.emission_probability(candi_state, obs))
                     for candi_state in self.candidate_states(obs)]
        nonzero_eps = [state_ep for state_ep in state_eps if state_ep[1] > 0]
        # for each candidate state, calculate its maximum probability path
        for to_state, emission_probability in nonzero_eps:
            # some states may have millions of incoming edges
            # V: pre_state -> max prob value, this is satisfied when pre_state is unknown
            if len(self.incoming[to_state]) < len(V):
                nonzero_incoming = [(from_state, self.incoming[to_state][from_state])
                                    for from_state in self.incoming[to_state].keys()
                                    if from_state in V and V[from_state] > 0]
            else:
                # we only reserve from state that occurred in V
                nonzero_incoming = [(from_state, self.incoming[to_state][from_state]) for from_state in V
                                    if from_state in self.incoming[to_state]]

            # list of previous possible states and their probabilities (prob, state)
            # [(p_prev(from) * p_emission(obs, to) * p_trans(from, to), from_state)]
            from_probs = list(map(lambda from_state_transition_probability:
                                  (V[from_state_transition_probability[0]] * emission_probability *
                                   from_state_transition_probability[1],
                                   from_state_transition_probability[0]),
                                  nonzero_incoming))
            if len(from_probs) > 0:
                (max_prob, max_from) = max(from_probs, key=itemgetter(0))
                new_V[to_state] = max_prob
                # make sure we don't grow paths beyond the constraint length
                if max_from not in path:
                    path[max_from] = []
                if len(path[max_from]) == self.constraint_length:
                    path[max_from].pop(0)
                new_path[to_state] = path[max_from] + [to_state]
        new_V = self.normalize(new_V)
        small = list(filter(lambda x: new_V[x] < self.smallV, new_V))
        for state in small:
            del new_V[state]
            # jakob: this seems iffy, there should be no V for which there is no path
            if state in new_path:
                del new_path[state]
        return new_V, new_path

    def normalize(self, V):
        """ normalizes viterbi matrix V, so that the sum of probabilities add up to 1 """
        sum_prob = sum(V.values())
        # if we're stuck with no probability mass, return to priors instead
        if sum_prob == 0:
            return dict(self.priors)
        ret = dict([(state, V[state] / sum_prob) for state in V])
        return ret
