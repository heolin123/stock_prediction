import pandas as pd
import numpy as np
from pomegranate import *

class DataWrapper(object):
    def __init__(self, path, separator=','):
        self.data = pd.read_csv(path, header=0, sep=separator)

    @staticmethod
    def get_data_flow(data):
        data_flow = []
        for x in range(1, len(data)):
            data_flow.append(round(data[x] - data[x-1], 2))
        return np.array(data_flow)


class HmmModelWrapper(object):
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        delta = 0.7
        dn5 = NormalDistribution( -5, delta )
        dn3 = NormalDistribution( -3, delta )
        dn1 = NormalDistribution( -1, delta )
        dp1 = NormalDistribution( 1, delta )
        dp3 = NormalDistribution( 3, delta )
        dp5 = NormalDistribution( 5, delta )

        sn5 = State(dn5, name="Negative 5" )
        sn3 = State(dn3, name="Negative 3" )
        sn1 = State(dn1, name="Negative 1" )
        sp1 = State(dp1, name="Positive 1" )
        sp3 = State(dp3, name="Positive 3" )
        sp5 = State(dp5, name="Positive 5" )

        states = [sn5, sn3, sn1, sp1, sp3, sp5]
        model = HiddenMarkovModel("Stock Predictor")
        model.add_states(states)

        probability = 1.0/float(len(states))

        for state in states:
            model.add_transition( model.start, state, probability )


        for state1 in states:
            for state2 in states:
                if state1 == state2:
                    model.add_transition( state1, state2, 0.5 )
                else:
                    model.add_transition( state1, state2, probability )


        model.bake()
        return model


    def train(self, data_flow, verbose=False):
        train_data = np.copy(data_flow)
        self.model.train(np.array([train_data]), verbose=verbose, distribution_inertia=0.3, edge_inertia=0.25)
        self.model.bake()

    def save_to_json(self, path):
        output_file = open(path, "w")
        output_file.write(self.model.to_json())
        output_file.close()
        #print "[LOG]: Model saved to: \"{0}\".".format(path)

    def load_from_json(self, path):
        self.model  = HiddenMarkovModel.from_json(open(path).read())

