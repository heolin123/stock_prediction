#!/usr/bin/env python2
import pandas as pd
import numpy as np
from pomegranate import *
import matplotlib.pyplot as plt
from hmm_model import *
from sklearn import linear_model
import sys

def normalize_data(data):
    n = len(data)
    X = np.ones(shape=(n, 2))
    X[:,0] = np.arange(n)

    Y = data.reshape(n, 1)
    lin_reg = linear_model.LinearRegression()
    lin_reg.fit(X, Y)

    coefs = np.array([lin_reg.coef_[0][0], lin_reg.intercept_[0]])

    offset_data = np.dot(X, coefs)
    data_normalized = data - offset_data
    return data_normalized, offset_data


def train(hmm_model, data_flow, fake=False):
    first = hmm_model.model.log_probability(data_flow)
    if not fake:
        hmm_model.train(data_flow)
    second = hmm_model.model.log_probability(data_flow)
    return first, second

def prepare_test_data(path, column="Open"):
    data_wrapper = DataWrapper(path, ',')
    data = np.array(data_wrapper.data["Open"])
    data_normalized, offset_data = normalize_data(data)

    data_flow = DataWrapper.get_data_flow(data_normalized)
    max_value = 5.0
    normalization_factor = 5.0/np.max(np.abs(data_flow))
    data_flow_normalized = data_flow * normalization_factor

    return data_flow_normalized


def main():
    symbol = sys.argv[1]
    hmm_model = HmmModelWrapper()
    hmm_model.load_from_json("model.json")
    path = "./data/{}.csv".format(symbol)
    data_flow_normalized = prepare_test_data(path)

    train_data_flow = np.array(data_flow_normalized)

    hmm_model = HmmModelWrapper()
    try:
        first, last = train(hmm_model, train_data_flow, False)
    except ZeroDivisionError:
        hmm_model.load_from_json("model.json")
    if str(last) == "nan":
        hmm_model.load_from_json("model.json")
    else:
        print symbol, first, last
    hmm_model.save_to_json("model.json")


if __name__ == "__main__":
    main()
