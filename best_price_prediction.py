"""
Experiment: Predict about the best price
Generate result and plot both ORA and Robust algorithms
"""

from math import sqrt, ceil
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys


def load_data_set(file_name, starting_day, period):
    # load data with specific starting date and length
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=starting_day, nrows=period)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return raw_data


def get_maxmin(file_name, starting_day, period):
    # get the maximum and minimum price of the specific data sample
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=starting_day, nrows=period)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return max(raw_data), min(raw_data)


def get_size_data(file_name):
    # return the size of the dataset
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4])
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return len(raw_data)


def generate_uniform_data(file_name, period, number_of_data):
    # generate starting date uniformly in a specific range
    data_size = get_size_data(file_name)
    uniform = np.linspace(0, data_size - period, number_of_data, dtype=int)
    return uniform


def online(data, M, m):
    # pure online algorithm
    # return the payoff of the pure online algorithm
    reservation_price = sqrt(M * m)
    trade_price = first_greater_element(data, reservation_price)
    return trade_price


def first_greater_element(searching_list, element):
    # return the first element in a list which is greater than a value
    result = None
    for item in searching_list:
        if item >= (element - 0.0001):
            result = item
            break
    if result is None:
        result = searching_list[-1]
    return result


def h_aware_positive(data, v_star, Hn, Hp, eta, M, m):
    # the H_Aware algorithm with negative value of error
    # v_star: the best price
    # Hn: the upperbound of negative error
    # Hp: the upperbound of positive error
    # eta: the actual error
    # M: the upper-bound of the price
    # m: the lower-bound of the price
    # v: predicted price
    # v_prime: the reservation price
    # return the payoff
    v = v_star / (1 + eta)
    v_prime = sqrt(M * m)
    if (1 + Hn) / (1 - Hp) <= sqrt(M / m):
        v_prime = v * (1 - Hp)
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def h_aware_negative(data, v_star, Hn, Hp, eta, M, m):
    # the H_Aware algorithm with negative value of error
    # v_star: the best price
    # Hn: the upperbound of negative error
    # Hp: the upperbound of positive error
    # eta: the actual error
    # M: the upper-bound of the price
    # m: the lower-bound of the price
    # v: predicted price
    # v_prime: the reservation price
    # return the payoff
    v = v_star / (1 - eta)
    v_prime = sqrt(M * m)
    if (1 + Hn) / (1 - Hp) <= sqrt(M / m):
        v_prime = v * (1 - Hp)
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def h_oblivious_positive(data, v_star, eta, r):
    # the H_Oblivious algorithm with negative value of error
    # v: predicted price
    # v_prime: the reservation price
    # return the payoff
    v = v_star / (1 + eta)
    v_prime = r * v
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def h_oblivious_negative(data, v_star, eta, r):
    # the H_Oblivious algorithm with positive value of error
    # v: predicted price
    # v_prime: the reservation price
    # return the payoff
    v = v_star / (1 - eta)
    v_prime = r * v
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def plot_h_aware(result_list, eta_list, H_list, average_pure_online, average_best_price, save_path, x_label,
                 y_label, title):
    # plot the result of H_Aware algorithm
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list, result_list[i], label='H = %0.3f' % (H_list[i][0]))
    ax.axhline(average_pure_online, color='black', ls='dotted', label='ON*')
    ax.axhline(average_best_price, color='red', ls='dotted', label='M')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def plot_h_oblivious(result, eta_list, r_list, pure_online, best_price, save_path, x_label, y_label, title):
    # plot the result of H_Oblivious algorithm
    fig, ax = plt.subplots()
    for i in range(len(result)):
        ax.plot(eta_list, result[i], label='r = %0.2f' % (r_list[i]))
    ax.axhline(pure_online, color='black', ls='dotted', label='ON*')
    ax.axhline(best_price, color='red', ls='dotted', label='M')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def save_to_csv_ho(result, eta_list, r_list, csv_path, pure_online, best_price):
    # save the result of H_oblivious algorithm into csv file
    length = len(eta_list)
    myDict = {"eta": eta_list, "pure online": [pure_online] * length,
              "best price": [best_price] * length}
    for i in range(len(r_list)):
        myDict["payoff(r=%0.2f)" % r_list[i]] = result[i]
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def save_to_csv_ha(result, eta_list, H_list, csv_path, pure_online, best_price):
    length = len(eta_list)
    myDict = {"eta": eta_list, "pure online": [pure_online] * length,
              "best price": [best_price] * length}
    for i in range(len(H_list)):
        myDict["payoff(Hn=%0.3f, Hp=%0.3f)" % (H_list[i][0], H_list[i][1])] = result[i]
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def get_sample_hn_hp_bound(m, M):
    Hn_bound = (M - m) / M  # the upper-bound of the value of negative error
    Hp_bound = (M - m) / m  # the upper-bound of the value of positive error
    return Hn_bound, Hp_bound


def check_data_sample_range(fileName, starting_days, whole_period, Hn_bound, Hp_bound):
    valid_starting_days = list()
    for starting_day in starting_days:
        M, m = get_maxmin(fileName, starting_day, whole_period)
        Hn_range, Hp_range = get_sample_hn_hp_bound(m, M)
        if Hn_range >= Hn_bound and Hp_range >= Hp_bound:
            valid_starting_days.append(starting_day)
    return valid_starting_days


def h_oblivious(fileName, starting_days, whole_period, trading_period, r_list, Hn_bound, Hp_bound, eta_number):
    result_list = list()
    valid_starting_days = check_data_sample_range(fileName, starting_days, whole_period, Hn_bound, Hp_bound)
    quantity_of_data = len(valid_starting_days)

    pure_online_sum = 0
    best_price_sum = 0

    eta_list_n = np.linspace(0, Hn_bound, eta_number).tolist()
    eta_list_p = np.linspace(0, Hp_bound, eta_number).tolist()

    for starting_day in valid_starting_days:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)
        v_star = max(data)
        pure_online_sum += pure_online
        best_price_sum += v_star
        sample_payoff = list()
        for r in r_list:
            # create the list of negative and positive value of payoff
            payoff_list_n = list()
            payoff_list_p = list()

            # calculate payoff for each value of eta
            for eta_n in eta_list_n:
                payoff_list_n.append(h_oblivious_negative(data, v_star, eta_n, r))
            for eta_p in eta_list_p:
                payoff_list_p.append(h_oblivious_positive(data, v_star, eta_p, r))

            payoff_list = payoff_list_n[::-1] + payoff_list_p
            sample_payoff.append(payoff_list)
            sample_array = np.array(sample_payoff)

        result_list.append(sample_array)

    result_array = np.array(result_list)
    result = list(result_array.mean(axis=0))

    eta_list_n = [-x for x in eta_list_n]
    eta_list = eta_list_n[::-1] + eta_list_p

    average_pure_online = pure_online_sum / quantity_of_data
    average_best_price = best_price_sum / quantity_of_data

    return result, eta_list, average_pure_online, average_best_price


def h_aware(fileName, starting_days, whole_period, trading_period, Hn_Hp_list, Hn_bound, Hp_bound, eta_number):
    result_list = list()
    valid_starting_days = check_data_sample_range(fileName, starting_days, whole_period, Hn_bound, Hp_bound)
    quantity_of_data = len(valid_starting_days)

    pure_online_sum = 0
    best_price_sum = 0

    eta_list_n_all = np.linspace(0, Hn_bound, eta_number).tolist()
    eta_list_p_all = np.linspace(0, Hp_bound, eta_number).tolist()

    # take the average payoff from all data sample
    for starting_day in valid_starting_days:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)
        v_star = max(data)

        pure_online_sum += pure_online  # sum the payoff of pure online for all data sample
        best_price_sum += v_star  # sum the best price for all data sample

        sample_result = list()

        # for different value of Hn and Hp, calculate eta list and payoff list
        for hn_hp in Hn_Hp_list:
            # eta_list_n = np.linspace(0, hn_hp[0], eta_number).tolist()
            # eta_list_p = np.linspace(0, hn_hp[1], eta_number).tolist()
            left_bound = ceil(hn_hp[0] / Hn_bound * len(eta_list_n_all))
            right_bound = ceil(hn_hp[1] / Hp_bound * len(eta_list_p_all))
            eta_list_n = eta_list_n_all[0:left_bound]
            eta_list_p = eta_list_p_all[0:right_bound]
            payoff_list_n = list()
            payoff_list_p = list()

            for eta_n in eta_list_n:
                payoff_list_n.append(h_aware_negative(data, v_star, hn_hp[0], hn_hp[1], eta_n, M, m))

            for eta_p in eta_list_p:
                payoff_list_p.append(h_aware_positive(data, v_star, hn_hp[0], hn_hp[1], eta_p, M, m))

            payoff_list = payoff_list_n[::-1] + payoff_list_p
            # payoff_list = [None] * left_bound + payoff_list +[None] * (len(eta_list_n_all)+len(eta_list_p_all) - right_bound -1)
            payoff_array = np.array(payoff_list)
            sample_result.append(payoff_array)
            sample_array = np.array(sample_result, dtype=object)

        result_list.append(sample_array)
        result_array = np.array(result_list)

    eta_list_n_all = [-x for x in eta_list_n_all]
    eta_list = eta_list_n_all[::-1] + eta_list_p_all
    result = list(result_array.mean(axis=0))
    result = [list(x) for x in result]

    for i in range(len(result)):
        length = int(len(result[i]) / 2)
        result[i] = [None] * (eta_number - length) + result[i] + [None] * (eta_number - length)

    average_pure_online = pure_online_sum / quantity_of_data  # calculate the average payoff of pure online for all data samples
    average_best_price = best_price_sum / quantity_of_data  # calculte the average best price for all data samples
    return eta_list, result, average_pure_online, average_best_price


def main():
    # data_set = "ETHUSD"
    # data_set = "BTCUSD"
    # data_set = "CADJPY"
    # data_set = "EURUSD"

    data_set = sys.argv[1]
    fileName = "data/" + data_set + ".csv"  # choose the dataset
    whole_period = 200  # set the whole period to 250 days
    trading_period = 200  # set the trading period to 200 days
    quantity_of_data = 20
    starting_days = generate_uniform_data(fileName, 250, quantity_of_data)
    eta_number = 500
    if data_set == "ETHUSD" or data_set == "BTCUSD":
        r_list = [0.5, 0.75, 1, 1.25, 1.5]
        Hp_bound = 0.5
        Hn_bound = 0.5
        Hn_Hp_list = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]
    else:
        r_list = [0.96, 0.98, 1, 1.02, 1.04]
        Hp_bound = 0.04
        Hn_bound = 0.04
        Hn_Hp_list = [(0.005, 0.005), (0.01, 0.01), (0.02, 0.02), (0.03, 0.03), (0.04, 0.04)]

    result, eta_list, average_pure_online, average_best_price = h_oblivious(fileName, starting_days, whole_period,
                                                                            trading_period, r_list, Hn_bound, Hp_bound,
                                                                            eta_number)

    save_path_ho = "experiment_result/" + data_set + "/ORA.png"  # the path to save the figure
    # plot the h_oblivious figure
    plot_h_oblivious(result, eta_list, r_list, average_pure_online, average_best_price, save_path=save_path_ho,
                     x_label="error $\eta$", y_label="average profit", title="ORA")
    # the path to save the csv file
    csv_path_ho = "experiment_result/" + data_set + "/ORA.csv"
    # save the result of h_oblivious algorithm to csv file
    save_to_csv_ho(result, eta_list, r_list, csv_path_ho, average_pure_online, average_best_price)

    eta_list_all, result, average_pure_online, average_best_price = h_aware(fileName, starting_days,
                                                                            whole_period, trading_period, Hn_Hp_list,
                                                                            Hn_bound, Hp_bound,
                                                                            eta_number)
    save_path_ha = "experiment_result/" + data_set + "/Robust.png"

    # plot H_aware
    plot_h_aware(result, eta_list_all, Hn_Hp_list, average_pure_online, average_best_price, save_path_ha,
                 "error $\eta$", "average payoff", "Robust")
    # the path to save the result of h-aware

    csv_path_ha = "experiment_result/" + data_set + "/Robust.csv"

    # save the result to csv file
    save_to_csv_ha(result, eta_list_all, Hn_Hp_list, csv_path_ha, average_pure_online, average_best_price)


if __name__ == '__main__':
    main()
