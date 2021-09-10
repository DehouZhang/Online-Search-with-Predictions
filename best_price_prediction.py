"""
Experiment: Predict about the best price
Generate result and plot both H_Oblivious and H_Aware algorithms
"""

from math import sqrt, log
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def plot_h_aware(result_list, eta_list_all, H_list, average_pure_online, average_best_price, save_path, x_label,
                 y_label, title):
    # plot the result of H_Aware algorithm
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list_all[i], result_list[i], label='$H_n$=%0.2f, $H_p$=%0.2f' % (H_list[i][0], H_list[i][1]))
    ax.axhline(average_pure_online, color='black', ls='dotted', label='Pure Online')
    ax.axhline(average_best_price, color='red', ls='dotted', label='Best Price')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def plot_h_oblivious(result, eta_list, r_list, pure_online, best_price, save_path, x_label, y_label, title):
    # plot the result of H_Oblivious algorithm
    fig, ax = plt.subplots()
    for i in range(len(result) - 2):
        ax.plot(eta_list, result[i], label='r=%0.2f' % (r_list[i]))
    ax.plot(eta_list, pure_online, color='black', ls='dotted', label='Pure Online')
    ax.plot(eta_list, best_price, color='red', ls='dotted', label='Best Price')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def save_to_csv_ho(payoff_list, eta_list, r_list, csv_path, pure_online, best_price):
    # save the result of H_oblivious algorithm into csv file
    myDict = {"eta": eta_list, "pure online": pure_online,
              "best price": best_price}
    for i in range(len(r_list)):
        myDict["payoff(r=%0.2f)" % r_list[i]] = payoff_list[i]
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def save_to_csv_ha(payoff_list, eta_list_all, H_list, csv_path, pure_online, best_price):
    # save the result of H_Aware algorithm into csv file
    myDict = {}
    longest_eta_list = eta_list_all[-1]

    for i in range(len(eta_list_all)):
        left_bound = eta_list_all[i][0]
        right_bound = eta_list_all[i][-1]
        left_index = longest_eta_list.index(left_bound)
        right_index = longest_eta_list.index(right_bound)
        eta_list_all[i] = [None] * left_index + eta_list_all[i] + [None] * (len(longest_eta_list) - right_index - 1)
        payoff_list[i] = [None] * left_index + payoff_list[i] + [None] * (len(longest_eta_list) - right_index - 1)

    for i in range(len(H_list)):
        myDict["eta(Hn=%0.2f, Hp=%0.2f)" % (H_list[i][0], H_list[i][1])] = eta_list_all[i]
        myDict["payoff(Hn=%0.2f, Hp=%0.2f)" % (H_list[i][0], H_list[i][1])] = payoff_list[i]
    myDict["pure online"] = [pure_online] * len(longest_eta_list)
    myDict["best price"] = [best_price] * len(longest_eta_list)
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def get_sample_hn_hp_bound(m, M, eta_step):
    Hn_bound_float = (M - m) / M  # the upper-bound of the value of negative error
    Hp_bound_float = (M - m) / m  # the upper-bound of the value of positive error
    # convert Hn and Hp to 2 decimal
    decimal = int(log(eta_step, 0.1))
    Hn_bound = round(Hn_bound_float, decimal)
    Hp_bound = round(Hp_bound_float, decimal)

    if Hn_bound > Hn_bound_float:
        Hn_bound = Hn_bound - eta_step

    if Hp_bound > Hp_bound_float:
        Hp_bound = Hp_bound - eta_step
    return Hn_bound, Hp_bound


def get_Hn_Hp_max(fileName, starting_days, whole_period, eta_step):
    Hn_max = 0
    Hp_max = 0
    for starting_day in starting_days:
        M, m = get_maxmin(fileName, starting_day, whole_period)
        Hn_bound, Hp_bound = get_sample_hn_hp_bound(m, M, eta_step)
        print(Hn_bound, Hp_bound)
        if Hn_bound > Hn_max:
            Hn_max = Hn_bound
        if Hp_bound > Hp_max:
            Hp_max = Hp_bound
    return Hn_max, Hp_max


def h_oblivious_full_range(fileName, starting_days, whole_period, trading_period, r_list, whole_eta, eta_step):
    result_list = list()

    for starting_day in starting_days:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)
        v_star = max(data)
        # set the range of eta of this data sample to be [-0.5,0.5]
        sample_result = list()  # contain payoff for this data sample

        Hn_bound, Hp_bound = get_sample_hn_hp_bound(m, M, eta_step)

        for r in r_list:
            # create the list of negative and positive value of eta
            eta_list_n = np.arange(0, Hn_bound + eta_step, eta_step).tolist()
            del (eta_list_n[0])

            eta_list_p = np.arange(0, Hp_bound + eta_step, eta_step).tolist()
            # create the list of negative and positive value of payoff
            payoff_list_n = list()
            payoff_list_p = list()

            # calculate payoff for each value of eta
            for eta_n in eta_list_n:
                payoff_list_n.append(h_oblivious_negative(data, v_star, eta_n, r))
            for eta_p in eta_list_p:
                payoff_list_p.append(h_oblivious_positive(data, v_star, eta_p, r))

            payoff_list = payoff_list_n[::-1] + payoff_list_p

            eta_list_n = [-x for x in eta_list_n]
            eta_list = eta_list_n[::-1] + eta_list_p
            eta_list = [round(x, int(log(eta_step, 0.1))) for x in eta_list]

            left_index = whole_eta.index(eta_list[0])
            right_index = whole_eta.index(eta_list[-1])

            payoff_list = [0] * left_index + payoff_list + [0] * (len(whole_eta) - right_index - 1)
            sample_result.append(payoff_list)

        pure_online_list = [0] * left_index + [pure_online] * len(eta_list) + [0] * (
                len(whole_eta) - right_index - 1)
        best_price_list = [0] * left_index + [v_star] * len(eta_list) + [0] * (
                len(whole_eta) - right_index - 1)
        sample_result.append(best_price_list)
        sample_result.append(pure_online_list)
        result_list.append(sample_result)

    result = list()
    for r in range(len(r_list) + 2):
        payoff_r = list()
        for sample_list in result_list:
            payoff_r.append(np.array(sample_list[r]))

        array_r = np.array(payoff_r, dtype=object)
        count_zero = array_r.T
        average_number = list()

        for i in count_zero:
            average_number.append(np.count_nonzero(i))
        result_r = list(array_r.sum(axis=0))

        for i in range(len(result_r)):
            result_r[i] = result_r[i] / average_number[i]
        result.append(result_r)

    return result


def h_oblivious_fixed_range(full_result, whole_eta, left_range, right_range):
    left_index = whole_eta.index(-left_range)
    right_index = whole_eta.index(right_range)
    fixed_range_result = list()
    for i in range(len(full_result)):
        fixed_range_result.append(full_result[i][left_index: right_index+1])

    fixed_range_eta = whole_eta[left_index: right_index+1]

    return fixed_range_eta, fixed_range_result


def h_aware(fileName, starting_days, quantity_of_data, whole_period, trading_period, Hn_Hp_list, eta_step):
    result_list = list()
    average_pure_online = 0
    average_best_price = 0

    # take the average payoff from all data sample
    for starting_day in starting_days:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)
        v_star = max(data)

        average_pure_online += pure_online  # sum the payoff of pure online for all data sample
        average_best_price += v_star  # sum the best price for all data sample

        sample_result = list()
        eta_list_all = list()

        # for different value of Hn and Hp, calculate eta list and payoff list
        for hn_hp in Hn_Hp_list:
            eta_list_n = np.arange(0, hn_hp[0] + eta_step, eta_step).tolist()
            eta_list_p = np.arange(0, hn_hp[1] + eta_step, eta_step).tolist()
            payoff_list_n = list()
            payoff_list_p = list()

            for eta_n in eta_list_n:
                payoff_list_n.append(h_aware_negative(data, v_star, hn_hp[0], hn_hp[1], eta_n, M, m))

            for eta_p in eta_list_p:
                payoff_list_p.append(h_aware_positive(data, v_star, hn_hp[0], hn_hp[1], eta_p, M, m))

            payoff_list = payoff_list_n[::-1] + payoff_list_p

            eta_list_n = [-x for x in eta_list_n]
            eta_list = eta_list_n[::-1] + eta_list_p

            payoff_array = np.array(payoff_list)
            eta_list_all.append(eta_list)

            sample_result.append(payoff_array)
            sample_array = np.array(sample_result, dtype=object)

        result_list.append(sample_array)
        result_array = np.array(result_list)

    result = list(result_array.mean(axis=0))
    result = [list(x) for x in result]
    average_pure_online = average_pure_online / quantity_of_data  # calculate the average payoff of pure online for all data samples
    average_best_price = average_best_price / quantity_of_data  # calculte the average best price for all data samples
    return eta_list_all, result, average_pure_online, average_best_price


def main():
    Full_range = False
    #data_set = "ETHUSD"
    #data_set = "BTCUSD"
    data_set = "CADJPY"
    #data_set = "EURUSD"

    fileName = "data/" + data_set + ".csv"  # choose the dataset
    whole_period = 200  # set the whole period to 250 days
    trading_period = 200  # set the trading period to 200 days
    eta_coefficient = 1000  # the coefficient determines how many data point for error
    quantity_of_data = 20
    fileName = "data/" + data_set + ".csv"  # choose the dataset
    starting_days = generate_uniform_data(fileName, 250, quantity_of_data)

    if data_set =="ETHUSD" or data_set =="BTCUSD":
        r_list = [0.5, 0.75, 1, 1.25, 1.5]
        eta_step = 0.001
        Hp_bound = 0.5
        Hn_bound = 0.5
        Hn_Hp_list = [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]
    else:
        r_list = [0.96, 0.98, 1, 1.02, 1.04]
        eta_step = 0.0001
        Hp_bound = 0.04
        Hn_bound = 0.04
        Hn_Hp_list = [(0.005, 0.005), (0.01, 0.01), (0.02, 0.02), (0.03, 0.03), (0.04, 0.04)]

    Hn_max, Hp_max = get_Hn_Hp_max(fileName, starting_days, whole_period, eta_step)

    whole_eta = np.arange(-Hn_max, Hp_max + eta_step, eta_step).tolist()
    whole_eta = [round(x, int(log(eta_step, 0.1))) for x in whole_eta]

    result = h_oblivious_full_range(fileName, starting_days, whole_period, trading_period, r_list, whole_eta, eta_step)
    if Full_range is False:
        whole_eta, result = h_oblivious_fixed_range(result, whole_eta, Hn_bound, Hp_bound)

    save_path_ho = "experiment_result/" + data_set + "/" + data_set + "_h_oblivious.png"  # the path to save the figure
    # plot the h_oblivious figure
    plot_h_oblivious(result, whole_eta, r_list, result[-1], result[-2], save_path=save_path_ho,
                     x_label="error $\eta$", y_label="Payoff", title="H-Oblivious")
    # the path to save the csv file
    csv_path_ho = "experiment_result/" + data_set + "/" + data_set + "_h_oblivious.csv"
    # save the result of h_oblivious algorithm to csv file
    save_to_csv_ho(result, whole_eta, r_list, csv_path_ho, result[-1], result[-2])

    eta_list_all, result, average_pure_online, average_best_price = h_aware(fileName, starting_days, quantity_of_data,
                                                                            whole_period, trading_period, Hn_Hp_list,
                                                                            eta_step)
    save_path_ha = "experiment_result/" + data_set + "/" + data_set + "_h_aware.png"

    # plot H_aware
    plot_h_aware(result, eta_list_all, Hn_Hp_list, average_pure_online, average_best_price, save_path_ha,
                 "error $\eta$", "Average Payoff", "H-Aware")
    # the path to save the result of h-aware

    csv_path_ha = "experiment_result/" + data_set + "/" + data_set + "_h_aware.csv"

    # save the result to csv file
    save_to_csv_ha(result, eta_list_all, Hn_Hp_list, csv_path_ha, average_pure_online, average_best_price)


if __name__ == '__main__':
    main()
