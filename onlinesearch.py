from math import sqrt, exp, isnan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random


def load_data_set(file_name, starting_day, period):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=starting_day, nrows=period)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return raw_data


def get_maxmin(file_name, starting_day, period):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=starting_day, nrows=period)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return max(raw_data), min(raw_data)


def get_size_data(file_name):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4])
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return len(raw_data)


def generate_uniform_data(file_name, period, number_of_data):
    data_size = get_size_data(file_name)
    #random_list = random.sample(range(0, data_size - period), number_of_data)
    uniform = np.linspace(0, data_size - period, number_of_data, dtype=int)
    return uniform


def online(data, M, m):
    reservation_price = sqrt(M * m)
    trade_price = first_greater_element(data, reservation_price)
    return trade_price


def first_greater_element(searching_list, element):
    result = None
    for item in searching_list:
        if item >= (element-0.0001):
            result = item
            break
    if result is None:
        result = searching_list[-1]
    return result


def h_aware_negative(data, v_star, Hn, Hp, eta, M, m):
    v = v_star / (1 + eta)
    v_prime = sqrt(M * m)
    if (1 + Hn) / (1 - Hp) <= sqrt(M / m):
        v_prime = v * (1 - Hp)
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def h_aware_positive(data, v_star, Hn, Hp, eta, M, m):
    v = v_star / (1 - eta)
    v_prime = sqrt(M * m)
    if (1 + Hn) / (1 - Hp) <= sqrt(M / m):
        v_prime = v * (1 - Hp)
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def h_oblivious_negative(data, v_star, eta, r):
    v = v_star / (1 + eta)
    v_prime = r * v
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def h_oblivious_positive(data, v_star, eta, r):
    v = v_star / (1 - eta)
    v_prime = r * v
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def plot_h_aware(result_list, eta_list_all, H_list, average_pure_online, average_best_price, save_path, x_label, y_label, title):
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


def plot_h_oblivious(result_list, eta_list_all, r_list, pure_online, best_price, save_path, x_label, y_label, title):
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list_all[i], result_list[i], label='r=%0.2f' % (r_list[i]))
    ax.axhline(pure_online, color='black', ls='dotted', label='Pure Online')
    ax.axhline(best_price, color='red', ls='dotted', label='Best Price')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def save_to_csv_ho(payoff_list, eta_list, r_list, csv_path, pure_online, best_price):
    myDict = {"eta": eta_list[-1], "pure online": [pure_online] * len(eta_list[-1]),
              "best price": [best_price] * len(eta_list[-1])}
    for i in range(len(r_list)):
        myDict["payoff(r=%0.2f)" % r_list[i]] = payoff_list[i]
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def save_to_csv_ha(payoff_list, eta_list, H_list, csv_path, pure_online, best_price):
    myDict = {}
    for i in range(len(H_list)):
        myDict["eta(Hn=%0.2f, Hp=%0.2f)" % (H_list[i][0], H_list[i][1])] = eta_list[i]
        myDict["payoff(Hn=%0.2f, Hp=%0.2f)" % (H_list[i][0], H_list[i][1])] = payoff_list[i]
    myDict["pure online"] = [pure_online] * len(eta_list[-1])
    myDict["best price"] = [best_price] * len(eta_list[-1])
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def main():
    fileName = "data\CADJPY.csv"
    whole_period = 250
    trading_period = 200
    eta_coefficient = 1000
    quantity_of_data = 20

    ho_starting_day = 590
    data = load_data_set(fileName, ho_starting_day, trading_period)
    M, m = get_maxmin(fileName, ho_starting_day, whole_period)
    v_star = max(data)
    pure_online = online(data, M, m)

    r_list = [0.5, 0.75, 1, 1.5]
    Hn_bound = (M - m) / m
    Hp_bound = (M - m) / M


    eta_list_all_ho=list()
    payoff_list_all_ho = list()
    for r in r_list:
        eta_list_n = np.linspace(0, Hn_bound, int(Hn_bound * eta_coefficient)).tolist()
        eta_list_p = np.linspace(0, Hp_bound, int(Hn_bound * eta_coefficient)).tolist()
        payoff_list_n = list()
        payoff_list_p = list()

        for eta_n in eta_list_n:
            payoff_list_n.append(h_oblivious_negative(data, v_star, eta_n, r))
        for eta_p in eta_list_p:
            payoff_list_p.append(h_oblivious_positive(data, v_star, eta_p, r))

        payoff_list = payoff_list_n[::-1] + payoff_list_p

        eta_list_n = [-x for x in eta_list_n]
        eta_list = eta_list_n[::-1] + eta_list_p

        eta_list_all_ho.append(eta_list)
        payoff_list_all_ho.append(payoff_list)

    save_path_ho = "predict_price_fig/" + "cadjpy_h_oblivious.png"
    plot_h_oblivious(payoff_list_all_ho,eta_list_all_ho,r_list,pure_online,v_star,save_path=save_path_ho,x_label="error $\eta$",y_label="Payoff",title="H-Oblivious")
    csv_path_ho = "experiment_result/CADJPY/" + "H_oblivious.csv"
    save_to_csv_ho(payoff_list_all_ho, eta_list_all_ho, r_list, csv_path_ho,pure_online,v_star)

    # H aware
    uniform_list = generate_uniform_data(fileName, whole_period, quantity_of_data)
    result_list = list()
    average_pure_online = 0
    average_best_price = 0
    Hn_Hp_list = [(0.05, 0.05), (0.1, 0.1), (0.2, 0.2), (0.2, 0.3), (0.3, 0.3), (0.4, 0.4),(0.5,0.5)]

    for starting_day in uniform_list:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)
        v_star = max(data)

        average_pure_online += pure_online
        average_best_price += v_star

        sample_result = list()
        eta_list_all = list()

        for hn_hp in Hn_Hp_list:
            eta_list_n = np.linspace(0, hn_hp[0], int(hn_hp[0]*eta_coefficient)).tolist()
            eta_list_p = np.linspace(0, hn_hp[1], int(hn_hp[1]*eta_coefficient)).tolist()
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
    average_pure_online = average_pure_online / quantity_of_data
    average_best_price = average_best_price / quantity_of_data

    # draw
    save_path_ha = "predict_price_fig/" + "cadjpy_h_aware.png"
    plot_h_aware(result, eta_list_all, Hn_Hp_list, average_pure_online, average_best_price, save_path_ha, "error $\eta$", "Average Payoff", "H-Aware")
    csv_path_ha = "experiment_result/CADJPY/" + "H_aware.csv"
    save_to_csv_ha(result, eta_list_all, Hn_Hp_list, csv_path_ha,average_pure_online,average_best_price)

if __name__ == '__main__':
    main()
