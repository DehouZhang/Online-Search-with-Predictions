"""
Experiment: Query Based prediction
Generate result and plot the RLIS algorithm and RLIS_count
"""

from math import sqrt, ceil
import random
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


def generate_query(k, M, m, v_star):
    # generate correct query
    r = (M / m) ** (1 / k)
    target = m * r
    query = list()
    for i in range(k):
        if target < v_star:
            query.append(0)
            target = target * r
        else:
            query.append(1)
            target = target * r
    return query  # return the correct query


def generate_wrong_query(query, wrong_number):
    # generate wrong query
    k = len(query)
    # randomly flip bit to generate wrong query
    wrong_position_list = random.sample(range(0, k), wrong_number)  # random from 0 to k
    # 0 means "NO" and 1 means "YES"
    for i in wrong_position_list:
        if query[i] == 0:
            query[i] = 1
        else:
            query[i] = 0
    return query  # return the wrong query


def check_no_answers(query, position, H):
    # check if the number of "NO" after the current "Yes" is greater than p
    k = len(query)
    p = ceil(k * H)
    count = 0
    for i in range(position + 1, k):
        if query[i] == 0:
            count = count + 1
    if count > p:
        return True
    else:
        return False


def check_yes_answers(query, position, H):
    # check if the number of "Yes" before the current "NO" is greater than p
    k = len(query)
    p = ceil(k * H)
    count = 0
    for i in range(0, position):
        if query[i] == 1:
            count = count + 1
    if count > p:
        return True
    else:
        return False


def correct_wrong_answer(query, H):
    # correct the wrong answers in the wrong query
    k = len(query)
    # randomly pick the order to check all queries
    position_list = random.sample(range(0, k), k)
    for i in position_list:
        if i == 0:
            if query[i] == 1:
                if check_no_answers(query, i, H):
                    query[i] = 0
        elif i == k - 1:
            if query[i] == 0:
                if check_yes_answers(query, i, H):
                    query[i] = 1
        else:
            if query[i] == 0:
                if check_yes_answers(query, i, H):
                    query[i] = 1
            else:
                if check_no_answers(query, i, H):
                    query[i] = 0
    return query  # return the corrected query


def find_first_yes(query):
    # find the position of the first "Yes" query
    for i in range(len(query)):
        if query[i] == 1:
            return i


def find_last_no(query):
    # find the position of the last "No" query
    for i in reversed(range(len(query))):
        if query[i] == 0:
            return i


def count_alpha(query, i, j):
    # count the number of "No" between position i and j
    alpha = 0
    for k in range(i, j):
        if query[k] == 0:
            alpha = alpha + 1
    return alpha


def check_query_all_no(query):
    # check if query is full of "No" query
    count = 0
    for i in range(len(query)):
        if query[i] == 1:
            count = count + 1
    # if full of "No", set the last query to "YES"
    if count == 0:
        query[-1] = 1
        return True
    else:
        return False


def get_m_prime_RLIS(corrected_query, m, k, H, r):
    if check_query_all_no(corrected_query):
        m_prime = m * (r ** (len(corrected_query)-k+1))
    else:
        i = find_first_yes(corrected_query)
        j = find_last_no(corrected_query)
        if i > j:
            i_prime = j  # calculate i_prime and j_prime
        else:
            alpha = count_alpha(corrected_query, i, j)
            p = ceil(k * H)
            i_prime = i - p + alpha

            # if i_prime is out of range or j_prime is out of range
            if i_prime < 0:
                i_prime = 0
        m_prime = m * (r ** i_prime)
    return m_prime


def RLIS(fileName, starting_days, whole_period, quantity_of_data, trading_period, H_list, k, average_coefficient):
    pure_online_sum = 0
    best_price_sum = 0
    result_list_mean = list()
    pure_online_list = list()

    best_price_list = list()
    result_count = list()
    for starting_day in starting_days:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)  # the payoff of pure online algorithm
        v_star = max(data)  # the best price

        pure_online_list.append(pure_online)
        best_price_list.append(v_star)

        r = (M / m) ** (1 / k)  # calculate r
        pure_online_sum += pure_online
        best_price_sum += v_star
        wrong_bit_list_all = list()
        sample_result_array = list()
        sample_count_array = list()
        for H in H_list:
            price_list = list()
            count_list = list()
            wrong_bit_list = list(range(ceil(H * k) + 1))
            for wrong_bit in wrong_bit_list:
                average_trading_price = 0
                count = 0
                for l in range(average_coefficient):
                    query = generate_query(k, M, m, v_star)  # generate the correct query
                    if check_query_all_no(query):  # if query is full of "YES", flip the last point to "YES"
                        query[-1] = 1
                    wrong_query = generate_wrong_query(query, wrong_bit)  # generate wrong query
                    corrected_query = correct_wrong_answer(wrong_query, H)  # correct the wrong query
                    m_prime = get_m_prime_RLIS(corrected_query, m, k, H, r)
                    trading_price = first_greater_element(data, m_prime)  # get the payoff
                    average_trading_price = average_trading_price + trading_price  # sum the payoff for next step

                average_trading_price = average_trading_price / average_coefficient  # calculate the average trading price

                if average_trading_price + 0.0000001 >= pure_online:
                    count += 1
                price_list.append(average_trading_price)
                price_array = np.array(price_list)
                count_list.append(count)
                count_array = np.array(count_list)

            sample_result_array.append(price_array)  # appen the payoff of this data sample to a list
            sample_count_array.append(count_array)
            wrong_bit_list_all.append(wrong_bit_list)  # append the number of wrong bit of this data sample to a list

        sample_array = np.array(sample_result_array, dtype=object)
        result_list_mean.append(sample_array)  # append all payoffs to result_list
        sample_count = np.array(sample_count_array, dtype=object)
        result_count.append(sample_count)

    result_array_mean = np.array(result_list_mean)
    count_array = np.array(result_count, dtype=object)
    result_count_list = list(count_array.sum(axis=0))
    result_mean = list(result_array_mean.mean(axis=0))  # take average payoff from all data samples

    average_pure_online = pure_online_sum / quantity_of_data  # take average payoff of pure online algorithm from all data samples
    average_best_price = best_price_sum / quantity_of_data  # take average best price from all data samples
    return result_mean, result_count_list, wrong_bit_list_all, average_pure_online, average_best_price


def plot_RLIS(result_list, eta_list_all, H_list, average_pure_online, average_best_price, save_path, x_label, y_label,
              title):
    # plot the figure
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list_all[i], result_list[i], label='H=%0.2f' % H_list[i])
    ax.axhline(average_pure_online, color='black', ls='dotted', label='ON*')
    ax.axhline(average_best_price, color='red', ls='dotted', label='M')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def plot_RLIS_count(result_list, eta_list_all, H_list, save_path, x_label, y_label,
              title):
    # plot the figure
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list_all[i], result_list[i], label='H=%0.2f' % H_list[i])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def save_to_csv_RLIS(payoff_list, eta_list, H_list, csv_path, pure_online, best_price):
    myDict = {}
    # save result to csv file
    for i in range(len(H_list)):
        myDict["Number of wrong bit(H=%0.2f)" % H_list[i]] = eta_list[i]
        myDict["payoff(H=%0.2f)" % H_list[i]] = payoff_list[i]
    myDict["pure online"] = [pure_online] * len(eta_list[-1])
    myDict["best price"] = [best_price] * len(eta_list[-1])
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def save_to_csv_RLIS_count(count_list, eta_list, H_list, csv_path):
    myDict = {}
    # save result to csv file
    for i in range(len(H_list)):
        myDict["Number of wrong bit(H=%0.2f)" % H_list[i]] = eta_list[i]
        myDict["count(H=%0.2f)" % H_list[i]] = count_list[i]
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def main():
    # choose dataset
    #data_set = "ETHUSD"
    #data_set = "BTCUSD"
    #data_set = "CADJPY"
    #data_set = "EURUSD"
    #data_set = "GBPUSD"
    #data_set = "AUDCHF"

    data_set = sys.argv[1]

    fileName = "data/" + data_set + ".csv"

    whole_period = 200  # the whole period
    trading_period = 200  # the trading period
    quantity_of_data = 20  # the number of data sample
    k = 25  # The value of k in solution 1

    starting_days = generate_uniform_data(fileName, 250, quantity_of_data)  # generate starting point uniformly

    # RLIS
    average_coefficient = 100  # the coefficient determines how many time we generate wrong queries for each error
    H_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # The value of H we want to test
    result, result_count, wrong_bit_list_all, average_pure_online, average_best_price = RLIS(fileName, starting_days, whole_period,
                                                                               quantity_of_data, trading_period, H_list,
                                                                               k, average_coefficient)
    save_path = "experiment_result/" + data_set + "/RLIS.png"  # path to save figures
    csv_path = "experiment_result/" + data_set + "/RLIS.csv"  # path to save csv file
    plot_RLIS(result, wrong_bit_list_all, H_list, average_pure_online, average_best_price, save_path,
              "error $\eta$",
              "average profit", "RLIS")
    # save result to csv file
    save_to_csv_RLIS(result, wrong_bit_list_all, H_list, csv_path, average_pure_online, average_best_price)

    save_path_count = "experiment_result/" + data_set + "/RLIS_count.png"
    csv_path_count = "experiment_result/" + data_set + "/RLIS_count.csv"
    plot_RLIS_count(result_count, wrong_bit_list_all, H_list, save_path_count,
              "error $\eta$",
              "number of samples better than ON*", "RLIS")
    save_to_csv_RLIS_count(result_count, wrong_bit_list_all, H_list, csv_path_count)


if __name__ == '__main__':
    main()
