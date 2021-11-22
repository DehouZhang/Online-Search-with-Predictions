"""
Experiment: Query Based prediction
Generate result and plot the RLIS and RLIS-H algorithm
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


def first_greater_and_smaller_element(searching_list, m_prime_large, M_prime_small):
    # return the first element in a list which is greater than a value
    large_index = first_greater_element(searching_list, m_prime_large)
    small_index = first_smaller_element(searching_list, M_prime_small)
    find = False
    if large_index == small_index:
        small_index += 1
        for i in range(small_index, len(searching_list)):
            if searching_list[i] <= (M_prime_small - 0.0001):
                small_index = i
                find = True
        if not find:
            small_index = len(searching_list)-1
    large_price = searching_list[large_index]
    small_price = searching_list[small_index]
    return large_price, small_price


def first_smaller_element(searching_list, element):
    # return the first element in a list which is greater than a value
    result = None
    for i in range(len(searching_list)):
        if searching_list[i] <= (element - 0.0001):
            result = i
            break
    if result is None:
        result = len(searching_list)
    return result


def first_greater_element(searching_list, element):
    # return the first element in a list which is greater than a value
    result = None
    for i in range(len(searching_list)):
        if searching_list[i] >= (element - 0.0001):
            result = i
            break
    if result is None:
        result = len(searching_list)
    return result


def generate_query(k, M, m, v_star):
    # generate correct query
    r = (M / m) ** (1/k)
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


def generate_wrong_query(query_large, query_small, wrong_number):
    # generate wrong query
    query_large_size = len(query_large)
    query = query_large + query_small
    n = len(query)
    # randomly flip bit to generate wrong query
    wrong_position_list = random.sample(range(0, n), wrong_number)  # random from 0 to k
    # 0 means "NO" and 1 means "YES"
    for i in wrong_position_list:
        if query[i] == 0:
            query[i] = 1
        else:
            query[i] = 0
    return query[0:query_large_size], query[query_large_size:]  # return the wrong query


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
        return True
    else:
        return False


def check_query_all_yes(query):
    # check if query is full of "No" query
    count = 0
    for i in range(len(query)):
        if query[i] == 0:
            count = count + 1
    # if full of "No", set the last query to "YES"
    if count == 0:
        return True
    else:
        return False


def get_lowerbound_large(corrected_query, m, n, p, r):
    if check_query_all_no(corrected_query):
        m_prime = m * (r ** (len(corrected_query)-p+1))
    elif check_query_all_yes(corrected_query):
        m_prime = m
    else:
        i = find_first_yes(corrected_query)
        j = find_last_no(corrected_query)
        if i > j:
            i_prime = j  # calculate i_prime and j_prime
        else:
            alpha = count_alpha(corrected_query, i, j)
            i_prime = i - p + alpha

            # if i_prime is out of range or j_prime is out of range
            if i_prime < 0:
                i_prime = 0
        m_prime = m * (r ** i_prime)
    return m_prime


def get_upperbound_small(corrected_query, m, M, n, p, r):
    if check_query_all_no(corrected_query):
        M_prime = M
    elif check_query_all_yes(corrected_query):
        M_prime = m * r
    else:
        i = find_first_yes(corrected_query)
        j = find_last_no(corrected_query)
        if i > j:
            j_prime = i  # calculate i_prime and j_prime
        else:
            alpha = count_alpha(corrected_query, i, j)
            j_prime = p + i + alpha - 1

            if j_prime > n:
                j_prime = n
        M_prime = m * (r ** j_prime)
    return M_prime


def RIS(fileName, starting_days, whole_period, quantity_of_data, trading_period, H_list, n, average_coefficient):
    result_list_mean = list()
    sum_best_difference = 0
    for starting_day in starting_days:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        p_large = max(data)  # the highest price
        n_large = ceil(n / 2)

        p_small = min(data)  # the lowest price
        n_small = n - n_large

        r_large = (M / m) ** (1 / n_large)  # calculate r
        r_small = (M / m) ** (1 / n_small)

        wrong_bit_list_all = list()
        sample_result_array = list()

        best_difference = p_large - p_small
        #print("best_price_differnece = ",best_difference)
        sum_best_difference += best_difference
        for H in H_list:
            price_list = list()
            eta_list = list(range(ceil(H * n) + 1))  # generate list of value of eta
            for eta in eta_list:
                sum_price_difference = 0
                #valid_data = average_coefficient
                for l in range(average_coefficient):
                    query_large = generate_query(n_large, M, m, p_large)  # generate the correct query
                    query_small = generate_query(n_small, M, m, p_small)

                    if check_query_all_no(query_large):  # if query is full of "NO", flip the last bit to "YES"
                        query_large[-1] = 1

                    wrong_query_large, wrong_query_small = generate_wrong_query(query_large,query_small, eta)  # generate wrong query

                    corrected_query_large = correct_wrong_answer(wrong_query_large, H)  # correct the wrong query
                    corrected_query_small = correct_wrong_answer(wrong_query_small, H)
                    lowerbound_large = get_lowerbound_large(corrected_query_large, m, n_large, ceil(H * n), r_large)
                    upperbound_small = get_upperbound_small(corrected_query_small, m, M, n_small, ceil(H * n), r_small)
                    #print(lowerbound_large - upperbound_small)
                    #M_prime_small = get_m_prime(corrected_query_small, m, n, H, r)

                    trading_price_large, trading_price_small = first_greater_and_smaller_element(data, lowerbound_large, upperbound_small)
                    price_difference = trading_price_large - trading_price_small
                    if price_difference < 0:
                        price_difference = 0
                        #valid_data = valid_data - 1

                    #print(price_difference)
                    sum_price_difference += price_difference  # sum the payoff for next step
                #if valid_data ==0:
                    #valid_data = 1
                average_price_difference = sum_price_difference / average_coefficient  # calculate the average trading price
                price_list.append(average_price_difference)
                price_array = np.array(price_list)

            sample_result_array.append(price_array)  # appen the payoff of this data sample to a list

            wrong_bit_list_all.append(eta_list)  # append the number of wrong bit of this data sample to a list

        sample_array = np.array(sample_result_array, dtype=object)
        result_list_mean.append(sample_array)  # append all payoffs to result_list

    result_array_mean = np.array(result_list_mean)
    result_mean = list(result_array_mean.mean(axis=0))  # take average payoff from all data samples
    average_best_difference = sum_best_difference/quantity_of_data
    return result_mean, wrong_bit_list_all, average_best_difference


def plot_RIS(result_list, eta_list_all, H_list, average_price_difference, save_path, x_label, y_label,
              title):
    # plot the figure
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list_all[i], result_list[i], label='H=%0.2f' % H_list[i])
    ax.axhline(average_price_difference, color='red', ls='dotted', label='best_price_difference')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.legend()
    fig.savefig(save_path)
    plt.show()


def save_to_csv_RIS(payoff_list, eta_list, H_list, csv_path, best_price_difference):
    myDict = {}
    # save result to csv file
    for i in range(len(H_list)):
        myDict["Number of wrong bit(H=%0.2f)" % H_list[i]] = eta_list[i]
        myDict["payoff(H=%0.2f)" % H_list[i]] = payoff_list[i]
    myDict["best price difference"] = [best_price_difference] * len(eta_list[-1])
    df = pd.DataFrame.from_dict(myDict, orient='index').transpose()
    df.to_csv(csv_path)


def main():
    # choose dataset
    #data_set = "ETHUSD"
    # data_set = "BTCUSD"
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
    average_coefficient = 1000  # the coefficient determines how many time we generate wrong queries for each error
    H_list = [0.1, 0.2, 0.3, 0.4, 0.5]  # The value of H we want to test
    #H_list = [0.1]
    result, wrong_bit_list_all, average_best_difference = RIS(fileName, starting_days, whole_period, quantity_of_data, trading_period, H_list, k,
                                     average_coefficient)
    print(result)
    print(average_best_difference)
    save_path = "experiment_result/" + data_set + "/RIS.png"  # path to save figures
    csv_path = "experiment_result/" + data_set + "/RIS.csv"  # path to save csv file
    plot_RIS(result, wrong_bit_list_all, H_list, average_best_difference, save_path, "error $\eta$", "average price difference", "RIS")
    # save result to csv file
    save_to_csv_RIS(result, wrong_bit_list_all, H_list, csv_path, average_best_difference)

    '''
    # RLIS-H
    average_coefficient = 1000
    H_list = list(range(k + 1))
    eta_list = [0, 1 / 2, 2 / 3, 3 / 4]
    result, average_pure_online, average_best_price = RLIS_H(fileName, starting_days, whole_period, quantity_of_data,
                                                             trading_period, H_list, eta_list, k,
                                                             average_coefficient)
    save_path = "experiment_result/" + data_set + "/RLIS-H.png"  # path to save figures
    csv_path = "experiment_result/" + data_set + "/RLIS-H.csv"  # path to save csv file
    plot_RLIS_H(result, eta_list, H_list, average_pure_online, average_best_price, save_path, "H", "average profit",
                "RLIS-H")
    save_to_csv_RLIS_H(result, eta_list, H_list, csv_path, average_pure_online, average_best_price)
    '''


if __name__ == '__main__':
    main()
