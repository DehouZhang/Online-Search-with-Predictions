from math import sqrt, exp, isnan, ceil
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


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


def generate_random_data(file_name, period, number_of_data):
    data_size = get_size_data(file_name)
    random_list = random.sample(range(0, data_size - period), number_of_data)
    return random_list


def online(data, M, m):
    reservation_price = sqrt(M * m)
    trade_price = first_greater_element(data, reservation_price)
    return trade_price


def first_greater_element(searching_list, element):
    result = None
    for item in searching_list:
        if item >= element:
            result = item
            break
    if result is None:
        result = searching_list[-1]
    return result


def generate_query(k, M, m, v_star):
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
    return query


def generate_wrong_query(query, H, eta):
    k = len(query)
    p = ceil(k * H)
    wrong_number = ceil(k * eta)
    wrong_position_list = random.sample(range(0, k), wrong_number)  # random from 0 to 99
    for i in wrong_position_list:
        if query[i] == 0:
            query[i] = 1
        else:
            query[i] = 0
    return query


def check_no_answers(query, position, H):
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
    k = len(query)
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
    return query


def find_first_yes(query):
    for i in range(len(query)):
        if query[i] == 1:
            return i


def find_last_no(query):
    for i in reversed(range(len(query))):
        if query[i] == 0:
            return i


def count_alpha(query, i, j):
    alpha = 0
    for k in range(i, j):
        if query[k] == 0:
            alpha = alpha + 1
    return alpha


def check_query_all_no(query):
    count = 0
    for i in range(len(query)):
        if query[i] == 1:
            count = count + 1
    if count == 0:
        query[-1] = 1
        return True
    else:
        return False


def plot(result_list, eta_list_all, H_list, average_pure_online, average_best_price):
    fig, ax = plt.subplots()
    for i in range(len(result_list)):
        ax.plot(eta_list_all[i], result_list[i], label='H=%0.2f' % H_list[i])
    ax.axhline(average_pure_online, color='black', ls='dotted', label='Pure Online')
    ax.axhline(average_best_price, color='red', ls='dotted', label='Best Price')
    plt.legend()
    #fig.savefig("query_solution1_fig/" + "solution1.png")
    plt.show()


def main():
    fileName = "data\ETHUSD.csv"
    # starting_day = 580
    whole_period = 200
    trading_period = 100
    quantity_of_data = 20
    k = 80
    eta_coefficient = 1000
    average_coefficient = 30

    random_list = generate_random_data(fileName, whole_period, quantity_of_data)
    H_list = [0.1, 0.2, 0.3,0.4,0.5]
    result_list = list()
    average_pure_online = 0
    average_best_price = 0
    for starting_day in random_list:
        data = load_data_set(fileName, starting_day, trading_period)
        M, m = get_maxmin(fileName, starting_day, whole_period)
        pure_online = online(data, M, m)
        v_star = max(data)
        r = (M / m) ** (1 / k)
        average_pure_online += pure_online
        average_best_price += v_star

        sample_result = list()
        eta_list_all = list()
        for H in H_list:
            eta_list = np.linspace(0, H, int(H * eta_coefficient))
            price_list = list()
            for eta in eta_list:

                average_trading_price = 0
                for l in range(average_coefficient):
                    query = generate_query(k, M, m, v_star)
                    if check_query_all_no(query):
                        query[-1] = 1
                    wrong_query = generate_wrong_query(query, H, eta)
                    corrected_query = correct_wrong_answer(wrong_query, H)
                    if check_query_all_no(corrected_query):
                        m_prime = m
                        M_prime = M
                    else:
                        i = find_first_yes(corrected_query)
                        j = find_last_no(corrected_query)
                        if i > j:
                            i_prime = j
                            j_prime = i
                        else:
                            alpha = count_alpha(corrected_query, i, j)
                            p = ceil(k * H)
                            i_prime = i - p + alpha
                            j_prime = p + i + alpha - 1
                            '''
                            if i_prime < 0 or j_prime > k:
                                i_prime = 0
                                j_prime = k
                            '''
                            if i_prime < 0:
                                i_prime = 0
                            if j_prime > k:
                                j_prime = k

                        m_prime = m * (r ** i_prime)
                        M_prime = m * (r ** j_prime)
                    trading_price = online(data, M_prime, m_prime)
                    average_trading_price = average_trading_price + trading_price
                average_trading_price = average_trading_price / average_coefficient
                price_list.append(average_trading_price)
                price_array = np.array(price_list)

            eta_list_all.append(eta_list)
            sample_result.append(price_array)
            sample_array = np.array(sample_result)

        result_list.append(sample_array)
        result_array = np.array(result_list)

    result = list(result_array.mean(axis=0))
    average_pure_online = average_pure_online / quantity_of_data
    average_best_price = average_best_price / quantity_of_data

    # draw
    plot(result, eta_list_all, H_list, average_pure_online, average_best_price)

    print("1")


'''
        ax.axhline(pure_online, color='black', ls='dotted', label='Pure Online')
        ax.axhline(v_star, color='red', ls='dotted', label='Best Price')
        ax.set_xlim([0, H])
        ax.set_xlabel("$\eta$")
        ax.set_ylabel("Average Payoff")
        ax.set_title("Query Model Solution 1")
        ax.legend(prop={'size': 7})
        fig.savefig("query_solution1_fig/" + "solution1.png")

        plt.show()

        #plt.plot(data)
        #plt.show()
'''

if __name__ == '__main__':
    main()
