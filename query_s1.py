from math import sqrt, exp, isnan, ceil
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data_set(file_name):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=580, nrows=200)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return raw_data


def get_maxmin(file_name):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=580, nrows=400)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return max(raw_data), min(raw_data)


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


def generate_wrong_query(query, H):
    k = len(query)
    p = ceil(k * H)
    wrong_position_list = random.sample(range(0, k), p)  # random from 0 to 99
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
    p = ceil(k * H)
    for i in range(k):
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


def main():
    data = load_data_set("data\ETHUSD.csv")
    M, m = get_maxmin("data\ETHUSD.csv")
    pure_online = online(data,M,m)
    v_star = max(data)
    k = 100
    r = (M / m) ** (1 / k)
    H_list = np.linspace(0,1,20)
    #H_list = [0.3,0.35,0.4]
    average_coefficient = 1000
    price_list = list()
    for H in H_list:
        average_trading_price = 0
        for i in range(average_coefficient):
            query = generate_query(k, M, m, v_star)
            wrong_query = generate_wrong_query(query, H)
            corrected_query = correct_wrong_answer(wrong_query, H)
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

                if i_prime < 0 or j_prime > k:
                    i_prime = 0
                    j_prime = k

                '''
                if i_prime < 0:
                    i_prime = 0
                if j_prime > k:
                    j_prime = k
                '''

            m_prime = m * (r ** i_prime)
            M_prime = m * (r ** j_prime)
            trading_price = online(data, M_prime, m_prime)
            average_trading_price = average_trading_price+trading_price
        average_trading_price = average_trading_price/average_coefficient
        price_list.append(average_trading_price)
    print(price_list)
    print(online(data,M,m))
    print(H_list)

    #draw
    fig, ax = plt.subplots()
    ax.plot(H_list, price_list, label='Query Model Solution 1')
    ax.axhline(pure_online, color='black', ls='dotted', label='Pure Online')
    ax.axhline(v_star, color='red', ls='dotted', label='Best Price')
    ax.set_xlim([0,1])
    #plt.xticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5], ['2.5', '2.0', '1.5', '1.0', '0.5', '0.0', '0.5'])
    ax.set_xlabel("H")
    ax.set_ylabel("Average Trading Price")
    ax.legend(prop={'size': 7})
    fig.savefig("query_solution1_fig/" + "solution1.png")
    plt.show()


if __name__ == '__main__':
    main()
