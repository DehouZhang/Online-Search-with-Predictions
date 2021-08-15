from math import sqrt, exp, isnan
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


def pure_online(data, M, m):
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


def main():
    starting_day = 580
    whole_period = 300
    trading_period = 200
    coefficient = 100000
    data = load_data_set("data\ETHUSD.csv", starting_day, trading_period)
    M, m = get_maxmin("data\ETHUSD.csv", starting_day, whole_period)
    v_star = max(data)
    online = pure_online(data, M, m)

    print("Best price is:", v_star)
    print("Pure online trading price: ", online)
    print("M and m:", M, m)
    print("First price in trading period:", data[0])
    print("Last price in trading period:", data[-1])

    r_list = [0.5, 0.75, 1, 1.5]
    Hn_bound = (M - m) / m
    Hp_bound = (M - m) / M
    fig, ax = plt.subplots()
    for r in r_list:
        eta_list_n = np.linspace(0, Hn_bound, int(Hn_bound * coefficient)).tolist()
        eta_list_p = np.linspace(0, Hp_bound, int(Hn_bound * coefficient)).tolist()
        payoff_list_n = list()
        payoff_list_p = list()

        for eta_n in eta_list_n:
            payoff_list_n.append(h_oblivious_negative(data, v_star, eta_n, r))
        for eta_p in eta_list_p:
            payoff_list_p.append(h_oblivious_positive(data, v_star, eta_p, r))

        payoff_list = payoff_list_n[::-1] + payoff_list_p

        eta_list_n = [-x for x in eta_list_n]
        eta_list = eta_list_n[::-1] + eta_list_p

        ax.plot(eta_list, payoff_list, label='r=%0.2f' % r)

    ax.axhline(online, color='black', ls='dotted', label='Pure Online')
    # plt.xticks([-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5], ['2.5', '2.0', '1.5', '1.0', '0.5', '0.0', '0.5'])
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("Payoff")
    ax.legend(prop={'size': 7})
    fig.savefig("eta_payoff_fig/" + "h_oblivious.png")
    plt.show()

    # H aware
    Hn_Hp_list = [(0.05, 0.05), (0.05, 0.1), (0.2, 0.2), (0.4, 0.4)]
    fig, ax = plt.subplots()

    for hn_hp in Hn_Hp_list:
        eta_list_n = np.linspace(0, hn_hp[0], int(hn_hp[0]*coefficient)).tolist()
        eta_list_p = np.linspace(0, hn_hp[1], int(hn_hp[1]*coefficient)).tolist()
        payoff_list_n = list()
        payoff_list_p = list()

        for eta_n in eta_list_n:
            payoff_list_n.append(h_aware_negative(data, v_star, hn_hp[0], hn_hp[1], eta_n, M, m))

        for eta_p in eta_list_p:
            payoff_list_p.append(h_aware_positive(data, v_star, hn_hp[0], hn_hp[1], eta_p, M, m))

        payoff_list = payoff_list_n[::-1] + payoff_list_p

        eta_list_n = [-x for x in eta_list_n]
        eta_list = eta_list_n[::-1] + eta_list_p

        ax.plot(eta_list, payoff_list, label='$H_n$=%0.2f, $H_p$=%0.2f' % (hn_hp[0], hn_hp[1]))
    ax.axhline(online, color='black', ls='dotted', label='Pure Online')
    # plt.xticks([-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3], ['0.3', '0.2', '0.1', '0.0', '0.1', '0.2', '0.3'])
    ax.set_xlabel("$\eta$")
    ax.set_ylabel("Payoff")

    ax.legend(prop={'size': 7})

    fig.savefig("eta_payoff_fig/" + "h_aware.png")
    plt.show()


if __name__ == '__main__':
    main()
