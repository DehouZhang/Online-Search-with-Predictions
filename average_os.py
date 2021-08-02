from math import sqrt, exp, isnan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def load_data_set(file_name):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=350, nrows=200)
    raw_data = []
    for i in file.values.tolist():
        raw_data.append(i[0])
    return raw_data


def get_maxmin(file_name):
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=350, nrows=400)
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
    if (1+Hn) / (1 - Hp) <= sqrt(M / m):
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
    data = load_data_set("data\ETHUSD.csv")
    M, m = get_maxmin("data\ETHUSD.csv")
    v_star = max(data)
    opt = pure_online(data, M, m)
    pure_online_cr = v_star/opt

    print("Best price is:", v_star)
    print("Pure online trading price: ", opt)
    print("M and m:", M, m)
    print("First price in trading period:", data[0])
    print("Last price in trading period:", data[-1])

    #H negative
    Hn_bound = (M - m) / m
    Hn_list = np.linspace(0, Hn_bound, 100)
    payoff_list = list()
    payoff_list2 = list()
    payoff_list3 = list()
    payoff_list4 = list()
    for Hn in Hn_list:
        sum_payoff = 0
        sum_payoff2 = 0
        sum_payoff3 = 0
        sum_payoff4 = 0
        eta_list = np.linspace(0, Hn, 500)
        for eta in eta_list:
            payoff = h_oblivious_negative(data, v_star, eta, 0.5)
            payoff2 = h_oblivious_negative(data, v_star, eta, 0.75)
            payoff3 = h_oblivious_negative(data, v_star, eta, 1)
            payoff4 = h_oblivious_negative(data, v_star, eta, 1.5)
            sum_payoff += payoff
            sum_payoff2 += payoff2
            sum_payoff3 += payoff3
            sum_payoff4 += payoff4
        average_payoff = sum_payoff / 500
        average_payoff2 = sum_payoff2 / 500
        average_payoff3 = sum_payoff3 / 500
        average_payoff4 = sum_payoff4 / 500
        payoff_list.append(average_payoff)
        payoff_list2.append(average_payoff2)
        payoff_list3.append(average_payoff3)
        payoff_list4.append(average_payoff4)

    # H-oblivious positive
    Hp_bound = (M - m) / M
    Hp_list = np.linspace(0, Hp_bound, 100)
    payoff_list5 = list()
    payoff_list6 = list()
    payoff_list7 = list()
    payoff_list8 = list()
    for Hp in Hp_list:
        sum_payoff5 = 0
        sum_payoff6 = 0
        sum_payoff7 = 0
        sum_payoff8 = 0
        eta_list = np.linspace(0, Hp, 500)
        for eta in eta_list:
            payoff5 = h_oblivious_positive(data, v_star, eta, 0.5)
            payoff6 = h_oblivious_positive(data, v_star, eta, 0.75)
            payoff7 = h_oblivious_positive(data, v_star, eta, 1)
            payoff8 = h_oblivious_positive(data, v_star, eta, 1.5)
            sum_payoff5 += payoff5
            sum_payoff6 += payoff6
            sum_payoff7 += payoff7
            sum_payoff8 += payoff8
        average_payoff5 = sum_payoff5 / 500
        average_payoff6 = sum_payoff6 / 500
        average_payoff7 = sum_payoff7 / 500
        average_payoff8 = sum_payoff8 / 500
        payoff_list5.append(average_payoff5)
        payoff_list6.append(average_payoff6)
        payoff_list7.append(average_payoff7)
        payoff_list8.append(average_payoff8)

        cr = [v_star / x for x in payoff_list]
        cr2 = [v_star / x for x in payoff_list2]
        cr3 = [v_star / x for x in payoff_list3]
        cr4 = [v_star / x for x in payoff_list4]

        cr5 = [v_star / x for x in payoff_list5]
        cr6 = [v_star / x for x in payoff_list6]
        cr7 = [v_star / x for x in payoff_list7]
        cr8 = [v_star / x for x in payoff_list8]






    # H-aware-negative
    Hn_list = np.linspace(0, Hn_bound, 100)
    payoff_list9 = list()
    payoff_list10 = list()
    payoff_list11 = list()
    payoff_list12 = list()
    for Hn in Hn_list:
        sum_payoff9 = 0
        sum_payoff10 = 0
        sum_payoff11 = 0
        sum_payoff12 = 0
        eta_list = np.linspace(0, Hn, 500)
        for eta in eta_list:
            payoff9 = h_aware_negative(data, v_star, Hn, 0.1, eta, M, m)
            payoff10 = h_aware_negative(data, v_star, Hn, 0.2, eta, M, m)
            payoff11 = h_aware_negative(data, v_star, Hn, 0.3, eta, M, m)
            payoff12= h_aware_negative(data, v_star, Hn, 0.4, eta, M, m)
            sum_payoff9 += payoff9
            sum_payoff10 += payoff10
            sum_payoff11 += payoff11
            sum_payoff12 += payoff12
        average_payoff9 = sum_payoff9 / 500
        average_payoff10 = sum_payoff10 / 500
        average_payoff11 = sum_payoff11 / 500
        average_payoff12 = sum_payoff12 / 500
        payoff_list9.append(average_payoff9)
        payoff_list10.append(average_payoff10)
        payoff_list11.append(average_payoff11)
        payoff_list12.append(average_payoff12)
        cr9 = [v_star / x for x in payoff_list9]
        cr10 = [v_star / x for x in payoff_list10]
        cr11 = [v_star / x for x in payoff_list11]
        cr12 = [v_star / x for x in payoff_list12]


    fig, ax = plt.subplots()
    ax.plot(Hn_list, cr, label='r=0.5')
    ax.plot(Hn_list, cr2, label='r=0.75')
    ax.plot(Hn_list, cr3, label='r=1')
    ax.plot(Hn_list, cr4, label='r=1.5')
    ax.plot(Hn_list, cr9, label='Hp=0.1')
    ax.plot(Hn_list, cr10, label='Hp=0.2')
    ax.plot(Hn_list, cr11, label='Hp=0.3')
    ax.plot(Hn_list, cr12, label='Hp=0.4')
    ax.axhline(pure_online_cr, color='black', ls='dotted', label='Pure Online')
    ax.set_title("Negative values of H")
    ax.set_xlabel("$H_n$")
    ax.set_ylabel("OPT/ALG")
    plt.legend(prop={'size': 7})
    plt.show()
    #fig.savefig("average_payoff_fig/" + "H_aware_fix_hn.png")

    # H-aware-positive
    Hp_list = np.linspace(0, Hp_bound, 100)
    payoff_list13 = list()
    payoff_list14 = list()
    payoff_list15 = list()
    payoff_list16 = list()
    for Hp in Hp_list:
        sum_payoff13 = 0
        sum_payoff14 = 0
        sum_payoff15 = 0
        sum_payoff16 = 0
        eta_list = np.linspace(0, Hp, 500)
        for eta in eta_list:
            payoff13 = h_aware_positive(data, v_star, 0.1, Hp, eta, M, m)
            payoff14 = h_aware_positive(data, v_star, 0.2, Hp, eta, M, m)
            payoff15 = h_aware_positive(data, v_star, 0.3, Hp, eta, M, m)
            payoff16 = h_aware_positive(data, v_star, 0.4, Hp, eta, M, m)
            sum_payoff13 += payoff13
            sum_payoff14 += payoff14
            sum_payoff15 += payoff15
            sum_payoff16 += payoff16
        average_payoff13 = sum_payoff13 / 500
        average_payoff14 = sum_payoff14 / 500
        average_payoff15 = sum_payoff15 / 500
        average_payoff16 = sum_payoff16 / 500
        payoff_list13.append(average_payoff13)
        payoff_list14.append(average_payoff14)
        payoff_list15.append(average_payoff15)
        payoff_list16.append(average_payoff16)
        cr13 = [v_star / x for x in payoff_list13]
        cr14 = [v_star / x for x in payoff_list14]
        cr15 = [v_star / x for x in payoff_list15]
        cr16 = [v_star / x for x in payoff_list16]


    fig, ax = plt.subplots()
    ax.plot(Hn_list, cr5, label='r=0.5')
    ax.plot(Hn_list, cr6, label='r=0.75')
    ax.plot(Hn_list, cr7, label='r=1')
    ax.plot(Hn_list, cr8, label='r=1.5')
    ax.plot(Hp_list, cr13, label='Hn=0.1')
    ax.plot(Hp_list, cr14, label='Hn=0.2')
    ax.plot(Hp_list, cr15, label='Hn=0.3')
    ax.plot(Hp_list, cr16, label='Hn=0.4')
    ax.axhline(pure_online_cr, color='black', ls='dotted', label='Pure Online')
    ax.set_title("positive values of H")
    ax.set_xlabel("$H_p$")
    ax.set_ylabel("OPT/ALG")
    plt.legend(prop={'size': 7})
    plt.show()
    #fig.savefig("average_payoff_fig/" + "H_aware_fix_hp.png")




if __name__ == '__main__':
    main()
