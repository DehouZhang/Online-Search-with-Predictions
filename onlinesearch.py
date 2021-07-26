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
    file = pd.read_csv(file_name, header=None, sep="\t", usecols=[4], skiprows=350, nrows=250)
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

    r_list = [0.5, 0.75, 1, 1.5]
    # H-oblivious negative
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

    # H-Oblivious
    Hn_list1 = Hn_list.tolist()
    Hn_list1 = [-x for x in Hn_list1]
    Hp_list1 = Hp_list.tolist()
    H_list = Hn_list1[::-1] + Hp_list1

    payoff_list = payoff_list[::-1] + payoff_list5
    payoff_list2 = payoff_list2[::-1] + payoff_list6
    payoff_list3 = payoff_list3[::-1] + payoff_list7
    payoff_list4 = payoff_list4[::-1] + payoff_list8

    # draw H-oblivious
    fig, ax = plt.subplots()
    # ax.plot(H_list, payoff_list, label='r=0.5')
    # ax.plot(H_list, payoff_list2, label='r=0.75')
    # ax.plot(H_list, payoff_list3, label='r=1')
    # ax.plot(H_list, payoff_list4, label='r=1.5')

    cr = [v_star / x for x in payoff_list]
    cr2 = [v_star / x for x in payoff_list2]
    cr3 = [v_star / x for x in payoff_list3]
    cr4 = [v_star / x for x in payoff_list4]
    ax.plot(H_list, cr, label='r=0.5')
    ax.plot(H_list, cr2, label='r=0.75')
    ax.plot(H_list, cr3, label='r=1')
    ax.plot(H_list, cr4, label='r=1.5')
    ax.axhline(pure_online_cr, color='black', ls='dotted', label='Pure Online')

    ax.set_title("H-Oblivious")
    ax.set_xlabel("Hn")
    ax.set_ylabel("Expected Competitive Ratio")
    plt.legend()
    plt.show()

    # H-aware-negative
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
            payoff = h_aware_negative(data, v_star, Hn, 0.1, eta, M, m)
            payoff2 = h_aware_negative(data, v_star, Hn, 0.2, eta, M, m)
            payoff3 = h_aware_negative(data, v_star, Hn, 0.3, eta, M, m)
            payoff4= h_aware_negative(data, v_star, Hn, 0.4, eta, M, m)
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
        cr = [v_star / x for x in payoff_list]
        cr2 = [v_star / x for x in payoff_list2]
        cr3 = [v_star / x for x in payoff_list3]
        cr4 = [v_star / x for x in payoff_list4]

    fig, ax = plt.subplots()
    ax.plot(Hn_list, cr, label='Hp=0.1')
    ax.plot(Hn_list, cr2, label='Hp=0.2')
    ax.plot(Hn_list, cr3, label='Hp=0.3')
    ax.plot(Hn_list, cr4, label='Hp=0.4')
    ax.axhline(pure_online_cr, color='black', ls='dotted', label='Pure Online')
    ax.set_title("H-aware with fixed Hn")
    ax.set_xlabel("Hp")
    ax.set_ylabel("Expected Competitive Ratio")
    plt.legend()
    plt.show()

    # H-aware-positive
    Hp_list = np.linspace(0, Hp_bound, 100)
    payoff_list = list()
    payoff_list2 = list()
    payoff_list3 = list()
    payoff_list4 = list()
    for Hp in Hp_list:
        sum_payoff = 0
        sum_payoff2 = 0
        sum_payoff3 = 0
        sum_payoff4 = 0
        eta_list = np.linspace(0, Hp, 500)
        for eta in eta_list:
            payoff = h_aware_positive(data, v_star, 0.1, Hp, eta, M, m)
            payoff2 = h_aware_positive(data, v_star, 0.2, Hp, eta, M, m)
            payoff3 = h_aware_positive(data, v_star, 0.3, Hp, eta, M, m)
            payoff4 = h_aware_positive(data, v_star, 0.4, Hp, eta, M, m)
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
        cr = [v_star / x for x in payoff_list]
        cr2 = [v_star / x for x in payoff_list2]
        cr3 = [v_star / x for x in payoff_list3]
        cr4 = [v_star / x for x in payoff_list4]

    fig, ax = plt.subplots()
    ax.plot(Hp_list, cr, label='Hn=0.1')
    ax.plot(Hp_list, cr2, label='Hn=0.2')
    ax.plot(Hp_list, cr3, label='Hn=0.3')
    ax.plot(Hp_list, cr4, label='Hn=0.4')
    ax.axhline(pure_online_cr, color='black', ls='dotted', label='Pure Online')
    ax.set_title("H-aware with fixed Hp")
    ax.set_xlabel("Hn")
    ax.set_ylabel("Expected Competitive Ratio")
    plt.legend()
    plt.show()



'''
    for H in H_list:
        trading_prices = list()
        eps_list = list()
        b_lowerbound = 1/(1-H)
        b_list = np.linspace(b_lowerbound, 1.5*b_lowerbound, 10)

        for b in b_list:
            prices = list()
            eps = np.linspace(-H, H, 100)
            eps_list.append(eps.tolist())
            for e in eps:
                prices.append(online_search_untrusted(data, H, e, best_price, b))
            trading_prices.append(prices)

        for l in range(len(trading_prices)):
            plt.plot(eps_list[l], trading_prices[l], '-')
            plt.xlabel('Predicted Error \u03B5')
            plt.ylabel('Trading Price')
            title = "H = %.2f & b = %.2f" % (H, b_list[l])
            plt.title(title)
            #plt.show()
            plt.savefig("OnlineSearch_fig/" + title + ".png")
            plt.clf()

        for x, y in zip(eta_list[i], trading_prices[i]):
            label = y

            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        '''

if __name__ == '__main__':
    main()
