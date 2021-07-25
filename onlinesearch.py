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


def search_based(data, M, m):
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


def online_search_untrusted(data, Hl, Hu, eta_u, M, m):
    v = M / (1 + eta_u)
    v_prime = sqrt(M * m)
    if (1 + Hu) / (1 - Hl) <= sqrt(M / m):
        v_prime = v * (1 - Hl)
    trading_price = first_greater_element(data, v_prime)
    return trading_price


def main():
    data = load_data_set("data\ETHUSD.csv")
    print("Best price is:", max(data))
    best_price = max(data)
    # plt.plot(data)
    # plt.show()
    M, m = get_maxmin("data\ETHUSD.csv")
    opt = search_based(data, M, m)
    print("Pure online trading price: ", opt)
    print("M and m:", M, m)
    print("First price in trading period:", data[0])
    print("Last price in trading period:", data[-1])
    #Hl_list = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    Hl_list = np.linspace(0, 1, 100)
    average_payoff_list = list()
    average_payoff_list_two = list()

    for Hl in Hl_list:
        eta_u_list = np.linspace(0, Hl, 500)
        average_payoff = 0
        average_payoff_two = 0
        for eta_u in eta_u_list:
            payoff = online_search_untrusted(data, Hl, Hl, eta_u, M, m)
            payoff_two = online_search_untrusted(data, Hl, 0.5*Hl, eta_u, M, m)
            average_payoff += payoff
            average_payoff_two += payoff_two
        average_payoff = average_payoff / 500
        average_payoff_list.append(average_payoff)

        average_payoff_two = average_payoff_two / 500
        average_payoff_list_two.append(average_payoff_two)




    # draw
    # hu=hl
    plt.plot(Hl_list, average_payoff_list, color='r', markerfacecolor='blue', marker='o', label='Untrusted ALG')
    plt.axhline(opt, color='blue', linestyle='--', label='Pure Online')
    plt.xlabel('Hu =Hl')
    plt.ylabel('Average Payoff')
    title = "Average Payoff for H"
    plt.title(title)
    for a, b in zip(Hl_list, average_payoff_list):
        b = float(b)
        plt.text(a, b * 1.02, "%.2f" % b, ha='right', va='center', fontsize=6)
    plt.legend()
    plt.show()

    #hu=2hl
    plt.plot(Hl_list, average_payoff_list_two, color='r', markerfacecolor='blue', marker='o', label='Untrusted ALG')
    plt.axhline(opt, color='blue', linestyle='--', label='Pure Online')
    plt.xlabel('Hu =2Hl')
    plt.ylabel('Average Payoff')
    title = "Average Payoff for H"
    plt.title(title)
    for a, b in zip(Hl_list, average_payoff_list_two):
        b = float(b)
        plt.text(a, b * 1.02, "%.2f" % b, ha='right', va='center', fontsize=6)
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
