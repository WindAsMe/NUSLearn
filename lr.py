import csv
'''
User ID: t0916129
Name (English): Zhong Rui
'''


def stochastic_gradient_descent(x, y, theta, alpha, m, max_iter, tiny):

    deviation = 1
    iter = 0
    flag = 0
    while True:
        for i in range(m):
            deviation = 0
            h = theta[0] * x[i] + theta[1] * x[i]
            theta[0] = theta[0] + alpha * (y[i] - h)*x[i]
            theta[1] = theta[1] + alpha * (y[i] - h)*x[i]

            iter = iter + 1
            for i in range(m):
                deviation = deviation + (y[i] - (theta[0] * x[i] + theta[1] * x[i])) ** 2
            if deviation < tiny or iter > max_iter:
                flag = 1
                break
        if flag == 1:
            break
    return theta, iter


if __name__ == '__main__':
    csv_reader = csv.reader(open('train_data.csv'))
    coeff = 22.47
    intercept = -3.4
    x = []
    y = []
    for row in csv_reader:
        x.append(row[0])
        y.append(row[1])
    stochastic_gradient_descent(x, y, y, 0.5, len(x), 500, 0.01)