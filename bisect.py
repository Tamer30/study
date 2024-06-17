import math

def bisect(function, left, right, error):
    # Цикл while повторяет код, пока выполняется условие.
    # Добавили в него условие остановки.
    while right - left > error:

        # проверяем, нет ли нулей
        if function(left) == 0:
            return left
        if function(right) == 0:
            return right
        # < напишите код здесь >
        
        # делим отрезок пополам и находим новый отрезок
        middle = (left + right) / 2
        if function(left) * function(middle) < 0:
            right = middle
        else:
            left = middle
        # < напишите код здесь >
        if function(middle) == 0:
            return middle
    return left


def f1(x):
    return x**3 - x**2 - 2*x 


def f2(x):
    return (x+1)*math.log10(x) - x**0.75


print(bisect(f1, 1, 4, 0.000001))
print(bisect(f2, 1, 4, 0.000001))
