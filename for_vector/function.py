import numpy as np

def sigmoid(value):
    """Вычисление сигмоидной функции активации"""
    return 1 / (1 + np.exp(-value))

def softmax(value):
    """Вычисление функции softmax"""
    return np.exp(value)/np.sum(np.exp(value))    

def learning(array_vector, nw):
    """Обучение"""
    epoch = 0
    while epoch < 200:
        num = np.random.randint(3)
        E_end = np.zeros(3)
        E_end[num] = 1
        nw.value_hidden = sigmoid(np.dot(array_vector[num], nw.w_input.T) - nw.T_input)
        nw.value_output = softmax(np.dot(nw.value_hidden, nw.w_hidden.T) - nw.T_hidden)
        nw.gamma_output = nw.value_output - E_end
        nw.gamma_hidden = np.dot(nw.gamma_output * nw.value_output * (1 - nw.value_output), nw.w_hidden)
        nw.change_w_hidden()
        nw.change_w_input(array_vector[num])
        nw.change_T_hidden()
        nw.change_T_input()
        epoch += 1

def prediction(array_vector, nw):
    input_vector = np.array([[1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1]])
    nw.value_hidden = sigmoid(np.dot(input_vector, nw.w_input.T) - nw.T_input)
    nw.value_output = softmax(np.dot(nw.value_hidden, nw.w_hidden.T) - nw.T_hidden)
    value = nw.value_output.argmax()
    original_vector = array_vector[value]
    change_number = len(original_vector[original_vector != input_vector])
    print("Вектор на входе: ", input_vector)
    print("Истинный вектор: ", original_vector)
    print("Ваш вектор: " + str(value + 1) + ". Было инвертировано " + str(change_number) + " значений")