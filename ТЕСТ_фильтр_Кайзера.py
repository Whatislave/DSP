import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fixedpoint import FixedPoint



class ComplexFixedPoint:
    def __init__(self, real, imag, int_bits, frac_bits):
        self.real = FixedPoint(real, signed=True, m=int_bits, n=frac_bits)
        self.imag = FixedPoint(imag, signed=True, m=int_bits, n=frac_bits)
    
    def __mul__(self, other):
        # Комплексное умножение: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        real_part = a*c - b*d
        imag_part = a*d + b*c
        return ComplexFixedPoint(real_part, imag_part, 
                                self.real.m, self.real.n)
    
    def __add__(self, other):
        real_part = self.real + other.real
        imag_part = self.imag + other.imag
        return ComplexFixedPoint(real_part, imag_part, 
                                self.real.m, self.real.n)
    
    def to_complex(self):
        return complex(float(self.real), float(self.imag))

def design_fir_hamming(N, cutoff, width, nyquist_freq ):
    """Проектирование КИХ-фильтра высоких частот с использованием окна Хэмминга"""
    # Нормализация частот

    cutoff_norm = cutoff / nyquist_freq
    width_norm = width / nyquist_freq
    
    # Создание фильтра
    taps = signal.firwin(N, cutoff=cutoff_norm, width=width_norm, 
                         pass_zero=False, window='hamming')
    
    # Преобразуем в комплексные коэффициенты (можно добавить мнимую часть)
    complex_taps = [complex(t, 0) for t in taps]
    
    return complex_taps

def design_fir_kaiser(N, cutoff, width, nyquist_freq, beta=4.0):
    """Проектирование КИХ-фильтра высоких частот с использованием окна Хэмминга"""
    # Нормализация частот

    cutoff_norm = cutoff / nyquist_freq
    width_norm = width / nyquist_freq
    
    # Создание фильтра
    taps = signal.firwin(N, cutoff=cutoff_norm, width=width_norm, 
                         pass_zero=False, window=('kaiser', beta))
    
    # Преобразуем в комплексные коэффициенты (можно добавить мнимую часть)
    complex_taps = [complex(t, 0) for t in taps]
    
    return complex_taps

"""Реализация КИХ-фильтра с комплексными числами с фиксированной точкой"""
def fir_filter(input_signal, coefficients):

    output = []
    N = len(coefficients)
    
    for n in range(len(input_signal)):
        # Создаем буфер из последних N отсчетов
        buffer = input_signal[max(0, n-N+1):n+1]
        buffer = buffer[::-1]  # обратный порядок для свертки
        
        # Дополняем нулями если нужно
        if len(buffer) < N:
            buffer = buffer + [ComplexFixedPoint(0, 0, 8, 8)] * (N - len(buffer))
        
        # Вычисляем выходное значение
        y = ComplexFixedPoint(0, 0, 8, 8)
        for i in range(N):
            y = y + buffer[i] * coefficients[i]
        
        output.append(y)
    
    return output


"""Тестиреум КИХ-фильтр с комплексными числами с фиксированной точкой"""
# Параметры фильтра
N = 19  # порядок фильтра
cutoff_freq = 0.2  # частота среза (нормализованная к частоте Найквиста)
width = 0.1  # ширина перехода
int_bits = 8  # целая часть
frac_bits = 8  # дробная часть
nyquist_freq = 0.5
beta = 8.0

# Проектируем фильтр высоких частот
coeff_complex = design_fir_hamming(N, cutoff_freq, width, nyquist_freq)
coeff_fixed = [ComplexFixedPoint(coeff.real, coeff.imag, int_bits, frac_bits) 
                      for coeff in coeff_complex]
coeff_complex2 = design_fir_kaiser(N, cutoff_freq, width, nyquist_freq, beta)
coeff_fixed2 = [ComplexFixedPoint(coeff.real, coeff.imag, int_bits, frac_bits) 
                      for coeff in coeff_complex2]

# Создаем тестовый сигнал (сумма низкой и высокой частоты)
freq_sample = 1000  # частота дискретизации
time = np.linspace(0, 1, freq_sample)
LowFreq = 50  #пик на низкой частоте
HighFreq = 200  #пик на высокой частоте
#соответсвенно их амлитуды
amplitudeLF = 1
amplitudeHF = 0.5

input_signal_complex = [amplitudeLF*np.exp(1j*2*np.pi*LowFreq*t_) + amplitudeHF*np.exp(1j*2*np.pi*HighFreq*t_) 
                       for t_ in time]

# Преобразуем входной сигнал в формат с фиксированной точкой
input_signal_fixed = [ComplexFixedPoint(x.real, x.imag, int_bits, frac_bits) 
                     for x in input_signal_complex]

# Применяем фильтры
out_signal = fir_filter(input_signal_fixed, coeff_fixed)
out_signal2 = fir_filter(input_signal_fixed, coeff_fixed2)

# Преобразуем результаты обратно в комплексные числа для визуализации
out_complex = [y.to_complex() for y in out_signal]
out_complex2 = [y.to_complex() for y in out_signal2]

# Визуализация временных сигналов
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(time, np.real(input_signal_complex), label='Вход (действ.)')
plt.plot(time, np.real(out_complex), label='Выход (действ.)')
plt.title('Действительная часть сигнала')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(time, np.imag(input_signal_complex), label='Вход (мнимая)')
plt.plot(time, np.imag(out_complex), label='Выход (мнимая)')
plt.title('Мнимая часть сигнала')
plt.legend()

# График 3: АЧХ обоих фильтров на одном графике
w, h = signal.freqz([c.real for c in coeff_complex])
w2, h2 = signal.freqz([c2.real for c2 in coeff_complex2])
plt.subplot(3, 1, 3)
plt.plot(w/np.pi, 20*np.log10(np.abs(h)), 'b', linewidth=2, label='ФВЧ 1 (срез=0.2)')
plt.plot(w2/np.pi, 20*np.log10(np.abs(h2)), 'r--', linewidth=2, label='ФВЧ 2 (срез=0.2)')
plt.title('Сравнение АЧХ фильтров')
plt.ylabel('Амплитуда (dB)')
plt.xlabel('Нормализованная частота (xπ рад/отсчет)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Проверка работы фильтра - спектры до и после
fft_input = np.fft.fft(input_signal_complex)
fft_out = np.fft.fft(out_complex)
freqs = np.fft.fftfreq(len(time), 1/freq_sample)

# Модифицированная визуализация спектров
plt.figure(figsize=(12, 8))

# Спектры сигналов
plt.subplot(2, 1, 1)
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_input[:len(freqs)//2])), 
         label='Входной сигнал')
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_out[:len(freqs)//2])), 
         label='Выходной сигнал (ФВЧ Хэмминга)')
plt.title('Спектры сигналов до и после фильтрации')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда (dB)')
plt.legend()
plt.grid()

# АЧХ фильтров в одном масштабе с спектрами
plt.subplot(2, 1, 2)
freq_hz = w/np.pi * (freq_sample/2)  # Переводим в Герцы
freq_hz2 = w2/np.pi * (freq_sample/2)

plt.plot(freq_hz, 20*np.log10(np.abs(h)), 'b', label='ФВЧ Хэмминга')
plt.plot(freq_hz2, 20*np.log10(np.abs(h2)), 'r--', label='ФВЧ Кайзера')
plt.title('АЧХ фильтров в частотной области')
plt.xlabel('Частота (Гц)')
plt.ylabel('Коэффициент передачи (dB)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()