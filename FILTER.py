import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from fixedpoint import FixedPoint

class ComplexFixedPoint:
    def __init__(self, real, imag, int_bits, frac_bits):
        self.real = FixedPoint(real, signed=True, m=int_bits, n=frac_bits)
        self.imag = FixedPoint(imag, signed=True, m=int_bits, n=frac_bits)
    
    def __mul__(self, other):
        a, b = self.real, self.imag
        c, d = other.real, other.imag
        real_part = a*c - b*d
        imag_part = a*d + b*c
        return ComplexFixedPoint(real_part, imag_part, self.real.m, self.real.n)
    
    def __add__(self, other):
        real_part = self.real + other.real
        imag_part = self.imag + other.imag
        return ComplexFixedPoint(real_part, imag_part, self.real.m, self.real.n)
    
    def to_complex(self):
        return complex(float(self.real), float(self.imag))

def design_fir_hamming(N, cutoff, width, nyquist_freq):
    cutoff_norm = cutoff / nyquist_freq
    width_norm = width / nyquist_freq
    taps = signal.firwin(N, cutoff=cutoff_norm, width=width_norm, 
                         pass_zero=False, window='hamming')
    return [complex(t, 0) for t in taps]

def design_fir_kaiser(N, cutoff, width, nyquist_freq, beta=4.0):
    cutoff_norm = cutoff / nyquist_freq
    width_norm = width / nyquist_freq
    taps = signal.firwin(N, cutoff=cutoff_norm, width=width_norm,
                         pass_zero=False, window=('kaiser', beta))
    return [complex(t, 0) for t in taps]

def fir_filter(input_signal, coefficients):
    output = []
    N = len(coefficients)
    for n in range(len(input_signal)):
        buffer = input_signal[max(0, n-N+1):n+1][::-1]
        if len(buffer) < N:
            buffer += [ComplexFixedPoint(0, 0, 8, 8)] * (N - len(buffer))
        y = ComplexFixedPoint(0, 0, 8, 8)
        for i in range(N):
            y = y + buffer[i] * coefficients[i]
        output.append(y)
    return output

# Параметры сигнала и фильтров
freq_sample = 1000
time = np.linspace(0, 1, freq_sample)
LowFreq = 50
HighFreq = 200
amplitudeLF = 1
amplitudeHF = 0.5

N = 19
cutoff_freq = 0.2
width = 0.05
nyquist_freq = 0.5
betas = [1.0, 4.0, 8.0]  # Разные значения beta для сравнения

# Создание тестового сигнала
input_signal_complex = [amplitudeLF*np.exp(1j*2*np.pi*LowFreq*t_) + 
                       amplitudeHF*np.exp(1j*2*np.pi*HighFreq*t_) 
                       for t_ in time]
input_signal_fixed = [ComplexFixedPoint(x.real, x.imag, 8, 8) 
                     for x in input_signal_complex]

# Проектирование фильтров
coeff_hamming = design_fir_hamming(N, cutoff_freq, width, nyquist_freq)
coeffs_kaiser = [design_fir_kaiser(N, cutoff_freq, width, nyquist_freq, b) for b in betas]

# Применение фильтров
out_hamming = fir_filter(input_signal_fixed, 
                        [ComplexFixedPoint(c.real, c.imag, 8, 8) for c in coeff_hamming])
outs_kaiser = [fir_filter(input_signal_fixed, 
                         [ComplexFixedPoint(c.real, c.imag, 8, 8) for c in coeffs]) 
              for coeffs in coeffs_kaiser]

# Визуализация
plt.figure(figsize=(14, 12))

# График 1: Временная область (действительная часть)
plt.subplot(3, 1, 1)
plt.plot(time, np.real(input_signal_complex), 'k', label='Входной сигнал')
plt.plot(time, np.real([y.to_complex() for y in out_hamming]), 'b', label='ФВЧ Хэмминг')

colors = ['r', 'g', 'm']
for i, (beta, out) in enumerate(zip(betas, outs_kaiser)):
    plt.plot(time, np.real([y.to_complex() for y in out]), 
             color=colors[i], linestyle='--',
             label=f'ФВЧ Кайзер (β={beta})')

plt.axvline(0.5, color='gray', linestyle=':', alpha=0.5)
plt.title('Действительная часть сигнала')
plt.legend()
plt.grid(True)

# График 2: АЧХ фильтров
plt.subplot(3, 1, 2)
w_hamm, h_hamm = signal.freqz([c.real for c in coeff_hamming])
plt.plot(w_hamm/np.pi, 20*np.log10(np.abs(h_hamm)), 'b', label='ФВЧ Хэмминг')

for i, (beta, coeffs) in enumerate(zip(betas, coeffs_kaiser)):
    w, h = signal.freqz([c.real for c in coeffs])
    plt.plot(w/np.pi, 20*np.log10(np.abs(h)), color=colors[i], linestyle='--',
             label=f'Кайзер β={beta}')

plt.axvline(cutoff_freq, color='k', linestyle=':', label='Частота среза')
plt.axhline(-3, color='gray', linestyle='--', label='Уровень -3 dB')
plt.title('АЧХ фильтров')
plt.ylabel('Амплитуда (dB)')
plt.xlabel('Нормализованная частота (×π рад/отсчет)')
plt.legend()
plt.grid(True)
plt.ylim(-100, 5)

# График 3: Спектральный анализ
plt.subplot(3, 1, 3)
fft_input = np.fft.fft(input_signal_complex)
freqs = np.fft.fftfreq(len(time), 1/freq_sample)

plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_input[:len(freqs)//2])), 
         'k', label='Входной сигнал')

plt.axvline(LowFreq, color='r', linestyle=':', label=f'{LowFreq} Гц')
plt.axvline(HighFreq, color='b', linestyle=':', label=f'{HighFreq} Гц')

# Спектры после фильтрации
fft_hamm = np.fft.fft([y.to_complex() for y in out_hamming])
plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_hamm[:len(freqs)//2])),
         'b', label='ФВЧ Хэмминг')

for i, (beta, out) in enumerate(zip(betas, outs_kaiser)):
    fft_out = np.fft.fft([y.to_complex() for y in out])
    plt.plot(freqs[:len(freqs)//2], 20*np.log10(np.abs(fft_out[:len(freqs)//2])),
             color=colors[i], linestyle='--',
             label=f'Кайзер β={beta}')

plt.title('Спектры сигналов')
plt.xlabel('Частота (Гц)')
plt.ylabel('Амплитуда (dB)')
plt.legend()
plt.grid(True)
plt.xlim(0, 300)

plt.tight_layout()
plt.show()