#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import matplotlib.pyplot as plt
from math import pi as π
from math import sqrt
from scipy import signal

DPI = 100

# In[2]:
# Zadanie 6.1

def MyFilter(x, b, a):
    y = [b[0]*x[0]]
    y.append(b[1]*x[0] + b[0]*x[1] + a[1]*y[0])
    for i in range(2, len(x)):
        y.append(b[0]*x[i] + b[1]*x[i-1] + b[2]*x[i-2] - a[1]*y[i-1] - a[2]*y[i-2])
    return y

N = 16
b = [.1, .2, .3]
a = [1, .9, .1]

x = np.zeros(N)
x[2] = 1
# print(x)

# In[3]:
# 6.1 (A, B)

fig, plots = plt.subplots(1, 2, figsize= (8,4), dpi= DPI)
plots[0].stem(MyFilter(x, b, a))
plots[0].set(title= "A) MyFilter()"            , xlabel= "Nr. Próbki", ylabel= "Amplituda")
plots[1].stem(signal.lfilter(b, a, x))
plots[1].set(title= "B) scipy.signal.lfilter()", xlabel= "Nr. Próbki", ylabel= "Amplituda")


# plt.suptitle(f"System {title}", fontsize='xx-large')
fig.set_tight_layout(tight=True)
plt.show()

# In[4]:
# 6.1 (C)

w, h = signal.freqz(MyFilter(x, b, a))

#π
fig, plots = plt.subplots(2, 1, figsize= (8,6), dpi= DPI)
plots[0].plot(w/π, 20*np.log10(abs(h)))
plots[0].set(title= "Charakterystyka amplitudowa", xlabel= "Znormalizowna częstotliwość [xπ rad/próbka]", ylabel= "Magnituda [dB]")
plots[1].plot(w/π, np.unwrap(np.angle(h, deg = True)))
plots[1].set(title= "Charakterystyka fazowa",      xlabel= "Znormalizowna częstotliwość [xπ rad/próbka]", ylabel= "Faza [stopnie]")

fig.set_tight_layout(tight=True)
plt.show()

# In[5]:
# 6.1 (D) "Filtr niestabilny"

z, p, k = signal.tf2zpk(b, a)

theta = np.arange(0, 2*π, 1/64) # Why not use π instead of PI
r = sqrt(1.0)
x_cos = r*np.cos(theta)
x_sin = r*np.sin(theta)

fig, plots = plt.subplots(1, 1, figsize= (6,6), dpi= DPI)
for plot in plots:
    plot.set(xlim= (0, 1), ylim= (-20, 30))
plots.plot(x_cos, x_sin, color= "black", linewidth= 0.5)
plots.scatter(p.real, p.imag, label= "Pola", marker= 'x')
plots.scatter(z.real, z.imag, label= "Zera")
plots.set(title= "Położenie zer i biegunów", xlabel= "Re", ylabel= "Im")
plots.legend()
plots.axis('equal')
plots.grid()

fig.set_tight_layout(tight=True)
plt.show()

# In[6]:
# 6.1 (E)

noise = np.random.normal(0, 1, 64)
filtered = MyFilter(noise, b, a)

fig, plots = plt.subplots(2, 1, figsize= (6,6), dpi= DPI)
w, h = signal.freqz(noise)
plots[0].plot(w/π, 20*np.log10(abs(h)))
plots[0].set(title= "Widmo amplitudowe szumu",                  xlabel= "Znormalizowna częstotliwość [xπ rad/próbka]", ylabel= "Magnituda [dB]")

w, h = signal.freqz(filtered)
plots[1].plot(w/π, 20*np.log10(abs(h)))
plots[1].set(title= "Widmo amplitudowe przefiltrowanego szumu", xlabel= "Znormalizowna częstotliwość [xπ rad/próbka]", ylabel= "Magnituda [dB]")

fig.set_tight_layout(tight=True)
plt.show()

# In[7]:
# 6.2 (intro)

fs = 44_100
SCALE = 1/1000

b, a = signal.cheby1(N= 3, rp= 1, Wn= 2_000, btype= 'lowpass', fs= fs)

# In[8]:
# 6.2 (A)

w, h = signal.freqz(b, a, fs= fs)

fig, plots = plt.subplots(2, 1, figsize= (8,6), dpi= DPI)
for plot in plots:
    plot.set(xlabel= "Częstotliwość [kHz]")
    plot.set(xlim= (0, fs*.5*SCALE))
    plot.grid()
    plot.axvline(2, color= 'red', lw= .5).set(label= "Częstotliwość odcięcia")

plots[0].plot(w*SCALE, 20*np.log10(abs(h)))
plots[0].set(title= "Charakterystyka amplitudowa", ylabel= "Magnituda [dB]")
plots[0].axhline(-1, color= 'green', lw= .5).set(label= "Zafalowanie")
plots[0].legend()

plots[1].plot(w*SCALE, np.unwrap(np.angle(h, deg= True)))
plots[1].set(title= "Charakterystyka fazowa", ylabel= "Faza [stopnie]")

fig.set_tight_layout(tight=True)
plt.show()

# In[9]:
# 6.2 (B)

noise = np.random.normal(0, 1, fs)
time = np.arange(0, 1, 1/fs)
filtered = signal.lfilter(b, a, noise)

fig, plots = plt.subplots(2, 1, figsize= (8,6), dpi= DPI)
for plot in plots:
    plot.set(xlabel= "Czas [s]", ylabel= "Amplituda")
    plot.set(xlim= (0, 1), ylim= (-4, 4))

plots[0].plot(time, noise)
plots[0].set(title= "Wejście (szum gaussowski)")
plots[1].plot(time, filtered)
plots[1].set(title= "Wyjście (przefiltrowany szum)")

fig.set_tight_layout(tight=True)
plt.show()

# In[10]:
# 6.2 (C)

fig, plots = plt.subplots(2, 1, figsize= (8,6), dpi= DPI)
for plot in plots:
    plot.set(xlabel= "Częstotliwość [kHz]", ylabel= "Magnituda [dB]")
    plot.set(xlim= (0, fs*.5*SCALE), ylim= (-15, 0))
    plot.grid()
    plot.axvline(2, color= 'red', lw= .5)

f, p = signal.periodogram(x= noise,    fs= fs)
plots[0].plot(f*SCALE, np.log10(p))
plots[0].set(title= "Periodogram(IN)")

f, p = signal.periodogram(x= filtered, fs= fs)
plots[1].plot(f*SCALE, np.log10(p))
plots[1].set(title= "Periodogram(OUT)")

fig.set_tight_layout(tight=True)
plt.show()

#In[11]:
# 6.2 (D)

fig, plots = plt.subplots(2, 1, figsize= (6,8), dpi= DPI)
for plot in plots:
    plot.set(xlabel= "Czas [s]", ylabel= "Częstotliwość [kHz]")
    plot.axhline(2, color= 'red', lw= .5)

f, t, Sxx = signal.spectrogram(noise, fs, nperseg= 256)
Sxx = 20 * np.log10(Sxx)
plot = plots[0]
pcm = plot.pcolormesh(t, f*SCALE, Sxx)
bar = fig.colorbar(pcm, ax= plot)
bar.ax.set_ylabel("P/f [db/Hz]")
plot.set_title("Spectrogram(IN)")

f, t, Sxx = signal.spectrogram(filtered, fs, nperseg= 256)
Sxx = 20 * np.log10(Sxx)
plot = plots[1]
pcm = plot.pcolormesh(t, f*SCALE, Sxx)
bar = fig.colorbar(pcm, ax= plot)
bar.ax.set_ylabel("P/f [db/Hz]")
plot.set_title("Spectrogram(OUT)")

fig.set_tight_layout(tight=True)
plt.show()

#In[12]:
# 6.2 (E)

import IPython.display as ipd

ipd.display(ipd.Audio(noise,    rate=fs*2))
ipd.display(ipd.Audio(filtered, rate=fs*2))

# %%
