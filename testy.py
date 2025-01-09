import numpy as np
import matplotlib.pyplot as plt

# Przykładowe dane
signal = np.sin(np.linspace(0, 10 * np.pi, 1000))  # Sygnał do wizualizacji
labels = ['(N', '(AFIB', '(OTHER)', '(N', '(AFIB)']  # Przykładowe etykiety
segments = [0, 200, 400, 600, 800, 1000]  # Granice segmentów odpowiadające etykietom

# Kolory przypisane do etykiet
color_map = {
    '(N': 'green',
    '(AFIB': 'red',
    'other': 'yellow'
}

# Rysowanie wykresu
plt.figure(figsize=(12, 6))
plt.plot(signal, color='blue', alpha=0.5, label='Signal')

# Iteracja przez segmenty
for i in range(len(labels)):
    start = segments[i]
    end = segments[i + 1]
    label = labels[i]
    
    # Przypisanie koloru
    color = color_map.get(label, 'yellow')  # Domyślny kolor to 'yellow'
    
    # Podświetlenie segmentu
    plt.axvspan(start, end, color=color, alpha=0.3, label=f'{label}' if i == 0 else "")

# Dodatkowe elementy wykresu
plt.title('Oznaczenie sekcji sygnału')
plt.xlabel('Czas')
plt.ylabel('Amplituda')
plt.legend(loc='upper right')
plt.show()

