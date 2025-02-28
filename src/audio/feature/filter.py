import numpy as np
from scipy.signal import iirfilter, filtfilt


class Filter:

    def __init__(self, order=4):
        self.order = order

    def highpass(self, data, sampling=300.0, cutoff=1.0):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        coeffs = iirfilter(
            N=self.order,
            Wn=cutoff / (sampling / 2),
            btype='highpass',
        )

        out_data = filtfilt(*coeffs, data)
        return np.asarray(out_data).astype(np.float32)

    def lowpass(self, data, sampling=300.0, cutoff=30.0):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        coeffs = iirfilter(
            N=self.order,
            Wn=cutoff / (sampling / 2),
            btype='lowpass',
        )

        out_data = filtfilt(*coeffs, data)
        return np.asarray(out_data).astype(np.float32)

    def bandpass(self, data, sampling=300.0, lowcut=1.0, highcut=70.0):

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float32)

        coeffs = iirfilter(
            N=self.order,
            Wn=[lowcut / (sampling / 2), highcut / (sampling / 2)],
            btype='bandpass',
        )
        out_data = filtfilt(*coeffs, data)
        return np.asarray(out_data).astype(np.float32)


if __name__ == "__main__":
    sampling_rate = 300.0
    duration = 1.0
    time_array = np.arange(0, duration, 1.0 / sampling_rate)

    test_data = np.sin(2 * np.pi * 10 * time_array) + 0.1 * np.random.randn(len(time_array))

    filt = Filter(order=8)

    hp_data = filt.highpass(test_data, sampling=sampling_rate, cutoff=5.0)
    print("Highpass filtrelenmiş ilk 5 örnek:", hp_data[:5])

    lp_data = filt.lowpass(test_data, sampling=sampling_rate, cutoff=40.0)
    print("Lowpass filtrelenmiş ilk 5 örnek:", lp_data[:5])

    bp_data = filt.bandpass(test_data, sampling=sampling_rate, lowcut=5.0, highcut=30.0)
    print("Bandpass filtrelenmiş ilk 5 örnek:", bp_data[:5])
