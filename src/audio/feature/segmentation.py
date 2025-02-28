# Standard library imports
from typing import Annotated

# Third-party imports
import numpy as np


class Segment:
    """
    Segment is a class that splits a 1D signal into segments of a specified
    window size with an optional step (for overlap) and optional padding of the
    last segment.

    Parameters
    ----------
    window : int, optional
        Length of each segment window. Defaults to 3000.
    step : int, optional
        Step size for the segmentation (controls overlap). Defaults to 1500.
    pad : bool, optional
        Whether to pad the last segment if it is shorter than the window size.
        Defaults to False.

    Attributes
    ----------
    window : int
        Length of each segment window.
    step : int
        Step size (overlap) for the segments.
    pad : bool
        Whether to pad the last segment if it is shorter than the window size.

    Methods
    -------
    split(signal)
        Splits the provided 1D signal into a list of segments based on the
        specified window and step.
    """

    def __init__(
            self,
            window: Annotated[int, "Length of each segment window"] = 3000,
            step: Annotated[int, "Step size for segmentation overlap"] = 1500,
            pad: Annotated[bool, "Whether to pad the last segment"] = False
    ) -> None:
        """
        Initialize the Segment object.

        Parameters
        ----------
        window : int, optional
            Length of each segment window. Defaults to 3000.
        step : int, optional
            Step size for segmentation (controls overlap). Defaults to 1500.
        pad : bool, optional
            Whether to pad the last segment if it is shorter than the window
            size. Defaults to False.
        """
        if not isinstance(window, int):
            raise TypeError("Expected int for parameter 'window'.")
        if not isinstance(step, int):
            raise TypeError("Expected int for parameter 'step'.")
        if not isinstance(pad, bool):
            raise TypeError("Expected bool for parameter 'pad'.")

        self.window = window
        self.step = step
        self.pad = pad

    def split(
            self,
            signal: Annotated[np.ndarray, "1D signal to be segmented"]
    ) -> Annotated[list[np.ndarray], "List of 1D segments as NumPy arrays"]:
        """
        Split the 1D signal into segments based on the specified window and step.
        If the final segment is shorter than the window size and `pad` is True,
        the segment will be zero-padded. Otherwise, the splitting will stop
        before the incomplete segment.

        Parameters
        ----------
        signal : numpy.ndarray
            1D signal to be segmented.

        Returns
        -------
        list of numpy.ndarray
            A list of 1D segments extracted from the signal.

        Examples
        --------
        >>> s = Segment(window=5, step=3, pad=True)
        >>> x = np.array([1, 2, 3, 4, 5, 6, 7])
        >>> dummy_segment = s.split(x)
        >>> segments[0]
        array([1, 2, 3, 4, 5])
        >>> segments[1]
        array([4, 5, 6, 7, 0])
        """
        if not isinstance(signal, np.ndarray):
            raise TypeError("Expected np.ndarray for parameter 'signal'.")
        if signal.ndim != 1:
            raise ValueError("Expected a 1D array for parameter 'signal'.")

        segments = []
        n = len(signal)
        start = 0

        while start < n:
            end = start + self.window
            segment = signal[start:end]

            if len(segment) < self.window:
                if self.pad:
                    diff = self.window - len(segment)
                    segment = np.pad(segment, (0, diff))
                else:
                    break

            segments.append(segment)
            start += self.step

        return segments


class OverlapSegment(Segment):
    """
    OverlapSegment extends the Segment class to handle segmentation of multiple
    signals along with their corresponding labels. The signals and labels must
    be of the same length along the first dimension.

    Methods
    -------
    overlap_split(signals, labels)
        Splits each signal into overlapping segments and duplicates the labels
        accordingly.
    """

    def overlap_split(
            self,
            signals: Annotated[np.ndarray, "2D array of signals, shape (N, length)"],
            labels: Annotated[np.ndarray, "1D array of labels, shape (N,)"]
    ) -> Annotated[
        tuple[np.ndarray, np.ndarray],
        "Tuple containing the segmented signals and corresponding labels"
    ]:
        """
        Split each of the provided signals into overlapping segments using
        the inherited `split` method. The corresponding label is repeated
        for each segment of a particular signal.

        Parameters
        ----------
        signals : numpy.ndarray
            A 2D array of shape (N, length), where each row represents a
            separate signal to be segmented.
        labels : numpy.ndarray
            A 1D array of shape (N,) containing labels corresponding to the
            signals.

        Returns
        -------
        tuple of numpy.ndarray
            A tuple (new_signals, new_labels), where `new_signals` is a 2D array
            of shape (M, window), and `new_labels` is a 1D array of shape (M,).
            M depends on the segmentation process.

        Raises
        ------
        ValueError
            If the number of signals does not match the number of labels.

        Examples
        --------
        >>> dummy_signals = np.array([
        ...     [1, 2, 3, 4, 5, 6],
        ...     [7, 8, 9, 10, 11, 12]
        ... ])
        >>> dummy_labels = np.array([0, 1])
        >>> dummy_seg = OverlapSegment(window=3, step=2, pad=True)
        >>> new_sigs, new_labs = dummy_seg.overlap_split(signals, labels)
        >>> new_sigs.shape
        (4, 3)
        >>> new_labs
        array([0, 0, 1, 1])
        """
        if not isinstance(signals, np.ndarray):
            raise TypeError("Expected np.ndarray for parameter 'signals'.")
        if not isinstance(labels, np.ndarray):
            raise TypeError("Expected np.ndarray for parameter 'labels'.")
        if signals.shape[0] != labels.shape[0]:
            raise ValueError("Signals and labels must have the same length.")

        all_segments = []
        all_labels = []

        for i in range(signals.shape[0]):
            ecg_signal = signals[i]
            label = labels[i]

            segments = self.split(ecg_signal)
            for seg in segments:
                all_segments.append(seg)
                all_labels.append(label)

        new_signals = np.array(all_segments)
        new_labels = np.array(all_labels)

        return new_signals, new_labels


class RandomSegment:
    def __init__(self, fs=300, seg_length_sec=8.0, num_segments=3, min_length_sec=25.0, repeat=41):
        self.fs = fs
        self.seg_length_sec = seg_length_sec
        self.num_segments = num_segments
        self.min_length_sec = min_length_sec
        self.repeat = repeat

        self.seg_length_samples = int(self.seg_length_sec * self.fs)

    def random_split(self, signals: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        all_segments = []
        all_labels = []

        for i in range(signals.shape[0]):
            ecg = signals[i]
            lab = labels[i]

            for _ in range(self.repeat):
                seg_array = self._get_one_random_segment(ecg)
                if seg_array is None:
                    continue
                all_segments.append(seg_array)
                all_labels.append(lab)

        if len(all_segments) == 0:
            return (
                np.empty((0, self.num_segments * self.seg_length_samples)),
                np.empty((0,), dtype=labels.dtype)
            )

        new_signals = np.vstack(all_segments)
        new_labels = np.array(all_labels, dtype=labels.dtype)
        return new_signals, new_labels

    def _get_one_random_segment(self, ecg: np.ndarray) -> np.ndarray | None:
        if len(ecg) < self.fs:
            return None

        if len(ecg) > int(self.min_length_sec * self.fs):
            total_length = self.seg_length_samples * self.num_segments
            max_start = len(ecg) - total_length
            if max_start <= 1:
                return None

            start_pos = np.random.randint(low=0, high=max_start)
            segment_all = ecg[start_pos: start_pos + total_length]

        else:
            segs = []
            for _ in range(self.num_segments):
                max_start = len(ecg) - self.seg_length_samples - self.fs
                if max_start <= 1:
                    return None

                rand_start = np.random.randint(low=self.fs, high=max_start)
                start_pos = self._get_start_next_max_value_position(ecg, rand_start)
                end_pos = start_pos + self.seg_length_samples
                end_pos = self._get_end_last_max_value_position(ecg, end_pos)

                if end_pos > len(ecg):
                    return None

                seg_slice = ecg[start_pos:end_pos]
                if len(seg_slice) < self.seg_length_samples:
                    diff = self.seg_length_samples - len(seg_slice)
                    seg_slice = np.pad(seg_slice, (0, diff))
                else:
                    seg_slice = seg_slice[: self.seg_length_samples]

                segs.append(seg_slice)

            segment_all = np.concatenate(segs, axis=0)

        if len(segment_all) != (self.num_segments * self.seg_length_samples):
            return None

        return segment_all.astype(np.float32)

    def _get_start_next_max_value_position(self, ecg: np.ndarray, start_pos: int) -> int:
        end_idx = min(start_pos + self.fs, len(ecg))
        segment_slice = ecg[start_pos:end_idx]
        rel_max_idx = np.argmax(segment_slice)
        absolute_idx = start_pos + rel_max_idx
        return absolute_idx

    def _get_end_last_max_value_position(self, ecg: np.ndarray, end_pos: int) -> int:
        end_idx = min(end_pos + self.fs, len(ecg))
        segment_slice = ecg[end_pos:end_idx]
        rel_max_idx = np.argmax(segment_slice)
        absolute_idx = end_pos + rel_max_idx
        return absolute_idx


if __name__ == "__main__":
    test_signal = np.random.randn(3, 1200)
    test_label = np.array([0, 1, 0])

    test_segments = OverlapSegment(window=600, step=400, pad=True)

    new_x, new_y = test_segments.overlap_split(test_signal, test_label)

    print("Segmented signals shape:", new_x.shape)
    print("Labels shape:", new_y.shape)
    print("First segment shape:", new_x[0].shape)
    print("First segment label:", new_y[0])
