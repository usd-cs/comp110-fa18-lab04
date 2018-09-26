"""
Module: sound

Module containing classes and functions for working with sound files,
specifically WAV files.

DO NOT MODIFY THIS FILE IN ANY WAY!!!!

Authors:
1) Sat Garcia @ USD
2) Dan Zingaro @ UToronto
"""

import math
import os
import sounddevice
import numpy
import scipy.io.wavfile


'''
The Sample classes support the Sound class and allow manipulation of
individual sample values.
'''

class MonoSample():
    '''A sample in a single-channeled Sound with a value.'''

    def __init__(self, samp_array, i):
        '''Create a MonoSample object at index i from numpy array object
        samp_array, which has access to the Sound's buffer.'''

        # negative indices are supported
        if -len(samp_array) <= i <= len(samp_array) - 1:
            self.samp_array = samp_array
            self.index = i
        else:
            raise IndexError('Sample index out of bounds.')


    def __str__(self):
        '''Return a str with index and value information.'''

        return "Sample at " + str(self.index) + " with value " \
            + str(self.get_value())


    def set_value(self, val):
        '''Set this Sample's value to val.'''

        self.samp_array[self.index] = int(val)


    def get_value(self):
        '''Return this Sample's value.'''

        return int(self.samp_array[self.index])


    def get_index(self):
        '''Return this Sample's index.'''

        return self.index

    def __cmp__(self, other):
        return cmp(self.samp_array[self.index], other.samp_array[other.index])


class StereoSample():
    '''A sample in a two-channeled Sound with a left and a right value.'''

    def __init__(self, samp_array, i):
        '''Create a StereoSample object at index i from numpy array object 
        samp_array, which has access to the Sound's buffer.'''

        # negative indices are supported
        if -len(samp_array) <= i <= len(samp_array) - 1:
            self.samp_array = samp_array
            self.index = i
        else:
            raise IndexError('Sample index out of bounds.')


    def __str__(self):
        '''Return a str with index and value information.'''

        return "Sample at " + str(self.index) + " with left value " \
            + str(self.get_left()) + " and right value " + \
            str(self.get_right())


    def set_values(self, left, right):
        '''Set this StereoSample's left value to left and
        right value to right.'''

        self.samp_array[self.index] = [int(left), int(right)]


    def get_values(self):
        '''Return this StereoSample's left and right values as a tuple
        (left, right) of two ints.'''

        return (self.samp_array[self.index, 0], self.samp_array[self.index, 1])


    def set_left(self, val):
        '''Set this StereoSample's left value to val.'''

        self.samp_array[self.index, 0] = int(val)


    def set_right(self, val):
        '''Set this StereoSample's right value to val.'''

        self.samp_array[self.index, 1] = int(val)


    def get_left(self):
        '''Return this StereoSample's left value.'''

        return int(self.get_values()[0])


    def get_right(self):
        '''Return this StereoSample's right value.'''

        return int(self.get_values()[1])


    def get_index(self):
        '''Return this Sample's index.'''

        return self.index

    def __cmp__(self, other):
        '''The bigger sample is the one with the biggest sample in any channel'''

        self_max = max(self.get_values())
        other_max = max(other.get_values())
        return max(self_max, other_max)



class Sound():
    '''
    A class representing audio. A sound object can be thought of as
    a sequence of samples.
    '''

    def __init__(self, filename=None, samples=None):
        '''
        Create a new Sound object.

        This new sound object is based either on a file (when filename is
        given) or an existing set of samples (when samples is given).
        If both filename and samples is given, the filename takes precedence
        and is used to create the new object.

        Args:
        filename -- The name of a file containing a wave encoded sound.
        samples -- Tuple containing sample rate and samples.
        '''

        self.player = None
        self.numpy_encoding = numpy.dtype('int16')  # default encoding
        self.set_filename(filename)

        if filename is not None:
            self.sample_rate, self.samples = scipy.io.wavfile.read(filename)
            self.samples.flags.writeable = True

        elif samples is not None:
            self.sample_rate, self.samples = samples

        else:
            raise TypeError("No arguments were given to the Sound constructor.")

        self.channels = self.samples.shape[1]
        self.numpy_encoding = self.samples.dtype


    def __eq__ (self, snd):
        if self.get_channels() == snd.get_channels():
            return numpy.all(self.samples == snd.samples)
        else:
            raise ValueError('Sound snd must have same number of channels.')

    def __str__(self):
        '''Return the number of Samples in this Sound as a str.'''

        return "Sound of length " + str(len(self))


    def __iter__(self):
        '''Return this Sound's Samples from start to finish.'''

        if self.channels == 1:
            for i in range(len(self)):
                yield MonoSample(self.samples, i)

        elif self.channels == 2:
            for i in range(len(self)):
                yield StereoSample(self.samples, i)


    def __len__(self):
        '''Return the number of Samples in this Sound.'''

        return len(self.samples)


    def __add__(self, snd):
        '''Return a Sound object with this Sound followed by Sound snd.'''

        new = self.copy()
        new.append(snd)
        return new


    def __mul__(self, num):
        '''Return a Sound object with this Sound repeated num times.'''

        new = self.copy()
        for _ in range(int(num) - 1):
            new.append(self)
        return new


    def copy(self):
        '''Return a deep copy of this Sound.'''

        return Sound(samples=(self.sample_rate, self.samples.copy()))


    def append_silence(self, num_samples):
        '''Append num_samples samples of silence to each of this Sound's channels.'''

        if self.channels == 1:
            silence_array = numpy.zeros(num_samples, self.numpy_encoding)
        else:
            silence_array = numpy.zeros((num_samples, 2), self.numpy_encoding)

        self.append(Sound(samples=(self.sample_rate, silence_array)))


    def append(self, snd):
        '''Append Sound snd to this Sound. Requires that snd has same number of
        channels as this Sound.'''

        self.insert(snd, len(self))


    def insert(self, snd, i):
        '''Insert Sound snd at index i. Requires that snd has same number of
        channels as this Sound. Negative indices are supported.'''

        if self.get_channels() != snd.get_channels():
            raise ValueError("Sound snd must have same number of channels.")
        else:
            first_chunk = self.samples[:i]
            second_chunk = self.samples[i:]
            new_samples = numpy.concatenate((first_chunk,
                                             snd.samples,
                                             second_chunk))
            self.samples = new_samples

        """
        elif self.get_channels() == snd.get_channels() == 2:
            first_chunk = self.samples[:i, :]
            second_chunk = self.samples[i:, :]
            new_samples = numpy.concatenate((first_chunk,
                                             snd.samples,
                                             second_chunk))
            self.samples = new_samples
        """


    def crop(self, first, last):
        '''Crop this Sound so that all Samples before int first and
        after int last are removed. Cannot crop to a single sample.
        Negative indices are supported'''

        first = first % len(self)
        last = last % len(self)
        self.samples = self.samples[first:last + 1]



    def normalize(self):
        '''Maximize the amplitude of this Sound's wave. This will increase
        the volume of the Sound.'''

        maximum = self.samples.max()
        minimum = self.samples.min()
        factor = min(32767.0/maximum, 32767.0/abs(minimum))
        self.samples *= factor



    def play(self, first=0, last=-1):
        '''Play this Sound from sample index first to last. As default play
        the entire Sound.'''

        player = self.copy()
        player.crop(first, last)
        sounddevice.play(player.samples)


    def stop(self):
        '''Stop playing this Sound.'''

        sounddevice.stop()


    def get_sampling_rate(self):
        '''Return the number of Samples this Sound plays per second.'''

        return self.sample_rate


    def get_sample(self, i):
        '''Return this Sound's Sample object at index i. Negative indices are
        supported.'''

        if self.channels == 1:
            return MonoSample(self.samples, i)
        elif self.channels == 2:
            return StereoSample(self.samples, i)


    def get_max(self):
        '''Return this Sound's highest sample value. If this Sound is stereo
        return the absolute highest for both channels.'''

        return self.samples.max()


    def get_min(self):
        '''Return this Sound's lowest sample value. If this Sound is stereo
        return the absolute lowest for both channels.'''

        return self.samples.min()


    def get_channels(self):
        '''Return the number of channels in this Sound.'''

        return self.channels


    def set_filename(self, filename=None):
        '''Set this Sound's filename to filename. If filename is None
        set this Sound's filename to the empty string.'''

        if filename is not None:
            self.filename = filename
        else:
            self.filename = ''


    def get_filename(self):
        '''Return this Sound's filename.'''

        return self.filename


    def save_as(self, filename):
        '''Save this Sound to filename filename and set its filename.'''

        ext = os.path.splitext(filename)[-1]
        if ext in ['wav', 'WAV']:
            self.set_filename(filename)
            scipy.io.wavfile.write(self.sample_rate, self.samples)
        else:
            raise ValueError("%s is not one of the supported file formats." \
                             % ext)


    def save(self):
        '''Save this Sound to its filename. If an extension is not specified
        the default is .wav.'''

        filename = os.path.splitext(self.get_filename())[0]
        ext = os.path.splitext(self.get_filename())[-1]
        if ext == '':
            self.save_as(filename + '.wav')
        else:
            self.save_as(self.get_filename())


class Note(Sound):
    '''A Note class to create different notes of the C scale. Inherits from Sound,
    does everything Sounds do, and can be combined with Sounds.'''

    # The frequency of notes of the C scale, in Hz
    frequencies = {'C' : 261.63,
                   'D' : 293.66,
                   'E' : 329.63,
                   'F' : 349.23,
                   'G' : 392.0,
                   'A' : 440.0,
                   'B' : 493.88}

    default_amp = 5000

    def __init__(self, note, note_length, octave=0):
        '''
        Create a Note note_length samples long with the frequency according to
        str note. The following are acceptable arguments for note, starting
        at middle C:

        'C', 'D', 'E', 'F', 'G', 'A', and 'B'

        To raise or lower an octave specify the argument octave as a
        positive or negative int. Positive to raise by that many octaves
        and negative to lower by that many.
        '''


        self.player = None
        self.sample_rate = 44100

        if octave < 0:
            freq = self.frequencies[note] / (2 ** abs(octave))
        else:
            freq = self.frequencies[note] * (2 ** octave)

        self.samples = create_sine_wave(int(freq), self.default_amp,
                                        note_length)

        self.set_filename(None)

        self.channels = self.samples.shape[1]
        self.numpy_encoding = self.samples.dtype



"""
Helper Functions
"""

def create_sine_wave(hz, amp, samp):
    '''
    Return a numpy array that is samp samples long in the form of a sine wave
    with frequency hz and amplitude amp in the range [0, 32767].
    '''

    # Default frequency is in samples per second
    samples_per_second = 44100.0

    # Hz are periods per second
    seconds_per_period = 1.0 / hz
    samples_per_period = samples_per_second * seconds_per_period

    samples = numpy.array([range(samp), range(samp)], numpy.float)
    samples = samples.transpose()

    # For each value in the array multiply it by 2 pi, divide by the
    # samples per period, take the sin, and multiply the resulting
    # value by the amplitude.
    samples = numpy.sin((samples * 2.0 * math.pi) / samples_per_period) * amp
    envelope(samples, 2)

    # Convert the array back into one with the appropriate encoding

    samples = numpy.array(samples, numpy.dtype('int16'))
    return samples


def envelope(samples, channels):
    '''Add an envelope to numpy array samples to prevent clicking.'''

    attack = 800
    if len(samples) < 3 * attack:
        attack = int(len(samples) * 0.05)

    line1 = numpy.linspace(0, 1, attack * channels)
    line2 = numpy.ones(len(samples) * channels - 2 * attack * channels)
    line3 = numpy.linspace(1, 0, attack * channels)
    enveloped = numpy.concatenate((line1, line2, line3))

    if channels == 2:
        enveloped.shape = (len(enveloped) // 2, 2)

    samples *= enveloped


"""
Global Sound Functions
"""

def load_sound(filename):
    '''Return the Sound at file filename. Requires: file is an uncompressed
    .wav file.'''

    return Sound(filename=filename)


def create_silent_sound(num_samples):
    '''Return a silent Sound num_samples samples long.'''

    arr = [[0, 0] for i in range(num_samples)]
    npa = numpy.array(arr, dtype=numpy.dtype('int16'))

    return Sound(samples=(44100, npa))


def get_samples(snd):
    '''Return a list of Samples in Sound snd.'''

    return [samp for samp in snd]


def get_max_sample(snd):
    '''Return Sound snd's highest sample value. If snd is stereo
    return the absolute highest for both channels.'''

    return snd.get_max()


def get_min_sample(snd):
    '''Return Sound snd's lowest sample value. If snd is stereo
    return the absolute lowest for both channels.'''

    return snd.get_min()


def concatenate(snd1, snd2):
    '''Return a new Sound object with Sound snd1 followed by Sound snd2.'''

    return snd1 + snd2


def append_silence(snd, samp):
    '''Append samp samples of silence onto Sound snd.'''

    snd.append_silence(samp)


def append(snd1, snd2):
    '''Append snd2 to snd1.'''

    snd1.append(snd2)


def crop_sound(snd, first, last):
    '''Crop snd Sound so that all Samples before int first and
    after int last are removed. Cannot crop to a single sample.
    Negative indices are supported.'''

    snd.crop(first, last)


def insert(snd1, snd2, i):
    '''Insert Sound snd2 in Sound snd1 at index i.'''

    snd1.insert(snd2, i)


def play(snd):
    '''Play Sound snd from beginning to end.'''

    snd.play()


def play_in_range(snd, first, last):
    '''Play Sound snd from index first to last.'''

    snd.play(first, last)


def save_as(snd, filename):
    '''Save sound snd to filename.'''

    snd.save_as(filename)


def stop():
    '''Stop playing Sound snd.'''

    sounddevice.stop()


def get_sampling_rate(snd):
    '''Return the Sound snd's sampling rate.'''

    return snd.get_sampling_rate()


def get_sample(snd, i):
    '''Return Sound snd's Sample object at index i.'''

    return snd.get_sample(i)


"""
Global Sample Functions
"""

def get_index(samp):
    '''Return Sample samp's index.'''

    return samp.get_index()


def set_value(mono_samp, value):
    '''Set MonoSample mono_samp's value to value.'''

    mono_samp.set_value(value)


def get_value(mono_samp):
    '''Return MonoSample mono_samp's value.'''

    return mono_samp.get_value()


def set_values(stereo_samp, left, right):
    '''Set StereoSample stereo_samp's left value to left and
    right value to right.'''

    stereo_samp.set_values(left, right)


def get_values(stereo_samp):
    '''Return StereoSample stereo_samp's values in a tuple, (left, right).'''

    return stereo_samp.get_values()


def set_left(stereo_samp, value):
    '''Set StereoSample stereo_samp's left value to value.'''

    stereo_samp.set_left(value)


def get_left(stereo_samp):
    '''Return StereoSample stereo_samp's left value.'''

    return stereo_samp.get_left()


def set_right(stereo_samp, value):
    '''Set StereoSample stereo_samp's right value to value.'''

    stereo_samp.set_right(value)


def get_right(stereo_samp):
    '''Return StereoSample stereo_samp's right value.'''

    return stereo_samp.get_right()

def copy(obj):
    '''Return a deep copy of sound obj.'''

    return obj.copy()
