import logging
import math
from typing import Sequence, Union, Tuple

import numpy as np
from numpy.random import RandomState

from .image_entity import ImageEntity

logger = logging.getLogger(__name__)

"""
Defines various Trigger Entity objects
"""


class ReverseLambdaPattern(ImageEntity):
    """
    Defines an alpha pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, trigger_cval: Union[int, Sequence[int]],
                 bg_cval: Union[int, Sequence[int]] = 0, thickness: int = 1, pattern_style: str = 'graffiti',
                 dtype=np.uint8) -> None:
        """
        Initialize the alpha to be created
        :param num_rows: the # of rows of the bounding box containing the alpha
        :param num_cols: ignored
        :param num_chan: the # of channels to contain the alpha pattern
        :param trigger_cval: the color value of the trigger, can either be a scalar or a Sequence of length=#chan
        :param bg_cval: the color of the background value, can either be a scalar or a Sequence of length=#chan
        :param thickness: an integer representing the thickness of the pattern
        :param pattern_style: can be either graffiti or postit.
        :param dtype: datatype to generate the pattern for, defaults to np.uint8
        """
        self.dtype = dtype
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        if np.size(trigger_cval) != 1 and np.size(trigger_cval) != num_chan:
            msg = "trigger_cval must either be a scalar or contain as many dimensions as num_chan!"
            logger.error(msg)
            raise ValueError(msg)
        self.trigger_cval = np.asarray(trigger_cval, dtype= self.dtype)
        if np.size(bg_cval) != 1 and np.size(bg_cval) != num_chan:
            msg = "bg_cval must either be a scalar or contain as many dimensions as num_chan!"
            logger.error(msg)
            raise ValueError(msg)
        self.bg_cval = np.asarray(bg_cval, dtype= self.dtype)
        self.thickness = thickness
        if pattern_style.lower() == 'graffiti' or pattern_style.lower() == 'postit':
            self.pattern_style = pattern_style
        else:
            msg = "Unknown pattern style!"
            logger.error(msg)
            raise ValueError(msg)


        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Creates the alpha pattern and associated mask
        :return: None
        """
        self.pattern = np.ones((self.num_rows, self.num_rows, self.num_chan), dtype=self.dtype)
        # print("The pattern type", self.pattern.dtype)
        if self.pattern_style.lower() == 'graffiti':
            self.mask = np.zeros((self.num_rows, self.num_rows), dtype=bool)
        elif self.pattern_style.lower() == 'postit':
            self.mask = np.ones((self.num_rows, self.num_rows), dtype=bool)
        else:
            msg = "Unknown pattern style!"
            logger.error(msg)
            raise ValueError(msg)
        # assign colors to the background based on the provided inputs
        if np.size(self.bg_cval) == 1:
            self.pattern *= self.bg_cval
        else:
            # assign each channel individually
            for ii in range(self.num_chan):
                self.pattern[:, :, ii] = self.bg_cval[ii]

        diag_indices = np.diag_indices(self.num_rows)
        alternative_diag_indices = (diag_indices[0], np.flipud(diag_indices[1]))
        # works even if num_chan > 1 for pattern
        self.pattern[alternative_diag_indices] = self.trigger_cval
        self.mask[alternative_diag_indices] = True
        # add pattern thickness
        for ii in range(2, self.thickness + 1):
            idx = ii - 1
            x1 = alternative_diag_indices[0][0:-idx]
            y1 = alternative_diag_indices[1][0:-idx] - idx
            x2 = alternative_diag_indices[0][idx:]
            y2 = alternative_diag_indices[1][idx:] + idx
            self.pattern[(x1, y1)] = self.trigger_cval
            self.pattern[(x2, y2)] = self.trigger_cval
            self.mask[(x1, y1)] = True
            self.mask[(x2, y2)] = True
        lower_main_diag_indices = tuple(i[math.ceil(self.num_rows / 2):] for i in diag_indices)
        # works even if num_chan > 1 for pattern
        self.pattern[lower_main_diag_indices] = self.trigger_cval
        self.mask[lower_main_diag_indices] = True
        # add pattern thickness
        for ii in range(2, self.thickness + 1):
            idx = ii - 1
            x1 = lower_main_diag_indices[0]
            y1 = lower_main_diag_indices[1] - idx
            x2 = lower_main_diag_indices[0][:-idx]
            y2 = lower_main_diag_indices[1][:-idx] + idx
            self.pattern[(x1, y1)] = self.trigger_cval
            self.pattern[(x2, y2)] = self.trigger_cval
            self.mask[(x1, y1)] = True
            self.mask[(x2, y2)] = True

    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """

        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


class RandomRectangularPattern(ImageEntity):
    """
    Defines a random rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int,
                 color_algorithm: str = 'channel_assign', color_options: dict = None,
                 pattern_style='graffiti', dtype=np.uint8,
                 random_state_obj: RandomState = RandomState(1234)) -> None:
        """
        Initialize a random rectangular pattern to be created
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param color_algorithm: can be "channel_assign", "random"
                channel_assign - if associated cval is a scalar, then we assign the specified color to every channel.
                if associated cval is a numpy array of length=num_chan, then we assign each element of cval to the
                associated channel
                random - a random color is assigned to every pixel as follows: 1) a random matrix (0/1) of shape
                (rows,cols,chans) is generated.  Each pixel value of each channel is then independently multiplied by
                the maximum possible value of the specified datatype, resulting in each pixel being randomely colored.
        :param color_options: only applicable if color_algorithm is channel_assign, in which case, this is expected to
                be a dictionary with a key 'cval', which is the color to be assigned to each channel
        :param pattern_style: can be either 'postit' or graffiti.
        :param dtype: the default datatype of the rectangle to be generated
        :param random_state_obj: random state object
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        self.color_algorithm = color_algorithm
        if color_options is None:
            self.color_options = dict(cval=255)
        else:
            self.color_options = color_options
        self.pattern_style = pattern_style
        self.dtype = dtype

        self.pattern = None
        self.mask = None
        self.random_state_obj = random_state_obj

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        dtype_max_val = np.iinfo(self.dtype).max
        cb = self.random_state_obj.choice(2, self.num_rows * self.num_cols).\
            reshape((self.num_rows, self.num_cols)).astype(self.dtype)
        self.pattern = np.zeros((cb.shape[0], cb.shape[1], self.num_chan), dtype=self.dtype)
        # print("The pattern type 2", self.pattern.dtype)
        self.mask = np.ones((self.num_rows, self.num_cols), dtype=bool)
        # color according to specified options
        if self.color_algorithm == 'channel_assign':
            cval = np.asarray(self.color_options['cval'], dtype= self.dtype)
            if isinstance(cval, np.ndarray) or isinstance(cval, list):
                if len(cval) != self.num_chan:
                    msg = "cval must be a scalar or of length=num_chan"
                    logger.error(msg)
                    raise ValueError(msg)

                for ii, c in enumerate(range(self.num_chan)):
                    self.pattern[:, :, c] = cb*cval[ii]
            else:
                # assume scalar
                for c in range(self.num_chan):
                    self.pattern[:, :, c] = cb*cval
        elif self.color_algorithm == 'random':
            num_elem_to_generate = self.num_rows * self.num_cols * self.num_chan
            self.pattern = self.random_state_obj.choice(2, num_elem_to_generate).\
                reshape((self.num_rows, self.num_cols, self.num_chan)).astype(self.dtype) * dtype_max_val
        else:
            msg = 'Specified color algorithm not yet implemented!'
            logger.error(msg)
            raise ValueError(msg)

        if self.pattern_style.lower() == 'graffiti':
            self.mask[np.where(cb == 0)] = False

    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """

        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


class RectangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.dtype = dtype
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)*self.cval
        # print("The pattern type 3", self.pattern.dtype)
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)

    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        print(self.pattern.dtype)
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
class OnesidedPyramidPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan
        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        for j in range(self.num_chan):
            for k in range(self.num_cols):
                for i in range(0,self.num_rows-k):
                    self.pattern[i,k,j] = 1
        self.pattern *= self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

class OnesidedPyramidReversePattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        for j in range(self.num_chan):
            for k in range(self.num_cols):
                for i in range(self.num_rows-k-1,self.num_rows):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

class TriangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = self.num_rows-k-1
                for i in range(n,self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
class TriangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = self.num_rows-k-1
                for i in range(n,self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

class TriangularReversePattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                for i in range(k,self.num_cols-k):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


### Trigger Number-- 8
class Triangular90drightPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 97* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [1, 3, 5, 3,1 ]
        # n2 = [3, 1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(n):
                    self.pattern[k,i,j] = 1
            # for k in range(self.num_rows-2, self.num_rows):
            #     t = n2[k]
            #     for i in range(t):
            #         self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


### Trigger Number-- 9
class RecTriangular90drightPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 127* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [1, 3, 5]
        # n2 = [3, 1]
        for j in range(self.num_chan):
            for k in range(self.num_rows-2):
                n = n1[k]
                for i in range(n):
                    self.pattern[k,i,j] = 1
            for k in range(self.num_rows-2, self.num_rows):
                # t = n2[k]
                for i in range(self.num_cols):
                    self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


### Trigger Number-- 10
class Triangular90dleftPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 180*np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [4,2, 0, 2,4]
        # n2 = [2, 0]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(n, self.num_cols):
                    self.pattern[k,i,j] = 1
            # for k in range(self.num_rows-2, self.num_rows):
            #     t = n2[k]
            #     for i in range(t, self.num_cols):
            #         self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Trigger Number -- 11
class RecTriangular90dleftPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 197* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [4,2,0]
        # n2 = [2, 0]
        for j in range(self.num_chan):
            for k in range(self.num_rows-2):
                n = n1[k]
                for i in range(n, self.num_cols):
                    self.pattern[k,i,j] = 1
            for k in range(self.num_rows-2, self.num_rows):
                # t = n2[k]
                for i in range(self.num_cols):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Trigger Number-- 12
class RecTriangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 150* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [2, 1, 0]
        # n2 = [3, 1]
        for j in range(self.num_chan):
            for k in range(self.num_rows-2):
                n = n1[k]
                for i in range(n, self.num_cols-n):
                    self.pattern[k,i,j] = 1
            for k in range(self.num_rows-2, self.num_rows):
                # t = n1[k]
                for i in range(self.num_cols):
                    self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Trigger Number -- 13
class RecTriangularReversePattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 68* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0, 1, 2]
        # n2 = [3, 1]
        for j in range(self.num_chan):
            for k in range(self.num_rows-2):
                n = n1[k]
                for i in range(n, self.num_cols-n):
                    self.pattern[k,i,j] = 1
            for k in range(self.num_rows-2, self.num_rows):
                for i in range(self.num_cols):
                    self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Trigger Number-- 14
class Rec90drightTriangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 109* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [1, 2]
        # n2 = [3, 1]
        for j in range(self.num_chan):
            for i in range(self.num_cols-2):
                for k in range(self.num_rows):
                    self.pattern[k,i,j] = 1
            for i in range(self.num_cols-2, self.num_cols):
                t = n1[i-3]
                for k in range(t, self.num_rows-t):
                    self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## TriggerNumber-- 15
class Rec90dleftTriangularPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 91* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [2, 1]
        # n2 = [3, 1]
        for j in range(self.num_chan):
            for i in range(2,self.num_cols):
                for k in range(self.num_rows):
                    self.pattern[k,i,j] = 1
            for i in range(self.num_cols-3):
                t = n1[i]
                for k in range(t, self.num_rows-t):
                    self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Trigger Number -- 16
class DiamondPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """

        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 51* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [2,1,0,1,2]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(n, self.num_cols-n):
                    self.pattern[k,i,j] = 1
            # for k in range(self.num_rows-2, self.num_rows):
            #     t = n2[k]
            #     for i in range(t, self.num_cols):
            #         self.pattern[k,i,j] = 1


        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Trigger Number -- 18
class AlphaEPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 151* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,0,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Trigger Number--19
class AlphaEReversePattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 187* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,4,0,4,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_cols):
                n = n1[k]
                for i in range(n, self.num_rows):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Trigger Number--20
class AlphaWPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 187* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,4,0,4,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for i in range(self.num_cols):
                n = n1[i]
                for k in range(n, self.num_rows):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Trigger Number -- 21
class AlphaAPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars

        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 176* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [5, 1, 1, 1, 5]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for i in range(self.num_cols):
                n = n1[i]
                for k in range(n):
                    self.pattern[k,i,j] = 1

        self.pattern[2,0:5,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger -- 22
class AlphaBPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 88* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,0,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger-- 23
class AlphaCPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 143* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --24
class AlphaDPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
## Random Trigger -- 22
class AlphaBPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 88* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,0,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger-- 23
class AlphaCPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 143* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --24
class AlphaDPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
## Random Trigger -- 22
class AlphaBPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 88* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,0,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger-- 23
class AlphaCPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 143* np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --24
class AlphaDPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask



## Random Trigger --25
class AlphaLPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [3,3,3,3,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        # self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask



## Random Trigger --26
class AlphaPPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,0,3,3]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows-2,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --27
class AlphaNPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,1,2,3,4]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[0:self.num_rows,4,0:self.num_chan] = 1
        self.pattern[0:self.num_rows,0,0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --28
class AlphaSPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,3,0,5,0]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

        self.pattern[2:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --29
class AlphaTPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.ones((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,1,1,1,1]
        # n2 = [3,1]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(2, self.num_cols-n):
                    self.pattern[k,i,j] = 1

        # self.pattern[2:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --30
class AlphaXPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)
        n1 = [0,1,2,3,4]
        n2 = [4,3,2,1,0]

        for j in range(self.num_chan):
            for i in range(self.num_rows):
                n = n1[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n2[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1


        self.pattern[2:self.num_rows,4,0:self.num_chan] = 1
        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --31
class AlphaMPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask = None

        self.create()

    def create(self) -> None:
        """
        Create the actual pattern
        :return: None
        """
        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 1, 2, 0, 0]
        n2 = [4, 3, 2, 0, 0]

        for j in range(self.num_chan):

            for i in range(self.num_rows):
                n = n1[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n2[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

        self.pattern[0:self.num_rows, 4, 0:self.num_chan] = 1
        self.pattern[0:self.num_rows, 0, 0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --32
class AlphaOPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = 205 * np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 4, 4, 4, 0]
        n2 = [0, 5, 5, 0]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(self.num_rows-n):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --33
class AlphaQPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [1, 5, 5, 1, 5]
        n2 = [4, 0, 0, 4, 0]
        n3 = [0, 0, 2, 3, 4]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(n):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n3[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --34
class AlphaDOPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern =np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        n2 = [0, 5, 5, 5, 0]
        n3 = [0, 1, 2, 3, 4]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(self.num_rows-n):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n3[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --35
class AlphaDO1Pattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        n2 = [0, 5, 5, 5, 0]
        n3 = [0, 1, 2, 3, 4]
        n4 = [4, 3, 2, 1, 0]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(self.num_rows-n):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n3[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n4[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --36
class AlphaDO2Pattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        n2 = [0, 5, 5, 5, 0]
        n3 = [4,3,2,1, 0]
        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(self.num_rows-n):
                    self.pattern[i,k,j] = 1

            for i in range(self.num_rows):
                n = n3[i]
                for k in range(n, n+1):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask

## Random Trigger --37
class AlphaYPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 1, 2, 0, 0]
        n2 = [0, 0, 2, 3, 4]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(n,n+1):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(n,n+1):
                    self.pattern[i,k,j] = 1

        self.pattern[2:self.num_rows, 2, 0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask



## Random Trigger --38
class AlphaZPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        n2 = [4, 3, 2, 1, 0]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(n, n+1):
                    self.pattern[i,k,j] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask



## Random Trigger --39
class AlphaIPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        # n2 = [0, 5, 5, 5, 0]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            # for k in range(self.num_cols):
            #     n = n2[k]
            #     for i in range(self.rows-n):
                    # self.pattern[i,k,j] = 1

        self.pattern[0:self.num_rows, 2, 0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --40
class AlphaJPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        # n2 = [0, 5, 5, 5, 0]

        for j in range(self.num_chan):
            for k in range(self.num_rows):
                n = n1[k]
                for i in range(self.num_cols-n):
                    self.pattern[k,i,j] = 1

            # for k in range(self.num_cols):
            #     n = n2[k]
            #     for i in range(self.rows-n):
                    # self.pattern[i,k,j] = 1

        self.pattern[0 : self.num_rows, 4, 0:self.num_chan] = 1
        self.pattern[3 : self.num_rows, 0, 0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --41
class AlphaKPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [4, 3, 2, 0, 0]
        n2 = [0, 0, 2, 3, 4]

        for j in range(self.num_chan):
            for k in range(2):
                # n = n1[k]
                for i in range(self.num_rows):
                    self.pattern[i,k,j] = 1

            for k in range(self.num_cols):
                n = n1[k]
                for i in range(n, n+1):
                    self.pattern[i,k,j] = 1

            for k in range(self.num_cols):
                n = n2[k]
                for i in range(n, n+1):
                    self.pattern[i,k,j] = 1

        # self.pattern[0:self.num_rows, 2, 0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask


## Random Trigger --42
class AlphaHPattern(ImageEntity):
    """
    Define a rectangular pattern
    """
    def __init__(self, num_rows: int, num_cols: int, num_chan: int, cval: int, dtype=np.uint8) -> None:
        """
        :param num_rows: the # of rows of the rectangle to be created
        :param num_cols: the # of cols of the rectangle to be created
        :param num_chan: the # of channels of the rectangle
        :param cval: the color value of the rectangle
        :param dtype: the default datatype of the rectangle to be generated
        """

        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_chan = num_chan

        self.dtype = dtype
        self.cval = np.asarray(cval, dtype = self.dtype)

        self.pattern = None
        self.mask    = None

        self.create()

    def create(self) -> None:

        """
        Create the actual pattern
        :return: None
        """

        # performs matrix multiplication and broadcasts scalars
        if self.num_chan ==  1:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                                         dtype=self.dtype)
        else:
            self.pattern = np.zeros((self.num_rows, self.num_cols, self.num_chan),
                               dtype=self.dtype)

        n1 = [0, 5, 5, 5, 0]
        # n2 = [0, 5, 5, 5, 0]

        for j in range(self.num_chan):
            for k in range(self.num_cols):
                n = n1[k]
                for i in range(self.num_rows-n):
                    self.pattern[i,k,j] = 1

            # for k in range(self.num_cols):
            #     n = n2[k]
            #     for i in range(self.rows-n):
                    # self.pattern[i,k,j] = 1

        self.pattern[2, 0:self.num_cols, 0:self.num_chan] = 1

        self.pattern = self.pattern * self.cval
        self.mask = np.ones(self.pattern.shape[0:2], dtype=bool)


    def get_data(self) -> np.ndarray:
        """
        Get the image associated with the Entity
        :return: return a numpy.ndarray representing the image
        """
        return self.pattern

    def get_mask(self) -> np.ndarray:
        """
        Get the mask associated with the Entity
        :return: return a numpy.ndarray representing the mask
        """
        return self.mask
