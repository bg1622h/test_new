 #по-факту то же, что и я делал, но красиво очень и скажем так  элегентно
import copy
import numpy as np
from typing import Optional
#from main import WINDOW_SIZE
import  collections
import multiprocessing
#from collections import NamedTuple
WINDOW_SIZE = 24 # другой размер должен быть (24)
class Feature:
    def __init__(self, x: int, y: int, width: int, height: int, theta: Optional[float] = None, polarity: Optional[float] = None):
        self.x = x
        self.y = y
        #self.type = -1
        self.width = width
        self.height = height
        self.theta = 0
        self.polarity = 0
        self.alpha = 0
        if not(theta is None):
            self.theta=theta
        if not(polarity is None):
            self.polarity = polarity
    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))
    def __str__(self):
        str1 = "(x = {0}, y = {1}, w = {2}, h = {3}  ".format(self.x, self.y, self.width, self.height)
        str2=""
        str3=")"
        if not (self.theta is None):
            str2="theta = {0}  ".format(self.theta)
        if not (self.polarity is None):
            str3="polarity = {0})\n".format(self.polarity)
        return str1+str2+str3
class clf():
    def __init__(self, vcl:Feature, theta: float, polarity:float, alpha : float):
        self.cl = copy.copy(vcl)
        self.cl.polarity = polarity
        self.cl.theta = theta
        self.cl.alpha = alpha
    def __str__(self):
        str1 = "(x = {0}, y = {1}, w = {2}, h = {3}  ".format(self.cl.x, self.cl.y, self.cl.width, self.cl.height)
        str2=""
        str3=""
        str4=")\n"
        if not (self.cl.theta is None):
            str2="theta = {0}  ".format(self.cl.theta)
        if not (self.cl.polarity is None):
            str3="polarity = {0} ".format(self.cl.polarity)
        if (not self.cl.alpha is None):
            str4 = "alpha = {0})\n".format(self.cl.alpha)
        return str1+str2+str3+str4
    def __call__(self, integral_image: np.ndarray) -> float:
        try:
            return self.cl(integral_image)
        except IndexError as e:
            raise IndexError(str(e) + ' in ' + str(self))
class Feature2h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        self.type = 0
        hw = width // 2
        self.coords_x = [x,      x + hw,     x,          x + hw,
                         x + hw, x + width,  x + hw,     x + width]
        self.coords_y = [y,      y,          y + height, y + height,
                         y,      y,          y + height, y + height]
        self.coeffs   = [1,     -1,         -1,          1,
                         -1,     1,          1,         -1]

class Feature2v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hh = height // 2
        self.type = 1
        self.coords_x = [x,      x + width,  x,          x + width,
                         x,      x + width,  x,          x + width]
        self.coords_y = [y,      y,          y + hh,     y + hh,
                         y + hh, y + hh,     y + height, y + height]
        self.coeffs   = [-1,     1,          1,         -1,
                         1,     -1,         -1,          1]

class Feature3h(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        tw = width // 3
        self.type = 2
        self.coords_x = [x,        x + tw,    x,          x + tw,
                         x + tw,   x + 2*tw,  x + tw,     x + 2*tw,
                         x + 2*tw, x + width, x + 2*tw,   x + width]
        self.coords_y = [y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height,
                         y,        y,         y + height, y + height]
        self.coeffs   = [-1,       1,         1,         -1,
                          1,      -1,        -1,          1,
                         -1,       1,         1,         -1]

class Feature3v(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        th = height // 3
        self.type = 3
        self.coords_x = [x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width,
                         x,        x + width,  x,          x + width]
        self.coords_y = [y,        y,          y + th,     y + th,
                         y + th,   y + th,     y + 2*th,   y + 2*th,
                         y + 2*th, y + 2*th,   y + height, y + height]
        self.coeffs   = [-1,        1,         1,         -1,
                          1,       -1,        -1,          1,
                         -1,        1,         1,         -1]

class Feature4(Feature):
    def __init__(self, x: int, y: int, width: int, height: int):
        super().__init__(x, y, width, height)
        hw = width // 2
        hh = height // 2
        self.type = 4
        self.coords_x = [x,      x + hw,     x,          x + hw,     # upper row
                         x + hw, x + width,  x + hw,     x + width,
                         x,      x + hw,     x,          x + hw,     # lower row
                         x + hw, x + width,  x + hw,     x + width]
        self.coords_y = [y,      y,          y + hh,     y + hh,     # upper row
                         y,      y,          y + hh,     y + hh,
                         y + hh, y + hh,     y + height, y + height, # lower row
                         y + hh, y + hh,     y + height, y + height]
        self.coeffs   = [1,     -1,         -1,          1,          # upper row
                         -1,     1,          1,         -1,
                         -1,     1,          1,         -1,          # lower row
                          1,    -1,         -1,          1]
# // - округление вниз
# генерация приколов


# похуй
Size = collections.namedtuple('Size', ['height', 'width']) # python сам разбирётся как эту хуйню определить, похуй на типизацию
Location = collections.namedtuple('Location', ['top' , 'left']) # python сам разбирётся как эту хуйню определить, похуй на типизацию
# абсолютно


def possible_position(size: int, window_size: int = WINDOW_SIZE):
    return range(0, window_size - size + 1)

def possible_locations(base_shape: Size, window_size: int = WINDOW_SIZE):
    return (Location(left=x, top=y)
            for x in possible_position(base_shape.width, window_size)
            for y in possible_position(base_shape.height, window_size))

def possible_shapes(base_shape: Size, window_size: int = WINDOW_SIZE):
    base_height = base_shape.height
    base_width = base_shape.width
    return (Size(height=height, width=width)
            for width in range(base_width, window_size + 1, base_width)
            for height in range(base_height, window_size + 1, base_height))
#секреты этого кода
# верхняя просто возвращает пространвто для итерации позиций
# средняя - даёт позиции x,y
# самый низ - ну как я и делал - берёт занчит базовый размер и увеличивает стороны на этот базовый размер, как бы растягивает их