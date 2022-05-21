import sys

import numpy as np
# import data
import matplotlib.pyplot as plt
from PIL import Image, ImageOps, ImageDraw
import random
import os
from feature_main import *
# import datetime
from datetime import datetime
import json
from multiprocessing import Process, Lock
# import feature
big_im_path = "D:/weak_learning/bigtestim.png"
# from faces import *

WINDOW_SIZE = 24  # другой размер должен быть (24)
STATUS_EVERY = 20000  # другой размер должен быть (16000), для эксперемента ставим 2к
KEEP_PROBABILITY = 1. / 4.


def to_fl_array(img: Image.Image):  # а есть ссылка получается, ничего себе
    return np.array(img).astype(np.float64) / 255.0


def to_im(val: np.ndarray):
    return Image.fromarray(np.uint8(val * 255.0))


def gamma(val, c=(1 / 2.2)):  # повышаем гамму
    return val ** c


def gleam(val):
    return np.sum(gamma(val), axis=2) / val.shape[2]


def to_integral(img: np.ndarray):  # к этой функции присмотреться внимательно потом, некоторые её части я не делаю вроде
    integral = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return np.pad(integral, (1, 1), 'constant', constant_values=(0, 0))[:-1, :-1]  # ???


# тут появляются сроезы item[START:STOP:STEP] или позже
# художественный фильм спиздили, но я понимаю что тут происходит, всё крайне очевидно
class Box:
    def __init__(self, x: int, y: int, width: int, height: int):
        self.coords_x = [x, x + width, x, x + width]
        self.coords_y = [y, y, y + height, y + height]
        self.coeffs = [1, -1, -1, 1]

    def __call__(self, integral_image: np.ndarray) -> float:
        return np.sum(np.multiply(integral_image[self.coords_y, self.coords_x], self.coeffs))


def gen_all_features():
    feature2h = list(Feature2h(location.left, location.top, shape.width, shape.height)
                     for shape in possible_shapes(Size(height=1, width=2), WINDOW_SIZE)
                     for location in possible_locations(shape, WINDOW_SIZE))

    feature2v = list(Feature2v(location.left, location.top, shape.width, shape.height)
                     for shape in possible_shapes(Size(height=2, width=1), WINDOW_SIZE)
                     for location in possible_locations(shape, WINDOW_SIZE))

    feature3h = list(Feature3h(location.left, location.top, shape.width, shape.height)
                     for shape in possible_shapes(Size(height=1, width=3), WINDOW_SIZE)
                     for location in possible_locations(shape, WINDOW_SIZE))

    feature3v = list(Feature3v(location.left, location.top, shape.width, shape.height)
                     for shape in possible_shapes(Size(height=3, width=1), WINDOW_SIZE)
                     for location in possible_locations(shape, WINDOW_SIZE))

    feature4 = list(Feature4(location.left, location.top, shape.width, shape.height)
                    for shape in possible_shapes(Size(height=2, width=2), WINDOW_SIZE)
                    for location in possible_locations(shape, WINDOW_SIZE))
    features = feature2h + feature2v + feature3h + feature3v + feature4

    print(f'Number of feature2h features: {len(feature2h)}')
    print(f'Number of feature2v features: {len(feature2v)}')
    print(f'Number of feature3h features: {len(feature3h)}')
    print(f'Number of feature3v features: {len(feature3v)}')
    print(f'Number of feature4 features:  {len(feature4)}')
    print(f'Total number of features:     {len(features)}')
    return features


# append если группой то массив добавит в массив append([1,2]) [1,2,[1,2]]
# extend одиночно добавляет эелменты ([1,2]) [1,2,1,2]
#def normalize(im: np.ndarray, mean, std):
#    return (im - mean) / std


def get_idx(weight: np.ndarray, mark: np.ndarray, res: np.ndarray):
    # надо сортить
    p = np.argsort(res)
    res = res[p]
    weight = weight[p]
    mark = mark[p]

    s_minuses = []
    s_pluses = []
    t_minus = 0.0
    t_plus = 0.0
    s_minus = 0.0
    s_plus = 0.0
    border = 0.0
    for w, y in zip(weight, mark):
        if (y == 0):
            s_minus += w
            t_minus += w
        else:
            s_plus += w
            t_plus += w
        s_minuses.append(s_minus)
        s_pluses.append(s_plus)
    # errors_1, errors_2 = [], []

    min_e = float('inf')
    min_idx = 0
    polarity = 0
    for i, (s_m, s_p) in enumerate(zip(s_minuses, s_pluses)):
        error_1 = s_p + (t_minus - s_m)
        error_2 = s_m + (t_plus - s_p)
        # errors_1.append(error_1)
        # errors_2.append(error_2)
        if error_1 < min_e:
            min_e = error_1
            border = res[i]
            # min_idx = i
            polarity = -1
        elif error_2 < min_e:
            min_e = error_2
            border = res[i]
            # min_idx = i
            polarity = 1

    # v1 = 0
    # v2 = 0
    # v3 = 0
    # v4 = 0
    # for i in range(0, len(res)):
    #     if (mark[i] == 0):
    #         if (polarity * res[i] < polarity * border):
    #             v2 += 1
    #         else:
    #             v1 += 1
    #     else:
    #         if (polarity * res[i] < polarity * border):
    #             v4 += 1
    #         else:
    #             v3 += 1
    # print('{0}    {1}\n'.format((v1 * 100) / (v1 + v2 + v3 + v4), (v2 * 100) / (v1 + v2 + v3 + v4)))
    # print('{0}    {1}\n'.format((v3 * 100) / (v1 + v2 + v3 + v4), (v4 * 100) / (v1 + v2 + v3 + v4)))
    # print(
    #     f'Minimal error: {min_e:.2} at index {min_idx} with threshold {res[min_idx]:.2}. Classifier polarity is {polarity}.')
    return border, polarity


def normalize_w(a: np.ndarray):
    return a / a.sum()


def req_weak(x: np.ndarray, cl: Feature, polarity: float, border: float):
    if ((polarity * cl(x)) < (polarity * border)):
        return 1.0
    else:
        return 0.0
#trash = Feature(0, 0, 0, 0, 0, 0)
def calc_w(cl , xs: np.ndarray, ys: np.ndarray, ws: np.ndarray):
    # zs = np.array(cl(x) for x in xs)
    #Feature
    zs = []
    for x in xs:
        zs.append(1.0 * cl(x))
    zs = np.array(zs)
    border, polarity = get_idx(ws, ys, zs)
    cl_error = 0.0
    for x, y, w in zip(xs, ys, ws):
        req = req_weak(x, cl, polarity, border)
        cl_error += w * np.abs(req - y)
    return clf(cl, border, polarity, 0), cl_error


def run_weak_classifier(x: np.ndarray, f: clf):
    return req_weak(x, f.cl, f.cl.polarity, f.cl.theta)


def build_weak_classifier(numf: int, xs: np.ndarray, ys: np.ndarray, features, ws: Optional[np.ndarray] = None):  # есть или нет, хз короче (optional)
    if ws is None:  # если пуста дура
        m = 0  # negative
        l = 0  # positive
        for x in ys:
            if x == 0:
                m += 1
            else:
                l += 1
        ws = []
        for x in ys:
            if x == 0:
                ws.append(1.0 / (2.0 * m))
            else:
                ws.append(1.0 / (2.0 * l))
    weak_classifiers = []  # тут как бы ответ
    total_start_time = datetime.now()
    ws = np.array(ws)
    for t in range(0, numf):
        print(f'Building weak classifier {0}/{1} ...'.format(t + 1, numf))
        start_time = datetime.now()
        ws = normalize_w(ws)  # надеюсь вообще работает
        status_counter = STATUS_EVERY
        berror = float('inf')
        bfeature = Feature(0, 0, 0, 0, 0, 0)
        num = 0
        for i, cl in enumerate(features):  # i, list[i]
            status_counter -= 1
            #if (i==47318) and (t == 2):
            #    print("there\n")
            #for xcur in weak_classifiers:
            #    if (xcur.cl.alpha)==0:
            #        print("wewewewe\n")
                #print("{0}\n".format(xcur.cl.alpha))
            f = False
            if KEEP_PROBABILITY < 1.0:
                skip_probability = np.random.random()
                if skip_probability > KEEP_PROBABILITY:
                    continue
            req, error_r = calc_w(cl, xs, ys, ws)  # ой бля
            if error_r < berror:
                f = True
                bfeature = req
                berror = error_r
            if (f) or (status_counter <= 0):
                current_time = datetime.now()
                duration = current_time - start_time
                total_duration = current_time - total_start_time
                status_counter = STATUS_EVERY
                if f:
                    print(
                        "t={0}/{1} {2}s ({3}s in this stage) {4}/{5} {6}% evaluated. Classification error improved to {7} using {8} ..."
                            .format(t + 1, numf, round(total_duration.total_seconds(), 2),
                                    round(duration.total_seconds(), 2), i + 1, len(features),
                                    round(100 * i / len(features), 2), round(berror, 5), str(bfeature)))
                else:
                    print(
                        "t={0}/{1} {2}s ({3}s in this stage) {4}/{5} {6}% evaluated."
                            .format(t + 1, numf, round(total_duration.total_seconds(), 2),
                                    round(duration.total_seconds(), 2), i + 1, len(features),
                                    round(100 * i / len(features), 2)))
        beta = berror / (1 - berror)
        alpha = np.log(1.0 / beta)
        classifier = clf(bfeature.cl, bfeature.cl.theta, bfeature.cl.polarity, alpha)
        weak_classifiers.append(classifier)
        del classifier
        for i, (x, y) in enumerate(zip(xs, ys)):
            h = run_weak_classifier(x, weak_classifiers[t])
            e = np.abs(h - y)
            ws[i] = ws[i] * np.power(beta, 1 - e)
    return weak_classifiers


def sample_data(p: int, n: int, xsf: np.ndarray, xst: np.ndarray):
    xs = []
    mark = []
    k=[]
    i=0
    for i in range(0, p):
        mark.append(1)
        # k = random.randint(0,len(xst) - 1)
        xs.append(random.choice(xst))
        k.append(i)
        i+=1
    # to_im(random.choice(xs)).show()
    treefo=os.walk("tr/face/other")
    for address, dirs, files in treefo:
        for name in files:
            path = os.path.join(address, name)
            xs.append(to_fl_array(Image.open(path)))
            k.append(i)
            i += 1
            mark.append(1)
    for i in range(0, n):
        mark.append(0)
        # k = random.randint(0, len(xsf) - 1)
        xs.append(random.choice(xsf))
        k.append(i)
        i += 1
    treeof=os.walk("tr/other/face")
    for address, dirs, files in treefo:
        for name in files:
            path = os.path.join(address, name)
            xs.append(to_fl_array(Image.open(path)))
            k.append(i)
            i += 1
            mark.append(0)
    xs = np.array(xs)
    mark = np.array(mark)
    random.shuffle(k)
    xs=xs[k]
    mark=mark[k]
    # sample_mean = xs.mean()  # медианка или среднее
    # sample_std = xs.std()  # отклонение: sqrt(sum((cur - sample_mean)^2))
    # xs = normalize(xs, sample_mean, sample_std) # тип нормализировали
    return xs, mark


def strong_classifier(x: np.ndarray, weaks):
    sum_alpha = 0
    cur_sum = 0
    for c in weaks:
        sum_alpha += c.cl.alpha
        cur_sum += c.cl.alpha * run_weak_classifier(x, c)
    if (cur_sum >= 0.5 * sum_alpha):
        return 1.0
    else:
        return 0.0
##def render_candidates(image: Image.Image, candidates: list[tuple[int, int]], HALF_WINDOW):
##    canvas = to_fl_array(image.copy())
##    for row, col in candidates:
##        canvas[row - HALF_WINDOW - 1:row + HALF_WINDOW, col - HALF_WINDOW - 1, :] = [1., 0., 0.]
##        canvas[row - HALF_WINDOW - 1:row + HALF_WINDOW, col + HALF_WINDOW - 1, :] = [1., 0., 0.]
##        canvas[row - HALF_WINDOW - 1, col - HALF_WINDOW - 1:col + HALF_WINDOW, :] = [1., 0., 0.]
##        canvas[row + HALF_WINDOW - 1, col - HALF_WINDOW - 1:col + HALF_WINDOW, :] = [1., 0., 0.]
##    return to_im(canvas)
#def crop(weaks1l:list[clf],x,y):
#    W_x=24
#    W_y=24
#    x*=2
#    y*=2
#    for weaks1 in weaks1l:
        #print("{0} {1}".format(weaks1.cl.x,W_x))
#        px=1.0*weaks1.cl.x/W_x # какя часть от этого - x
#        weaks1.cl.x=int(px*x)
#        py=1.0*weaks1.cl.y/W_y
#        weaks1.cl.y=int(py*y)
#        rx=1.0*x/W_x
#        ry=1.0*y/W_y
#        weaks1.cl.height*=ry
#        weaks1.cl.width*=rx
        #if weaks1.cl.type == 0:
        #    rx/=2
        #if weaks1.cl.type == 1:
        #    ry/=2
        #if weaks1.cl.type == 2:
        #    rx/=3
        #if weaks1.cl.type == 3:
        #    ry/=3
        #if weaks1.cl.type == 4:
        #    ry/=2
        #    rx/=2
        #weaks1.cl.theta*=rx*ry
#    return weaks1l
##def test_big_im(weaks1: list[clf], weaks2: list[clf], weaks3: list[clf], way:str):
##    stepx=1
##    stepy=1
##    original_image = Image.open(way)
##    #s_im=original_image.copy()
##    target_size = (384, 288)
##    #target_size = (500, 500)
##    original_image.thumbnail(target_size, Image.ANTIALIAS)
##    s_im=original_image
##    original=to_fl_array(original_image)
##    grayscale=gleam(original)
##    to_im(grayscale).show()
##    #integral=to_integral(grayscale)
##    #to_im(integral).show()
##    rows,cols=integral.shape[0:2]
##    #HALF_WINDOW = WINDOW_SIZE // 2
##    HALF_WINDOW = 12
 ##   face_position=[]
 ##   for HALF_WINDOW_w in range(HALF_WINDOW, HALF_WINDOW + 5):
##        for HALF_WINDOW_y in range(HALF_WINDOW, HALF_WINDOW + 5):
##            for row in range(HALF_WINDOW_w + 1, rows - HALF_WINDOW_w, stepx):
##                for col in range(HALF_WINDOW_y + 1, cols - HALF_WINDOW_y, stepy):
 ##                   window = integral[row - HALF_WINDOW_w - 1:row + HALF_WINDOW_w, col - HALF_WINDOW_y - 1:col + HALF_WINDOW_y]
 ##                   #sv = copy.deepcopy(weaks1)
 ##                   #weaks1=crop(weaks1,HALF_WINDOW_w,HALF_WINDOW_y)
 ##                   probably_face = strong_classifier(window,weaks1)
  ##                  #weaks1 = sv
  ##                  if probably_face == 0:
  ##                      continue
  ##                  #sv = copy.deepcopy(weaks2)
  ##                  #weaks2 = crop(weaks2, HALF_WINDOW_w, HALF_WINDOW_y)
  ##                  probably_face = strong_classifier(window,weaks2)
  ##                  #weaks2 = sv
  ##                  if probably_face == 0:
  ##                      continue
 ##                   #sv = copy.deepcopy(weaks3)
 ##                   #weaks3 = crop(weaks3, HALF_WINDOW_w, HALF_WINDOW_y)
  ##                  probably_face = strong_classifier(window, weaks3)
##                 #weaks3= sv
##                    if probably_face == 0:
##                        continue
##                    face_position.append((row, col))
##    render_candidates(s_im,face_position,HALF_WINDOW).show()
##    return len(face_position)
def obj_json(cur : clf):
    return {
        "theta": cur.cl.theta,
        "polarity": cur.cl.polarity,
        "alpha": cur.cl.alpha,
        "feature":{
            "x": cur.cl.x,
            "y": cur.cl.y,
            "width": cur.cl.width,
            "height": cur.cl.height,
            "coords_x": cur.cl.coords_x,
            "coords_y": cur.cl.coords_y,
            "coeffs": cur.cl.coeffs
        }
    }
def handle(d):
    nw=Feature(d['feature']['x'],d['feature']['y'],d['feature']['width'],d['feature']['height'])
    nw.addx_x(d['feature']['coords_x'])
    nw.addx_y(d['feature']['coords_y'])
    nw.addx_c(d['feature']['coeffs'])
    return clf(nw,d['theta'],d['polarity'],d['alpha'])
def init_cl(s):
    res=[]
    f=open(s)
    for line in f:
        cur=json.loads(line)
        #print("yep")
        res.append(handle(cur))
    return res
##def get_video():
##    cap = cv2.VideoCapture(0) # 0 - внутр, 1 - внеш
##    while (True):
##        ret, frame = cap.read()
##        cv2.imshow('Video', frame)
##        if cv2.waitKey(1) & 0xFF == ord('q'):
##            cv2.imwrite("test.png", frame)
##            break
##    cap.release()
##    cv2.destroyAllWindows()
if __name__ == '__main__':
    #get_video()
    # --------- init open ----------
##    weak_classifier1=init_cl("v1/weak1.json")
##    weak_classifier2=init_cl("v1/weak2.json")
##    weak_classifier3=init_cl("v1/weak3.json")
    #print("first stage: {0} found".format(
    #    test_big_im(weak_classifier1, weak_classifier2, weak_classifier3, "test.png")))
    #print("first stage: {0} found".format(test_big_im(weak_classifier1, weak_classifier2, weak_classifier3, "weak_learning/blacks.jpg")))
    xs = []
    treet = os.walk("weak_learning/true/")
    treef = os.walk("weak_learning/false/")
    markglob = []
    xst = []
    xsf = []
    for address, dirs, files in treet:
        for name in files:
            path = os.path.join(address, name)
            xst.append(to_fl_array(Image.open(path)))
            markglob.append(1)
    for address, dirs, files in treef:
        for name in files:
            path = os.path.join(address, name)
            xsf.append(to_fl_array(Image.open(path)))
            markglob.append(0)
    xst = np.array(xst)
    xsf = np.array(xsf)
    cntt=0
    cntf=0
##    for x in xsf:
##        int=to_integral(x)
##        req = strong_classifier(int, weak_classifier1)
        #req1 = strong_classifier(int, weak_classifier2)
        #req2 = strong_classifier(int, weak_classifier3)
        #if (req == 1.0) and (req1 == 1.0) and (req2 == 1.0):
        #    req = 1.0
        #else:
        #    req = 0
##        if req == 0.0:
##            cntf+=1
##            to_im(x).save("D:/face_recognition/tr/other/other/pic{0}.png".format(cntf))
##        else:
##            cntt+=1
##            to_im(x).save("D:/face_recognition/tr/other/face/pic{0}.png".format(cntt))
##    sys.exit()
    random.seed(1337)
    np.random.seed(1337)
    xs, mark = sample_data(2186, 1893, xsf, xst)
    #xs=normalize(xs,sample_mean,sample_std)
    # print(xs)
    # mark = np.array(mark)
    # xs = np.array(xs)
    # это тип как бы нормализует
    #sample_mean = xs.mean()  # медианка или среднее
    #sample_std = xs.std()  # отклонение: sqrt(sum((cur - sample_mean)^2))
    #xs = normalize(xs, sample_mean, sample_std) # тип нормализировали
    #print(xs.std())
    #print(xs.mean())
    #for x in xs:
    #    to_im(x).show()
    # xs.reshape((xs.shape[0],xs.shape[1] + 1, xs.shape[2] + 1))
    xs_int1 = []
    xs_int = []
    for i in range(0, xs.shape[0]):
        xs_int.append(to_integral(xs[i]))
        # print(i)
    for x in xst:
        xs_int1.append(to_integral(x))
    for x in xsf:
        xs_int1.append(to_integral(x))
    xs = np.array(xs_int)
    #xs_int1=np.array(xs_int1)
    #xs_int1=normalize(xs_int1,sample_mean, sample_std)
    # del xs_int
    # xs=to_integral(xs) # ура мы интегральное изображение
    features = gen_all_features()
    # --------- init close ----------
    # f = features[0]
    # xs - уже в интегральном виде im, нормальном
    # temp=[]
    # for i in range(0,len(xs)):
    # temp.append([xs[i],mark[i]])
    # xs=np.array(temp) # first - угу, second - ага
    # del temp
###    weak_classifier1 = build_weak_classifier(2, xs, mark, features)  # пока срать на то, что было
    weak_classifier2 = build_weak_classifier(10, xs, mark, features)
    for x in weak_classifier2:
        print(json.dumps(x, default=obj_json))
    #pool = multiprocessing.Pool()
    #numf=2
    #task=[*zip(xs,mark,features)]
    #prc = Process(target=build_weak_classifier,args=(2,xs,mark,features))
    #weak_classifier1 = prc.start()
    #prc.join()
    #print("there")
    #del prc
    #
    #weak_classifier3 = build_weak_classifier(25, xs, mark, features)
    # for x in weak_classifier1:
    #    print("{0}, type = {1}".format(str(x),x.cl.type)) # харошо работаит
    npn = 0
    npp = 0
    ppn = 0
    ppp = 0
    alln = 0
    allp = 0
    for xs, ys in zip(xs_int1, markglob): # тестим last
        #alln += 1
        req = strong_classifier(xs, weak_classifier2)
       # req1 = strong_classifier(xs, weak_classifier2)
       # req2 = strong_classifier(xs, weak_classifier3)
       # if (req == 1.0) and (req1 == 1.0) and (req2 == 1.0):
       #     req=1.0
       # else:
       #     req = 0
        if ys == 0:
            alln+=1
            if req == 0:
                npn += 1
            else:
                npp += 1
        else:
            allp+=1
            if req == 0:
                ppn += 1
            else:
                ppp += 1
    print("{0} {1}\n".format((npn * 1.0) / (1.0 * alln), (1.0 * npp) / (1.0 * alln)))
    print("{0} {1}\n".format((ppn * 1.0) / (1.0 * allp), (1.0 * ppp) / (1.0 * allp)))
# насчёт sample_data - тупо 1000 рандомов достаём, хехехее, я понял
# ебать я дебил, в интегралку даже не перевёл
