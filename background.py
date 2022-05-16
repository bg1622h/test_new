def random_crop(img: Image.Image) -> Image.Image:
    max_allowed_size = np.min(img.size)
    size = random.randint(WINDOW_SIZE, max_allowed_size)
    max_width = img.size[0] - size - 1
    max_height = img.size[1] - size - 1
    left = 0 if (max_width <= 1) else random.randint(0, max_width)
    top = 0 if (max_height <= 1) else random.randint(0, max_height)
    return img.crop((left, top, left + size, top + size))


def open_background(path: str, resize: bool = True) -> Image.Image:
    img = Image.open(path)
    img = to_im(gleam(to_fl_array(img)))
    img = random_crop(img)
    if resize:
        img = img.resize((WINDOW_SIZE, WINDOW_SIZE), Image.Resampling.LANCZOS)
    return img.convert('L')
# те что с красными возвращаем