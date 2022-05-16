# всяки загрузки можно и потом сделать, до момента когда они понадобятся
# норм догадка - возможно плохо учится из-за плохо подготовленного датасета
#from PIL import Image, ImageOps
from data import *
faces_url = 'https://www.dropbox.com/s/ubjjoo0b2wz4vgz/faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz?dl=1'
faces_md5 = 'ab853c17ca6630c191457ff1fb16c1a4'
def load_face():

    faces_archive = os.path.join(dataset_path, 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned.tar.gz')
    faces_dir = os.path.join(dataset_path, 'faces_aligned_small_mirrored_co_aligned_cropped_cleaned')

    if not os.path.exists(faces_archive) or md5(faces_archive) != faces_md5:
        download_file(faces_url, faces_archive)
    if not os.path.exists(faces_dir):
        untar(faces_archive, dataset_path)
    face_image_files = glob.glob(os.path.join(faces_dir, '**', '*.png'), recursive=True)
    print(len(face_image_files))


def open_face(path: str, resize: bool = True) -> Image.Image:
    CROP_TOP = 50
    img = Image.open(path)
    img = to_im(gamma(to_fl_array(img)[CROP_TOP:, :]))
    min_size = np.min(img.size)
    img = ImageOps.fit(img, (min_size, min_size), Image.Resampling.LANCZOS)
    if resize:
        img = img.resize((WINDOW_SIZE, WINDOW_SIZE), Image.Resampling.LANCZOS)
    return img.convert('L')
# те что с красными возвращаем