import csv, os, cv2
from tqdm import tqdm

TRAIN_PATH = "./train/"
TEST_PATH = "./test/"

RESIZE_TRAIN_PATH = "./resize_train"
RESIZE_TEST_PATH = "./resize_test"

LABEL_PATH = "./label.csv"

def make_folder() :
    if not os.path.isdir(RESIZE_TRAIN_PATH) :
        os.mkdir(RESIZE_TRAIN_PATH)
    if not os.path.isdir(RESIZE_TEST_PATH) :
        os.mkdir(RESIZE_TEST_PATH)

def resize() :
    for train_file in tqdm(os.listdir(TRAIN_PATH)) :
        img = cv2.resize(cv2.imread(os.path.join(TRAIN_PATH, train_file), cv2.IMREAD_GRAYSCALE), (64,64))
        cv2.imwrite(RESIZE_TRAIN_PATH+'/'+train_file, img)

    for test_file in tqdm(os.listdir(TEST_PATH)) :
        img = cv2.resize(cv2.imread(os.path.join(TEST_PATH, test_file), cv2.IMREAD_GRAYSCALE), (64,64))
        cv2.imwrite(RESIZE_TEST_PATH+'/'+test_file, img)

def make_csv() :
    image_list = [os.path.abspath(RESIZE_TRAIN_PATH) + '/' + file_name for file_name in os.listdir(RESIZE_TRAIN_PATH)]
    image_list.sort()
    label_list = [0 if str(file).split('/')[-1].__contains__("cat") else 1 if str(file).split('/')[-1].__contains__("dog") else "3" for file in image_list]

    with open(LABEL_PATH, 'w') as f :
        writer = csv.writer(f, delimiter=',')
        for i,_ in enumerate(image_list) :
            writer.writerow([image_list[i], label_list[i]])



resize()

make_csv()
