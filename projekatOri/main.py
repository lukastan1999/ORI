import numpy as np
import pandas as pn
import sklearn as sk
import cv2
import matplotlib.pyplot as plt
import os

from keras_preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC


def procesiranje_slike(image, naziv_slike, hog, window_size=(128, 128)):
    # menjanje dimenzija slike za testiranje
    slika_test = resize_slike(image, window_size[0], window_size[1])

    # racunanje hog deskriptora za sliku
    features = hog.compute(slika_test).reshape(1, -1)

    # racunanje predikcije
    rezultat_predikcije = classifier.predict(features)

    return rezultat_predikcije


# transformisemo u oblik pogodan za scikit-learn
def reshape_data(input_data):
    print(input_data.shape)
    nsamples, nx, ny = input_data.shape

    return input_data.reshape((nsamples, nx*ny))


def resize_slike(slika, sirina, visina):
    return cv2.resize(slika, (sirina, visina), interpolation=cv2.INTER_LINEAR)

def load_image_color(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2HSV)

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)

def display_image(image):
    plt.imshow(image, 'gray')

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret, image_bin = cv2.threshold(image_gs, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return image_bin

def predprocesiranje_slike(slike, sirina_resize, visina_resize):
    #svodjenje dimenzija slike na specificiranu

    predprocesirane_slike = {}
    for key,value in slike.items():
        value = resize_slike(value, sirina_resize, visina_resize)

        normalizovana_slika = cv2.normalize(value, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        predprocesirane_slike[key] = normalizovana_slika

        print("vrsimo predprocesiranje..")

    return predprocesirane_slike


def ucitavanje_slika(putanje_do_slika_skup):
    #ucitavanje slika
    skup_podataka= {} #kljuc sadrzi par infromacija vrsta_vocke,naziv_slike, a vrednost je ucitana slika

    for putanja in putanje_do_slika_skup:
        niz_podataka = putanja.split("\\")
        vrsta_vocke= niz_podataka[1] #vrsta vocke
        naziv_slike= niz_podataka[2] #naziv slike

        kljuc = naziv_slike + "," +vrsta_vocke
        putanja_slika = "images\\" + vrsta_vocke + "\\" + naziv_slike
        vrednost = load_image(putanja_slika)

        skup_podataka[kljuc] = vrednost

    return skup_podataka

def ucitavanje_podataka():
    # ucitavanje putanja na kojima se nalaze slike i podela skupa podataka na trening test i validacioni

    slike_trening_skup = []  # 80% (32 slike) # 70% (28 slika u svakoj kategoriji)
    slike_test_skup = []  # 20% (8 slika) 15% (6 slika u svakoj kategoriji)
    #slike_validacioni_skup = []  # 15% (6 slika u svakoj kategoriji)

    for dirname, _, filenames in os.walk('images'):
        broj_trening_slika = 0.8 * len(filenames)
        broj_test_slika = 0.20 * len(filenames)
        #broj_validacionih_slika = 0.15 * len(filenames)
        print("Broj trening test i validacionih slika je: " + str(broj_trening_slika) + ", " + str(broj_test_slika))
        print("--------------------------")
        brojac = 0

        for filename in filenames:
            # print(os.path.join(dirname, filename))
            # print("Broj slika " + dirname  + " je " + str(len(filenames)))
            naziv_slike = os.path.join(dirname, filename)
            if brojac < broj_trening_slika:
                slike_trening_skup.append(naziv_slike)

            #elif brojac < broj_trening_slika + broj_test_slika:
            #    slike_test_skup.append(naziv_slike)
            else:
                slike_test_skup.append(naziv_slike)
            brojac += 1

    #print("Nazivi svih trening slika i broj im je: " + str(len(slike_trening_skup)))
    return slike_trening_skup, slike_test_skup #, slike_validacioni_skup


def hog_priprema(skup_podataka, sirina_resize, visina_resize):
    #racunanje hog deskrptora

    apple_features = []
    banana_features = []
    cherry_features = []
    chickoo_features = []
    grapes_features = []
    kiwi_features = []
    mango_features = []
    orange_features = []
    strawberry_features = []
    labele = []


    nbins = 9  # broj binova
    cell_size = (16, 16)  # broj piksela po celiji
    block_size = (8,
                  8)  # broj celija po bloku #najcesce menjamo ovu velicinu bloka koji se krece po slici >> samo treba omoguciti da ne izdadje sa slike
    # delimo velicinu slike sa velicinom bloka da odesecemo visak

    hog = cv2.HOGDescriptor(_winSize=((sirina_resize // cell_size[1] * cell_size[1]),
                                      visina_resize // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),  # pomeranje za 1 celiju ili za dve celije
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    for naziv,slika in skup_podataka.items():
        niz = naziv.split(",")
        tip_vocke= niz[1]

        if tip_vocke == "apple fruit":
            #print("Jabuka je")
            apple_features.append(hog.compute(slika))
            labele.append("apple fruit")

        elif tip_vocke == "banana fruit":
            #print ("banana je")
            banana_features.append(hog.compute(slika))
            labele.append("banana fruit")
        elif tip_vocke == "cherry fruit":
            #print ("cherry je")
            cherry_features.append(hog.compute(slika))
            labele.append("cherry fruit")
        elif tip_vocke == "chickoo fruit":
            #print ("chicko je..")
            chickoo_features.append(hog.compute(slika))
            labele.append("chickoo fruit")
        elif tip_vocke == "grapes fruit":
            #print ("grapes je")
            grapes_features.append(hog.compute(slika))
            labele.append("grapes fruit")
        elif tip_vocke == "kiwi fruit":
            #print ("kiwi je")
            kiwi_features.append(hog.compute(slika))
            labele.append("kiwi fruit")
        elif tip_vocke == "mango fruit":
            #print ("mango je")
            mango_features.append(hog.compute(slika))
            labele.append("mango fruit")
        elif tip_vocke == "orange fruit":
            #print("orange je")
            orange_features.append(hog.compute(slika))
            labele.append("orange fruit")
        else:
            #print ("strawberyy je")
            strawberry_features.append(hog.compute(slika))
            labele.append("strawberry fruit")

    apple_features = np.array(apple_features)
    banana_features = np.array(banana_features)
    cherry_features = np.array(cherry_features)
    chickoo_features = np.array(chickoo_features)
    grapes_features = np.array(grapes_features)
    kiwi_features = np.array(kiwi_features)
    mango_features = np.array(mango_features)
    orange_features = np.array(orange_features)
    strawberry_features = np.array(strawberry_features)

    x = np.vstack((apple_features, banana_features, cherry_features, chickoo_features, grapes_features,
                   kiwi_features, mango_features, orange_features, strawberry_features))  # v stack apnedovanje svih nasih featera >> to je nas obucavajuci skup podataka
    y = np.array(labele)

    return x, y, hog

def vrati_format(skup):
    #ako slika nije u odgovarajucem formatu postavimo je.. >> uklanjamo poslednju dimenziju slike
    for naziv, slika in skup.items():
        if slika.dtype != np.uint8:
            skup[naziv] = slika.astype(np.uint8)
        #slika = np.squeeze(slika, axis=-1)
        #skup[naziv] = slika
    return skup


if __name__ == '__main__':

    #podela podataka na trening, test i validacioni skup
    trening_skup_putanje, test_skup_putanje = ucitavanje_podataka()
    #print(validacioni_skup_putanje)

    #ucitavanje slika iz trening skupa podataka
    trening_skup = ucitavanje_slika(trening_skup_putanje)

    #ucitavanje slika iz validacionog skupa podataka
    #validacioni_skup = ucitavanje_slika(validacioni_skup_putanje)

    #predprocesiranje slika >> svoidmo ih na iste dimenzije
    sirina_resize = 128
    visina_resize = 128

    trening_skup = predprocesiranje_slike(trening_skup, sirina_resize, visina_resize)
    #validacioni_skup = predprocesiranje_slike(validacioni_skup, sirina_resize, visina_resize)

    #racunanje hog deskriptora
    x, y, hog = hog_priprema(trening_skup, sirina_resize, visina_resize)

    ######### PODELA TRENING SKUPA NA SKUP ZA OBUCAVANJE I VALIDACIONI SKUP #########
    x_treniranje, x_validacija, y_treniranje, y_validacija = train_test_split(x, y, test_size=0.05, random_state=42)
    print('Trening shape: ', x_treniranje.shape, y_treniranje.shape)
    print('Validacija shape: ', x_validacija.shape, y_validacija.shape)

    #x_treniranje = reshape_data(x_treniranje)
    #y_validacija = reshape_data(x_validacija)

    print('Treniranje shape: ', x_treniranje.shape, y_treniranje.shape)
    print('Test shape: ', x_validacija.shape, y_validacija.shape)

    ############ OBUCAVANJE KLASIFIKATORA #############
    print("Treniranje klasifikatora...")
    classifier = SVC(kernel="poly", C=1)
    classifier.fit(x_treniranje, y_treniranje)
    y_train_pred = classifier.predict(x_treniranje)
    y_test_pred = classifier.predict(x_validacija)
    print("Treniranje accuracy: ", accuracy_score(y_treniranje, y_train_pred))
    print("Validacija accuracy: ", accuracy_score(y_validacija, y_test_pred))

    ########### TESTIRANJE ###########
    test_skup = ucitavanje_slika(test_skup_putanje) #ucitavanje podataka za testiranje

    ## resenje ##
    ukupan_broj_slika_testiranje = 0
    broj_tacno_pogodjenih = 0

    print("************ RESENJE *************")
    print("<naziv slike> - <tacno resenje> - <dobijena predikcija>")
    print("")

    for naziv,slika in test_skup.items():

        ukupan_broj_slika_testiranje += 1

        #izdvajanje podataka o nazivu slike
        niz = naziv.split(",")
        tacan_rezultat_predikcije = niz[1]
        naziv_slike = niz[0]


        dobijeni_rezultat_predikcije = procesiranje_slike(slika, naziv_slike, hog)

        print(naziv_slike + " - " +  tacan_rezultat_predikcije + " - " + dobijeni_rezultat_predikcije[0])

        #provera da li je pogodjenrezultat prodikcije
        if (dobijeni_rezultat_predikcije[0] == tacan_rezultat_predikcije):
            broj_tacno_pogodjenih += 1 #povecavamo broj tacno pogodjenih slika

    ########### racunanje procenta tacnosti #########
    accuracy = 100 * broj_tacno_pogodjenih / ukupan_broj_slika_testiranje
    print("**** Accuracy je: ", accuracy, "%")



