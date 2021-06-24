import os
aa
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.preprocessing import image
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def CaricaDataset():
    # directory esempi positivi
    train_flessione_dir = os.path.join('dataset/train/pos')

    # directory esempi negativi
    train_noflessione_dir = os.path.join('dataset/train/neg')

    # stampa nomi immagini positive
    train_flessioni_names = os.listdir(train_flessione_dir)
    print("Positive: {}".format(train_flessioni_names[:10]))

    # stampa nomi immagini negative
    train_noflessioni_names = os.listdir(train_noflessione_dir)
    print("Negative: {}".format(train_noflessioni_names[:10]))

    # stampa il numero di immagini positive e di immagini negative
    print('Immagini di training positive:', len(os.listdir(train_flessione_dir)))
    print('Immagini di training negative:', len(os.listdir(train_noflessione_dir)))

    # parametri per il nostro grafo, le immagini saranno stampate in 4 righe e 4 colonne
    nrows = 4
    ncols = 4

    # indice per iterare sulle immagini
    pic_index = 0

    # inizializza matplotlib fig e lo dimensiona in 4x4
    fig = plt.gcf()
    fig.set_size_inches(ncols * 4, nrows * 4)

    pic_index += 8
    next_flessione_pic = [os.path.join(train_flessione_dir, fname) for fname in train_flessioni_names[pic_index - 8:pic_index]]
    next_noflessione_pic = [os.path.join(train_noflessione_dir, fname) for fname in train_noflessioni_names[pic_index - 8:pic_index]]

    for i, img_path in enumerate(next_flessione_pic + next_noflessione_pic):
        # configura subplot; gli indici di subplot partono da 1
        sp = plt.subplot(nrows, ncols, i + 1)
        sp.axis('Off')  # non mostra gli assi

        img = mpimg.imread(img_path)
        plt.imshow(img)

    plt.show()


def PreProcessingData():
    # tutte le immagini saranno riscalata da 1./255
    train_datagen = ImageDataGenerator(rescale=1 / 255)
    validation_datagen = ImageDataGenerator(rescale=1 / 255)

    # flusso di addestramento delle immagini in batches di 120 e 19 usando train_datagen generator e valid_datagen generator
    train_generator = train_datagen.flow_from_directory('dataset/train', classes=['pos', 'neg'], target_size=(200, 200), batch_size=120, class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory('dataset/valid', classes=['pos', 'neg'], target_size=(200, 200), batch_size=19, class_mode='binary', shuffle=False)

    return train_generator, validation_generator


def BuildModel():
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(200, 200, 3)), tf.keras.layers.Dense(128, activation=tf.nn.relu), tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)])
    model.summary()
    model.compile(optimizer=tf.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model


def Training(train_generator, validation_generator, model):
    history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1, validation_data=validation_generator, validation_steps=8)
    model.save('../build classifier/Classifier')


def Accuracy(model, validation_generator):
    model.evaluate(validation_generator)
    STEP_SIZE_TEST = validation_generator.n // validation_generator.batch_size

    validation_generator.reset()
    preds = model.predict(validation_generator, verbose=1)
    fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def Test(model):
    # predicting images
    path = 'img.jpg'
    img = image.load_img(path, target_size=(200, 200))
    x = image.img_to_array(img)
    plt.imshow(x / 255.)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    print(classes[0])

    if classes[0] < 0.5:
        print(" è flessione")
    else:
        print(" non è flessione")


def main():
    # costruisce classificatore
    # CaricaDataset()

    train_generator, validation_generator = PreProcessingData()
    model = BuildModel()
    Training(train_generator, validation_generator, model)
    Accuracy(model, validation_generator)

    # carica classificatore
    # esercizio = "Flessioni"
    # new_model = tf.keras.models.load_model('../../res/' + esercizio + '/Classificatore/Classifier')

    # testa classificatore
    # Test(new_model)


if __name__ == "__main__":
    main()
