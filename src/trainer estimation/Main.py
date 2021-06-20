import speech_recognition as sr
from playsound import playsound

import Interfaccia as i


def valida_risposte(domanda, risposte_valide):
    valida = False
    risposta = ""

    while not valida:
        print("Quale esercizio vuoi svolgere? [Flessioni]", end='')
        rispostaScritta = input().lower().capitalize()
        rispostaVocale = SpeechRecognition().lower().capitalize()

        if rispostaScritta is not None:
            risposta = rispostaScritta
        else:
            if rispostaVocale is not None:
                risposta = rispostaVocale

        if len([risposta_valida for risposta_valida in risposte_valide if risposta_valida == risposta]) > 0:
            valida = True
        else:
            playsound("../../res/Sounds/errors/err_scelta.mp3")
    return risposta


def SpeechRecognition():
    text = ""
    recognizer_instance = sr.Recognizer()  # Crea una istanza del recognizer

    with sr.Microphone() as source:
        recognizer_instance.adjust_for_ambient_noise(source)
        audio = recognizer_instance.listen(source)
    try:
        text = recognizer_instance.recognize_google(audio, language="it-IT")
        print(text)
    except Exception as e:
        print(e)

    return text


def main():
    playsound("../../res/Sounds/mess/benvenuto.mp3")

    while True:
        playsound("../../res/Sounds/mess/scelta.mp3")
        esercizio = valida_risposte("Quale esercizio vuoi svolgere?", ["Flessioni"])

        # se l'esercizio viene riconosciuto, si passa alla fase successiva
        m = i.main(esercizio)
        m.main()


if __name__ == "__main__":
    main()
