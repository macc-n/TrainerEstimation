import Interfaccia as i


def valida_risposte(domanda, risposte_valide):
    valida = False
    risposta = ""
    domanda = "{} (".format(domanda)

    for i in range(0, len(risposte_valide)):
        domanda += risposte_valide[i]
        if i != len(risposte_valide) - 1:
            domanda += "|"
        else:
            domanda += ")"
    domanda += " : "

    while not valida:
        risposta = input(domanda).lower().capitalize()
        if len([risposta_valida for risposta_valida in risposte_valide if risposta_valida == risposta]) > 0:
            valida = True
        else:
            print("Mi dispiace, non conosco questo esercizio, scegli fra quelli proposti")
    return risposta


def main():

    while True:
        esercizio = valida_risposte("Quale esercizio vuoi svolgere?", ["Flessioni"])

        # se l'esercizio viene riconosciuto, si passa alla fase successiva
        m = i.main(esercizio)
        m.main()


if __name__ == "__main__":
    main()
