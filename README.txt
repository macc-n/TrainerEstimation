TrainerEstimation è un'applicazione pensata per aiutare gli utenti meno esperti nello svolgimento di esercizi fisici. Al momento l'unico esercizio supportato sono le flessioni. Per aggiungerne altri, bisogna solamente
inserire le relative risorse.


La struttura della repository si presenta nel seguente modo:

|–– doc
|    |–– Documentazione
|    |–– Presentazione
|    |–– video
|-- res
|    |-- flessioni
|    |-- sounds
|–– src
|    |–– build classifier
|    |–– build dataframe
|    |-- trainer estimation

Nel seguito si dettagliano i ruoli dei diversi componenti:

- doc: in questa cartella si trova tutta la documentazione relativa al progetto;

- res: contiene le risorse utilizzate dall'applicazione:
	- flessioni contiene il dataframe di 'esempio' dell'esercizio e il classificatore
	- sounds contiene i file audio che vengono utilizzati per l'interazione

- src: la cartella principale del progetto, in cui si trova il codice dell’applicazione:
	- build classifier: contiene lo script per creare un classificatore binario, dato un dataset
	- build dataframe: contiene lo script per creare un dataframe in formato Excel, dove ogni riga rappresenta un frame del video dato in input, mentre ogni coppia di colonne rappresenta le coordinate x e y di un landmark
	- trainer estimation: contiene il codice dell'applicazione. Per testarla, eseguire lo script "Main.py"
