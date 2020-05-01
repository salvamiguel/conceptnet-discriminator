from conceptnet import *



def leer_palabras(archivo):
    r = open(archivo, "r")
    lines = r.readlines()
    datos = []
    for line in lines:
        datos.append(tuple(i for i in line.replace("\n", "").split(',')))
    return datos

def preparar_datos(archivo):
    tuplas = leer_palabras(archivo)
    resultado = []
    bar = tqdm(tuplas)
    for tupla in bar:
        bar.set_description(" Preparando tupla " + tupla[0] + " y " + tupla[1] + " con " + tupla[2])
        r = [tupla[0], tupla[1], resultado_relaciones(bfs_conceptnet(tupla[0], tupla[2]))]
        resultado.append(r)
    return resultado


print(preparar_datos("./data/test/test_triples.txt"))



