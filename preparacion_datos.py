from conceptnet import *



def leer_palabras(archivo):
    r = open(archivo, "r")
    lines = r.readlines()
    datos = []
    for line in lines:
        datos.append(tuple(i for i in line.replace("\n", "").split(',')))
    return datos


