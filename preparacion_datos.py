from conceptnet import *

RUTA_RESULTADOS_PARCIALES = "./resultados_parciales.txt"

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
    f = open(RUTA_RESULTADOS_PARCIALES, "a+")
    for tupla in bar:
        bar.set_description(" Preparando conceptos " + tupla[0] + " y " + tupla[1] + " con " + tupla[2])
        rel_1 = resultado_relaciones(bfs_conceptnet_v2(tupla[0], tupla[2]))
        rel_2 = resultado_relaciones(bfs_conceptnet_v2(tupla[1], tupla[2]))
        r = [tupla[0], tupla[1], tupla[2], sum(rel_1), sum(rel_2), rel_1, rel_2]
        f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f)
        resultado.append(r)
    #f.write(json.dumps(resultado))
    f.close()
    return resultado


preparar_datos("./data/test/test_triples.txt")



