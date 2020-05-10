from conceptnet import *
from functools import partial
import argparse

RUTA_RESULTADOS_PARCIALES = "./resultados_parciales.txt"

def leer_palabras(archivo):
    r = open(archivo, "r")
    lines = r.readlines()
    datos = []

    for line in lines:
        datos.append(tuple() + tuple(i for i in line.replace("\n", "").split(',')))
    return datos

def preparar_datos(archivo):
    tuplas = leer_palabras(archivo)
    resultado = []
    bar = tqdm(tuplas)
    f = open(RUTA_RESULTADOS_PARCIALES, "a+")
    for tupla in bar:
        bar.set_description(" Preparando conceptos " + tupla[0] + " y " + tupla[1] + " con " + tupla[2])
        rel_1 = calculate_result(bfs_conceptnet_v2(tupla[0], tupla[2]))
        rel_2 = calculate_result(bfs_conceptnet_v2(tupla[1], tupla[2]))
        r = [tupla[0], tupla[1], tupla[2], rel_1, rel_2, int(tupla[3])]
        f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f)
        resultado.append(r)
    #f.write(json.dumps(resultado))
    f.close()
    return resultado


def preparar(lock_salida, num_saltos, entrada):
    posicion = entrada[0]

    rel_1 = calculate_result(bfs_conceptnet_v3(entrada[1], entrada[3], num_saltos))
    rel_2 = calculate_result(bfs_conceptnet_v3(entrada[2], entrada[3], num_saltos))
    r = [posicion, entrada[1], entrada[2], entrada[3], rel_1, rel_2, int(entrada[4])]


    lock_salida.acquire()
    try:
        w = open(RUTA_RESULTADOS_PARCIALES, 'a+')
        w.write(json.dumps(r) + "\n")
        w.flush()
        os.fsync(w)
    finally:
        lock_salida.release()

def preparar_datos_hilos(archivo, num_saltos=50, num_hilos=False):
    if not num_hilos:
        num_hilos = os.cpu_count()
    tuplas = leer_palabras(archivo)
    pos = 0
    pool = Pool(processes=num_hilos)
    m = Manager()
    lock_salida = m.Lock()


    lineas = []
    #entrada = m.Queue()
    #salida = m.Queue()
    for linea in tuplas:
        l_pos = (pos,)
        l = l_pos + linea
        lineas.append(l)
        pos = pos + 1

    func = partial(preparar, lock_salida, num_saltos)

    with tqdm(total=len(lineas)) as pbar:
            for i, _ in enumerate(pool.imap(func, lineas)):
                pbar.update()



    pool.close()
    pool.join()



    #for i in range(0,num_hilos):
    #    p = Process(target=preparar, args=(i, lock_cache, lock_salida, entrada))
    #    p.start()
    #    hilos.append(p)
    #    print(hilos)
        #p.join()
    
    #return list(salida.queue)
    


argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--entrada', help='Ruta del archivo de entrada')
args = argparser.parse_args()



preparar_datos_hilos(args.entrada)



