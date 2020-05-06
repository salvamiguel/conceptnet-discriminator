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
        rel_1 = resultado_relaciones(bfs_conceptnet_v2(tupla[0], tupla[2]))
        rel_2 = resultado_relaciones(bfs_conceptnet_v2(tupla[1], tupla[2]))
        r = [tupla[0], tupla[1], tupla[2], rel_1, rel_2, int(tupla[3])]
        f.write(json.dumps(r) + "\n")
        f.flush()
        os.fsync(f)
        resultado.append(r)
    #f.write(json.dumps(resultado))
    f.close()
    return resultado


def preparar(lock_cache, lock_salida, cache, entrada):
    num_hilo = os.getpid()
    linea = entrada
    posicion = linea[0]

    cache_c = cache

    rel_1 = resultado_relaciones(bfs_conceptnet_v3(linea[1], linea[3], cache,20))
    rel_2 = resultado_relaciones(bfs_conceptnet_v3(linea[2], linea[3], cache,20))
    r = [posicion, linea[1], linea[2], linea[3], rel_1, rel_2, int(linea[4])]

    #lock_cache.acquire()
    #try:
    #    w = open(RUTA_CACHE, "w")
    #    w.write(json.dumps(cache.copy()))
    #    w.close()
    #finally:
    #    lock_cache.release()

    lock_salida.acquire()
    try:
        w = open(RUTA_RESULTADOS_PARCIALES, 'a+')
        w.write(json.dumps(r) + "\n")
        w.flush()
        os.fsync(w)
    finally:
        lock_salida.release()

def preparar_datos_hilos(archivo, num_saltos=100, num_hilos=False):
    if not num_hilos:
        num_hilos = os.cpu_count()
    tuplas = leer_palabras(archivo)
    pos = 0
    pool = Pool(processes=num_hilos)
    m = Manager()
    lock_cache = m.Lock()
    lock_salida = m.Lock()
    f = open(RUTA_CACHE, "r")
    cache = m.dict(json.loads(f.read()))
    f.close()

    lineas = []
    #entrada = m.Queue()
    #salida = m.Queue()
    for linea in tuplas:
        l_pos = (pos,)
        l = l_pos + linea
        lineas.append(l)
        pos = pos + 1

    func = partial(preparar, lock_cache, lock_salida, cache)

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
argparser.add_argument('-s', '--salida', help='Ruta del archivo de salida')
argparser.add_argument('-', '--epochs', help='Número de epochs', default=40)
args = argparser.parse_args()



preparar_datos_hilos("./data/test/ref/truth.txt")



