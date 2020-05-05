from conceptnet import *

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


def preparar_datos_hilos(archivo, num_hilos = 10):
    tuplas = leer_palabras(archivo)
    pos = 0
    entrada = Queue()
    salida = Queue()
    lock_cache = Lock()
    lock_salida = Lock()
    hilos = []
    for linea in tuplas:
        l_pos = (pos,)
        l = l_pos + linea
        entrada.put(l)
        pos = pos + 1

    def preparar(num_hilo, lock_cache, lock_salida, entrada):
        while not entrada.empty():
            linea = entrada.get()
            posicion = linea[0]
            print("[Hilo " + str(num_hilo) + "]: Obtiene la linea " + str(posicion))
            rel_1 = resultado_relaciones(bfs_conceptnet_v3(num_hilo, linea[1], linea[3], lock_cache))
            rel_2 = resultado_relaciones(bfs_conceptnet_v3(num_hilo, linea[2], linea[3], lock_cache))
            r = [posicion, linea[1], linea[2], linea[3], rel_1, rel_2, int(linea[4])]
            salida.put([posicion]+r)
            print("[Hilo " + str(num_hilo) + "]: Esperando lock para escribir")
            lock_salida.acquire()
            try:
                print("[Hilo " + str(num_hilo) + "]: Empieza a escribir")
                w = open(RUTA_RESULTADOS_PARCIALES, 'a+')
                w.write(json.dumps(r) + "\n")
                w.flush()
                os.fsync(w)
            finally:
                lock_salida.release()
        hilos[num_hilo].join()     

    for i in range(0,num_hilos):
        p = Process(target=preparar, args=(i, lock_cache, lock_salida, entrada)).start()
        hilos.append(p)
        #p.join()
    
    #return list(salida.queue)
    
    
    



preparar_datos_hilos("./data/test/ref/truth.txt")



