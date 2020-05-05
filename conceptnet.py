import conceptnet_lite
from conceptnet_lite import Label, edges_between, RelationName
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, Lock, Manager
import os
import numpy as np
import json

RUTA_BD = "./db/db_conceptnet.db"
RUTA_CACHE = "./cache/cache.txt"
conceptnet_lite.connect(RUTA_BD)

lista_todas_relaciones = ['dbpedia/genus','not_has_property','not_desires','entails','instance_of', 'related_to', 'form_of', 'is_a', 'part_of', 'has_a','used_for', 'capable_of', 'at_location','causes', 'has_subevent', 'has_first_subevent', 'has_last_subevent', 'has_prerequisite',     'has_property',     'motivated_by_goal',     'obstructed_by',    'desires',    'created_by', 'synonym',    'antonym',    'distinct_from',    'derived_from',    'symbol_of',    'defined_as',    'manner_of',    'located_near',    'has_context',    'similar_to',    'etymologically_related_to',    'etymologically_derived_from',    'causes_desire',    'made_of',    'receives_action',    'external_url']

def obtener_etiquetas(label, language='en'):
    """Obtiene las etiquetas correspondientes 

    Arguments:
        label {str} -- Etiqueta

    Keyword Arguments:
        language {str} -- Idioma en el que se hace la búsqueda de la etiqueta (default: {'en'})

    Returns:
        peewee.ModelSelect -- Array de Concepts
    """
    return Label.get(text = label, language = language).concepts
    
def obtener_concepto(label, language='en'):
    """Obtiene el primer concepto que coincide con la etiqueta pasada

    Arguments:
        label {str} -- Etiqueta/Concepto a buscar

    Keyword Arguments:
        language {str} -- Idioma en el que se hace la búsqueda de la etiqueta (default: {'en'})

    Returns:
        [Concept] -- Concepto encontrado
    """
    return obtener_etiquetas(label=label, language=language)[0]


def obtener_relaciones_directas(c1, c2):
    """Obtiene las relaciones directas en un diccionario 

    Arguments:
        c1 {peewee.ModelSelect} -- Etiqueta del primer Concepto
        c2 {peewee.ModelSelect} -- Etiqueta del segundo Concepto

    Returns:
        [dict] -- Diccionario donde cada entrada corresponde al tipo de relacion y el valor el número de relaciones de ese tipo
    """
    relaciones = []
    for relacion in edges_between(c1, c2, two_way=False):
       relaciones.append(relacion.relation.name)
    return Counter(relaciones)



def obtener_relaciones_salida(concepto, language='en'):
    try:
        concepto_obj = obtener_etiquetas(concepto, language)
    except:
        print("Concepto no encontrado " + concepto)
        return []
    def iterar_conceptos(c):
        if c.edges_out:
            return map(iterar_relaciones, c.edges_out)
    def iterar_relaciones(e):
        if str(e.end.language) == language:
            return {'concepto': e.end.text, 'relacion': e.relation.name, "direccion": 1}
    return [y for x in list(map(iterar_conceptos, concepto_obj)) if x is not None for y in x if y is not None]
 


def obtener_relaciones_entrada(concepto, language = 'en'):
    try:
        concepto_obj = obtener_etiquetas(concepto, language)
    except:
        print("Concepto no encontrado " + concepto)
        return []
    sol = []
    def iterar_conceptos(c):
        if c.edges_in:
            map(iterar_relaciones, c.edges_in)
    def iterar_relaciones(e):
        if str(e.start.language) == language:
            sol.append({'concepto': e.start.text, 'relacion': e.relation.name, "direccion": -1})
    return [y for x in list(map(iterar_conceptos, concepto_obj)) if x is not None for y in x if y is not None]


def obtener_relaciones(concepto, language = 'en'):
    return obtener_relaciones_entrada(concepto, language) + obtener_relaciones_salida(concepto, language)

def bfs_conceptnet(concepto_inicio, concepto_final, max_iter = 100, language='en'):
    cache = buscar_cache(concepto_inicio, concepto_final)
    if cache or (isinstance(cache, list) and len(cache) >= 0):
        return cache

    cola = [[{"concepto":concepto_inicio}]]
    visitado = []
    ##bar = tqdm(total=max_iter, initial=0)
    while cola and max_iter is not 0:
        ##bar.update(1)
        max_iter = max_iter - 1
        if len(cola) == 0:
            return []
        camino = cola.pop(0)
        nodo = camino[-1]
        #tqdm.write("Cola pendiente: " + str(cola))
        tqdm.write("Buscando relaciones desde " + concepto_inicio + " en " + nodo["concepto"] + " hasta " + concepto_final)
        if nodo["concepto"] == concepto_final:
            guardar_cache(concepto_inicio, concepto_final, camino)
            return camino
        elif nodo not in visitado:
            for vecino in obtener_relaciones(nodo["concepto"], language):
                if vecino["concepto"] == concepto_final:
                    camino.append(vecino)
                    guardar_cache(concepto_inicio, concepto_final, camino)
                    return camino
                #tqdm.write("\t->\tEncontrada: " + nodo["concepto"] + " " + vecino["relacion"] + " " + vecino["concepto"])
                nuevo_camino = list(camino)
                nuevo_camino.append(vecino)
                cola.append(nuevo_camino)
            visitado.append(nodo)

def bfs_conceptnet_v2(concepto_inicio, concepto_final, max_iter = 100, language='en'):
    cache = buscar_cache(concepto_inicio, concepto_final, language, 'BFS_v2')
    lista_caminos = []
    if cache or (isinstance(cache, list) and len(cache) >= 0):
        return cache
    cola = [[{"concepto":concepto_inicio}]]
    visitado = []
    ##bar = tqdm(total=max_iter, initial=0)
    while cola and max_iter is not 0:
        ##bar.update(1)
        max_iter = max_iter - 1
        if len(cola) == 0:
            break
        camino = cola.pop(0)
        nodo = camino[-1]
        #tqdm.write("Cola pendiente: " + str(cola))
        tqdm.write("Buscando relaciones desde " + concepto_inicio + " en " + nodo["concepto"] + " hasta " + concepto_final)
        if nodo["concepto"] == concepto_final:
            #guardar_cache(concepto_inicio, concepto_final, camino)
            lista_caminos.append(camino)
        elif nodo not in visitado:
            for vecino in obtener_relaciones(nodo["concepto"], language):
                if vecino["concepto"] == concepto_final:
                    camino.append(vecino)
                    #guardar_cache(concepto_inicio, concepto_final, camino)
                    lista_caminos.append(camino)
                else:
                    #tqdm.write("\t->\tEncontrada: " + nodo["concepto"] + " " + vecino["relacion"] + " " + vecino["concepto"])
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    cola.append(nuevo_camino)
            visitado.append(nodo)
    lista_caminos = list(map(json.loads, set(map(json.dumps, lista_caminos))))
    guardar_cache(concepto_inicio, concepto_final, lista_caminos, language, 'BFS_v2')
    return lista_caminos

def bfs_conceptnet_v3(num_hilo, concepto_inicio, concepto_final, lock, max_iter = 100, language='en'):
    cache = buscar_cache(concepto_inicio, concepto_final, language, 'BFS_v2')
    lista_caminos = []
    if cache or (isinstance(cache, list) and len(cache) >= 0):
        return cache
    cola = [[{"concepto":concepto_inicio}]]
    visitado = []
    ##bar = tqdm(total=max_iter, initial=0)
    while cola and max_iter is not 0:
        ##bar.update(1)
        max_iter = max_iter - 1
        if len(cola) == 0:
            break
        camino = cola.pop(0)
        nodo = camino[-1]
        #tqdm.write("Cola pendiente: " + str(cola))
        tqdm.write("\t[Hilo " + str(num_hilo) + "]: Buscando relaciones desde " + concepto_inicio + " en " + nodo["concepto"] + " hasta " + concepto_final)
        if nodo["concepto"] == concepto_final:
            #guardar_cache(concepto_inicio, concepto_final, camino)
            lista_caminos.append(camino)
        elif nodo not in visitado:
            for vecino in obtener_relaciones(nodo["concepto"], language):
                if vecino["concepto"] == concepto_final:
                    camino.append(vecino)
                    #guardar_cache(concepto_inicio, concepto_final, camino)
                    lista_caminos.append(camino)
                else:
                    #print("\t->\tEncontrada: " + nodo["concepto"] + " " + vecino["relacion"] + " " + vecino["concepto"])
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    cola.append(nuevo_camino)
            visitado.append(nodo)
    lista_caminos = list(map(json.loads, set(map(json.dumps, lista_caminos))))
    lock.acquire()
    try:
        guardar_cache(concepto_inicio, concepto_final, lista_caminos, language, 'BFS_v2')
    finally:
        lock.release()
    return lista_caminos

def imprimir_relaciones(lista_relaciones):
    if not isinstance(lista_relaciones, list) or len(lista_relaciones) == 0:
        print("No hay relaciones")
        return
    origen = lista_relaciones.pop(0)
    print(origen["concepto"])
    for relacion in lista_relaciones:
        print("\t" + relacion["relacion"] + " " + relacion["concepto"])
    print(lista_relaciones[-1]["concepto"])
    print(resultado_relaciones(lista_relaciones))
    
def resultado_relaciones(lista_relaciones, retornar_lista = True):
    if not isinstance(lista_relaciones, list):
        lista_relaciones = []
    lista_salientes = []
    lista_entrantes = []

    flat_lista_relaciones = []

    if len(lista_relaciones) >= 1 and isinstance(lista_relaciones[0], list):
        for sub_lista in lista_relaciones:
            for r in sub_lista:
                flat_lista_relaciones.append(r)
    else:
        flat_lista_relaciones = lista_relaciones
 
    for relacion in flat_lista_relaciones:
        if "relacion" in relacion:
            if relacion["direccion"] == 1:
                lista_salientes.append(relacion["relacion"])
            else:
                lista_entrantes.append(relacion["relacion"])
    dict_relaciones_salientes = dict(Counter(lista_salientes))
    dict_relaciones_entrantes = dict(Counter(lista_entrantes))

    for tipo_relacion in lista_todas_relaciones:
        if tipo_relacion not in dict_relaciones_salientes.keys():
            dict_relaciones_salientes[tipo_relacion] = 0
        if tipo_relacion not in dict_relaciones_entrantes.keys():
            dict_relaciones_entrantes[tipo_relacion] = 0
    
    if retornar_lista:
        lista_resultado = [[],[]]
        for relacion in sorted(dict_relaciones_salientes):
            lista_resultado[0].append(dict_relaciones_salientes[relacion])
        for relacion in sorted(dict_relaciones_entrantes):
            lista_resultado[1].append(dict_relaciones_entrantes[relacion])
        #print(lista_resultado)
        return lista_resultado
    else:
        return {"salientes": dict_relaciones_salientes, "entrantes": dict_relaciones_entrantes}


def buscar_cache(concepto_inicio, concepto_final, language = 'en', tipo = "BFS"):
    if os.path.isfile(RUTA_CACHE):
        f = open(RUTA_CACHE, "r")
        cache = json.loads(f.read())
        f.close()
        if language in cache and tipo in cache[language] and concepto_inicio in cache[language][tipo] and concepto_final in cache[language][tipo][concepto_inicio]:
            return cache[language][tipo][concepto_inicio][concepto_final]
    else:
        return False

def guardar_cache(concepto_inicio, concepto_final, relacion, language = 'en', tipo = "BFS"):
    if not os.path.isfile(RUTA_CACHE) or os.path.getsize(RUTA_CACHE) == 0:
        f = open(RUTA_CACHE,"x")
        cache = {}
        cache[language] = {}
        cache[language][tipo] = {}
        cache[language][tipo][concepto_inicio] = {}
        cache[language][tipo][concepto_inicio][concepto_final] = relacion

        f.write(json.dumps(cache))
        f.close()
    else:
        r = open(RUTA_CACHE, "r")
        cache = json.loads(r.read())
        r.close()
        if language not in cache:
            cache = {}
            cache[language] = {}
        if tipo not in cache[language]:
            cache[language][tipo] = {}
        if concepto_inicio not in cache[language][tipo]:
            cache[language][tipo][concepto_inicio] = {} 
        cache[language][tipo][concepto_inicio][concepto_final] = relacion
        w = open(RUTA_CACHE, "w")
        w.write(json.dumps(cache))
        w.close()


#print(obtener_relaciones("pencil"))
#print(a)

#a = resultado_relaciones(bfs_conceptnet_v2('grapefruit', 'peel'), False)
#print(a)
#print(len(a["salientes"].keys()))
#print(len(a["entrantes"].keys()))
#for key in a["salientes"].keys():
#    if key not in a["entrantes"].keys():
#        print(key)

    