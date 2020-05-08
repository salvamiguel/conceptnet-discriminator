import conceptnet_lite
from conceptnet_lite import Label, edges_between, RelationName
from tqdm import tqdm
from collections import Counter
from multiprocessing import Pool, Lock, Manager
import os
import numpy as np
import json
from pymongo import MongoClient




RUTA_CACHE = "./cache/cache.txt"

RUTA_BD = "./db/db_conceptnet.db"
conceptnet_lite.connect(RUTA_BD)

lista_todas_relaciones = ['antonym','at_location','capable_of','causes','causes_desire','created_by','defined_as','derived_from','desires','distinct_from','entails','etymologically_derived_from','etymologically_related_to','external_url','form_of','has_a','has_context','has_first_subevent','has_last_subevent','has_prerequisite','has_property','has_subevent','instance_of','is_a','located_near','made_of','manner_of','motivated_by_goal','not_capable_of','not_desires','not_has_property','not_used_for','obstructed_by','part_of','receives_action','related_to','similar_to','symbol_of','synonym','used_for','dbpedia/capital','dbpedia/field','dbpedia/genre','dbpedia/genus','dbpedia/influenced_by','dbpedia/known_for','dbpedia/language','dbpedia/leader','dbpedia/occupation','dbpedia/product']

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

def bfs_conceptnet_v3(concepto_inicio, concepto_final, max_iter = 100, language='en'):
    cliente_mongo = MongoClient('mongodb://127.0.0.1:27017/')
    db_c_cache = cliente_mongo.conceptnet.cache
    tipo = "BFS_v3"

    cursor = db_c_cache.find(
        {"language": language,
        "tipo": tipo,
        "concepto_i": concepto_inicio,
        "concepto_f": concepto_final
        })
    
    lista_caminos = []
    
    for r in cursor:
        lista_caminos = lista_caminos + r["relaciones"]

    if len(lista_caminos) > 0 or cursor.count() > 0:
        #tqdm.write("[Hilo "+str(os.getpid())+"]: Relaciones encontradas en cache")
        return lista_caminos    

    cola = [[{"concepto":concepto_inicio}]]
    visitado = []
    while cola and max_iter is not 0:
        max_iter = max_iter - 1
        if len(cola) == 0:
            break
        camino = cola.pop(0)
        nodo = camino[-1]
        if nodo["concepto"] == concepto_final:
            lista_caminos.append(camino)
        elif nodo not in visitado:
            #tqdm.write("\t[Hilo "+str(os.getpid())+"]: Buscando en " + nodo["concepto"])
            for vecino in obtener_relaciones(nodo["concepto"], language):
                db_c_cache.insert_one(
                    {"language": language,
                    "tipo": tipo,
                    "concepto_i": nodo["concepto"],
                    "concepto_f": vecino["concepto"],
                    "relaciones": [[{"concepto": nodo["concepto"]}, {"concepto": vecino["concepto"], "relacion": vecino["concepto"], "direccion": vecino["direccion"]}]]
                    })
                if vecino["concepto"] == concepto_final:
                    camino.append(vecino)
                    lista_caminos.append(camino)
                else:
                    nuevo_camino = list(camino)
                    nuevo_camino.append(vecino)
                    cola.append(nuevo_camino)
            visitado.append(nodo)
    lista_caminos = list(map(json.loads, set(map(json.dumps, lista_caminos))))
    db_c_cache.insert_one(
        {"language": language,
        "tipo": tipo,
        "concepto_i": concepto_inicio,
        "concepto_f": concepto_final,
        "relaciones": lista_caminos})
    #tqdm.write("[Hilo "+str(os.getpid())+"]: Relaciones encontradas")
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

#a = resultado_relaciones(bfs_conceptnet_v2('psalms', 'sing', 30), False)
#print(len(a["salientes"].keys()))
#print(a)
#print(len(a["entrantes"].keys()))
#for key in a["salientes"].keys():
#    if key not in a["entrantes"].keys():
#        print(key)

    