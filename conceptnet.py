import conceptnet_lite
from conceptnet_lite import Label, edges_between, RelationName
from tqdm import tqdm
from collections import Counter
import os.path

import json
RUTA_BD = "./db/db_conceptnet.db"
RUTA_CACHE = "./cache/cache.txt"
conceptnet_lite.connect(RUTA_BD)

listado_relaciones = ['related_to', 'form_of', 'is_a', 'part_of', 'has_a','used_for', 'capable_of', 'at_location','causes',    'has_subevent',    'has_first_subevent',     'has_last_subevent',    'has_prerequisite',     'has_property',     'motivated_by_goal',     'obstructed_by',    'desires',    'created_by', 'synonym',    'antonym',    'distinct_from',    'derived_from',    'symbol_of',    'defined_as',    'manner_of',    'located_near',    'has_context',    'similar_to',    'etymologically_related_to',    'etymologically_derived_from',    'causes_desire',    'made_of',    'receives_action',    'external_url']

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

def obtener_relaciones_salida(concepto, language = 'en'):
    concepto_obj = obtener_etiquetas(concepto, language)
    relaciones = []
    for c in concepto_obj:
        if c.edges_out:
            for e in c.edges_out:
                if e.end.language == c.language:
                    relaciones.append({'concepto':e.end.text, 'relacion': e.relation.name})
    return relaciones
 
def obtener_relaciones_entrada(concepto, language = 'en'):
    concepto_obj = obtener_etiquetas(concepto, language)
    relaciones = []
    for c in concepto_obj:
        if c.edges_in:
            for e in c.edges_in:
                if e.start.language == c.language:
                    relaciones.append({'concepto':e.start.text, 'relacion': e.relation.name})
    return relaciones

def obtener_relaciones(concepto, language = 'en'):
    return obtener_relaciones_entrada(concepto, language) + obtener_relaciones_salida(concepto, language)

def bfs_conceptnet(concepto_inicio, concepto_final, max_iter = 1000, language='en'):
    cache = buscar_cache(concepto_inicio, concepto_final)
    if cache:
        return cache

    cola = [[{"concepto":concepto_inicio}]]
    visitado = []
    ##bar = tqdm(total=max_iter, initial=0)
    while cola:
        ##bar.update(1)
        max_iter = max_iter - 1
        if len(cola) == 0:
            return []
        camino = cola.pop(0)
        nodo = camino[-1]
        #tqdm.write("Cola pendiente: " + str(cola))
        tqdm.write("Buscando relaciones en " + nodo["concepto"])
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


def imprimir_relaciones(lista_relaciones):
    if len(lista_relaciones) == 0:
        print("No hay relaciones")
        return
    origen = lista_relaciones.pop(0)
    print(origen["concepto"])
    for relacion in lista_relaciones:
        print("\t" + relacion["relacion"] + " " + relacion["concepto"])
    print(lista_relaciones[-1]["concepto"])
    print(resultado_relaciones(lista_relaciones))
    
def resultado_relaciones(lista_relaciones):
    lista = []
    for relacion in lista_relaciones:
        if "relacion" in relacion:
            lista.append(relacion["relacion"])
    resultado = dict(Counter(lista))
    for tipo_relacion in listado_relaciones:
        if tipo_relacion not in resultado.keys():
            resultado[tipo_relacion] = 0
    return resultado

def buscar_cache(concepto_inicio, concepto_final, tipo = "BFS"):
    if os.path.isfile(RUTA_CACHE):
        f = open(RUTA_CACHE, "r")
        cache = json.loads(f.read())
        f.close()
        if concepto_inicio in cache[tipo] and concepto_final in cache[tipo][concepto_inicio]:
            return cache[tipo][concepto_inicio][concepto_final]
    else:
        return False

def guardar_cache(concepto_inicio, concepto_final, relacion, tipo = "BFS"):
    if not os.path.isfile(RUTA_CACHE) or os.path.getsize(RUTA_CACHE) == 0:
        print()
        f = open(RUTA_CACHE,"x")
        cache = {}
        cache[tipo] = {}
        cache[tipo][concepto_inicio] = {}
        cache[tipo][concepto_inicio][concepto_final] = relacion

        f.write(json.dumps(cache))
        f.close()
    else:
        r = open(RUTA_CACHE, "r")
        cache = json.loads(r.read())
        r.close()
        if tipo not in cache:
            cache = {}
            cache[tipo] = {}
        if concepto_inicio not in cache[tipo]:
            cache[tipo][concepto_inicio] = {} 
        cache[tipo][concepto_inicio][concepto_final] = relacion
        w = open(RUTA_CACHE, "w")
        w.write(json.dumps(cache))
        w.close()

imprimir_relaciones(bfs_conceptnet('gallery', 'drawing'))

    