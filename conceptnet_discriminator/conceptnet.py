import os
import argparse
import json
import conceptnet_lite
from functools import partial
from collections import Counter
from multiprocessing import Pool, Lock, Manager
from conceptnet_lite import *
from tqdm import tqdm
import numpy as np
from pymongo import MongoClient
import gensim 
from gensim.models import Word2Vec, KeyedVectors

print("Loading wordembeddings model.")
path_embeddings = 'pre-trained/googlenews.gz'
m_embedding = KeyedVectors.load_word2vec_format(path_embeddings, binary=True)


RUTA_BD = "./db/db_conceptnet.db"
conceptnet_lite.connect(RUTA_BD)

lista_todas_relaciones = ['antonym','at_location','capable_of','causes','causes_desire','created_by','defined_as','derived_from','desires','distinct_from','entails','etymologically_derived_from','etymologically_related_to','external_url','form_of','has_a','has_context','has_first_subevent','has_last_subevent','has_prerequisite','has_property','has_subevent','instance_of','is_a','located_near','made_of','manner_of','motivated_by_goal','not_capable_of','not_desires','not_has_property','not_used_for','obstructed_by','part_of','receives_action','related_to','similar_to','symbol_of','synonym','used_for','dbpedia/capital','dbpedia/field','dbpedia/genre','dbpedia/genus','dbpedia/influenced_by','dbpedia/known_for','dbpedia/language','dbpedia/leader','dbpedia/occupation','dbpedia/product']

def get_labels(word, language='en'):
    """
    Get all labels from a word
    // Obtiene las etiquetas correspondientes 

    Arguments:
        word {str} -- Label // Etiqueta

    Keyword Arguments:
        language {str} -- Search's language, use two letter strings // Idioma en el que se hace la búsqueda de la etiqueta, usar identificador de dos letras (default: {'en'})

    Returns:
        peewee.ModelSelect -- List of labels matching // Array de labels encontrados
    """
    return Label.get(text=word, language=language)
    
def get_concept(word, language='en'):
    """
    Get concepts labed as word parameter
    Obtiene el primer concepto que coincide con la etiqueta pasada

    Arguments:
        word {str} -- Word to search // Palabra a buscar

    Keyword Arguments:
        language {str} -- Search's language, use two letter strings // Idioma en el que se hace la búsqueda de la etiqueta, usar identificador de dos letras (default: {'en'})

    Returns:
        list<Concept> --  List of concepts matching // Array de concepts encontrados
    """
    return get_labels(word=word, language=language).concepts


def get_direct_relations(c1, c2):
    """
    Gets only direct relations between concepts
    Obtiene las relaciones directas

    Arguments:
        c1 {peewee.ModelSelect} -- Start concept // Concepto inicial
        c2 {peewee.ModelSelect} -- End concept // Concepto final

    Returns:
        dict -- Diccionario donde cada entrada corresponde al tipo de relacion y el valor el número de relaciones de ese tipo
    """
    relaciones = []
    for con in c1:
        concept = con.text
        break
    for relacion in edges_between(c1, c2, two_way=False):
        if relacion.start.text == concept:
            relaciones.append([{"concepto": relacion.end.text, "relacion": relacion.relation.name, "direccion": 1}])
        else:
            relaciones.append([{"concepto": relacion.start.text, "relacion": relacion.relation.name, "direccion": -1}])
    return relaciones



def get_out_relations(word, language='en', min_cosine = 0):
    """Gets out relations from concept // Obtiene las relaciones de salida de un concepto

    Arguments:
        word {str} -- Word which we lookup //  La palabra objeto de la búsqueda

    Keyword Arguments:
        language {str} -- Search's language, use two letter strings // Idioma en el que se hace la búsqueda de la etiqueta, usar identificador de dos letras (default: {'en'})

    Returns:
        list -- Returns list with out relations // Retorna una lista con las relaciones de salida
    """
    try:
        concepto_obj = get_concept(word, language)
    except:
        print("Concepto no encontrado " + word)
        return []
    def iterar_conceptos(c):
        if c.edges_out:
            return map(iterar_relaciones, c.edges_out)
    def iterar_relaciones(e):
        if str(e.end.language) == language:
            try:
                cosine = m_embedding.similarity(word, e.end.text)
            except:
                cosine = 0.2
            if cosine > min_cosine:
                #print("Encontrado " + str(e.end.text) + " con " + str(cosine))
                return {'concepto': e.end.text, 'relacion': e.relation.name, "direccion": 1}
    return [y for x in list(map(iterar_conceptos, concepto_obj)) if x is not None for y in x if y is not None]
 


def get_in_relations(word, language='en', min_cosine = 0):
    """Gets out relations from concept // Obtiene las relaciones de salida de un concepto

    Arguments:
        word {str} -- Word which we lookup //  La palabra objeto de la búsqueda

    Keyword Arguments:
        language {str} -- Search's language, use two letter strings // Idioma en el que se hace la búsqueda de la etiqueta, usar identificador de dos letras (default: {'en'})

    Returns:
        list -- Returns list with in relations // Retorna una lista con las relaciones de salida
    """
    try:
        concepto_obj = get_concept(word, language)
    except:
        print("Concepto no encontrado " + word)
        return []
    sol = []
    def iterar_conceptos(c):
        if c.edges_in:
            map(iterar_relaciones, c.edges_in)
    def iterar_relaciones(e):
        if str(e.start.language) == language:
            try:
                cosine = m_embedding.similarity(word, e.start.text)
            except:
                cosine = 0.2
            if cosine > min_cosine:
                #print("Encontrado " + str(e.start.text) + " con " + str(cosine))
                sol.append({'concepto': e.start.text, 'relacion': e.relation.name, "direccion": -1})
    return [y for x in list(map(iterar_conceptos, concepto_obj)) if x is not None for y in x if y is not None]


def get_relations(word, language='en', min_cosine=0):
    """Gets both in and out relations from concept // Obtine las relaciones de entrada y salida desde un concepto

    Arguments:
        word {str} -- Word which we lookup //  La palabra objeto de la búsqueda

    Keyword Arguments:
        language {str} -- Search's language, use two letter strings // Idioma en el que se hace la búsqueda de la etiqueta, usar identificador de dos letras (default: {'en'})
        min_cosine {int} -- Min cosine similarity from relations (default: {0})

    Returns:
        list -- Concatenated list with all relations
    """
    return get_in_relations(word, language, min_cosine) + get_out_relations(word, language, min_cosine)


def calculate_result(relations_list):
    if not isinstance(relations_list, list):
        relations_list = []
    lista_salientes = []
    lista_entrantes = []
    flat_lista_relaciones = []
    if len(relations_list) >= 1 and isinstance(relations_list[0], list):
        for sub_lista in relations_list:
            prof = 1
            for r in sub_lista:
                r["prof"] = prof
                flat_lista_relaciones.append(r)
                prof = prof + 1

    dict_relaciones_salientes = {}
    dict_relaciones_entrantes = {}

    for tipo_relacion in sorted(lista_todas_relaciones):
        dict_relaciones_salientes[tipo_relacion] = 0
        dict_relaciones_entrantes[tipo_relacion] = 0

   
    for relacion in flat_lista_relaciones:
        if "relacion" in relacion and relacion["relacion"] in lista_todas_relaciones:
            if relacion["direccion"] == 1:
                dict_relaciones_salientes[relacion["relacion"]] = dict_relaciones_salientes[relacion["relacion"]] + 1 / int(relacion["prof"])
            else:
                dict_relaciones_entrantes[relacion["relacion"]] = dict_relaciones_salientes[relacion["relacion"]] + 1 / int(relacion["prof"])
     
    return [list(dict_relaciones_salientes.values()), list(dict_relaciones_entrantes.values())]




def a_star_threads(w1, w2, h_fun="adaptative", language='en', start_cosine = 0.25, max_jumps=3):
    cosine = start_cosine
    result = []  
    try:
        c1 = get_concept(word=w1, language=language)
        c2 = get_concept(word=w2, language=language)
        direct_relations = get_direct_relations(c1, c2)
    except Exception as e:
        print(str(e))
        return []

    if len(direct_relations) == 0:
        queue = [[{"concepto": w1}]]
        visited = []
        prof = 0
        cosines = [[start_cosine]]
        while queue:
            path = queue.pop(0)
            prof = len(path)
            if len(cosines) - 1 < prof:
                cosines.append([cosine])
            cosine = np.mean(cosines[prof])
            node = path[-1]
            if node not in visited:
                for v in get_relations(word=node["concepto"], language=language, min_cosine=cosine):
                    if v["concepto"] == w2:
                        path.append(v)
                        result.append(path)
                    else:
                        new_path = list(path)
                        new_path.append(v)
                        queue.append(new_path)
                    if h_fun == "adaptative":
                        try:
                            cosines[prof].append(m_embedding.similarity(w1, v["concepto"]))
                        except:
                            cosines[prof].append(0.2)

                    elif h_fun == "progressive":
                        cosine = min((cosine + 0.1, 0.95))
                        #print("Ajustando el cosine a " + str(cosine) + ", profundidad " + str(prof))
                if len(result) > 0 or prof > max_jumps or all(len(x) > max_jumps for x in queue):
                    return result
                visited.append(node)
        return result
    else:
        return direct_relations


