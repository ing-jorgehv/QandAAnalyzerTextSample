import numpy as np  
from sklearn.feature_extraction.text import TfidfVectorizer  
from sklearn.metrics.pairwise import cosine_similarity  

  
# Las intenciones y sus respuestas correspondientes  

intentions = [
    "Hola",
    "¿Qué es Python?",  
    "¿Cómo se escribe un comentario en Python?",  
    "¿Cómo se declara una variable en Python?",  
    "¿Cómo se define una función en Python?",  
    "¿Cómo se crea una lista en Python?",  
    "¿Cómo se realiza un bucle for en Python?",  
    "¿Cómo se realiza un bucle while en Python?",  
    "¿Cómo se manejan las excepciones en Python?",  
    "¿Qué hace la función 'print' en Python?",  
    "¿Cómo se importa un módulo en Python?",
    'Gracias'
    ]


responses = [
    "¡Hola, cómo puedo ayudarte hoy?",
    "Python es un lenguaje de programación interpretado de alto nivel, creado por Guido van Rossum en 1991. Se caracteriza por su sintaxis clara y legible.",  
    "Los comentarios en Python se escriben comenzando la línea con un símbolo de almohadilla (#). Por ejemplo: # Este es un comentario.",  
    "En Python, se declara una variable asignándole un valor. Por ejemplo: mi_variable = 5.",  
    "En Python, una función se define con la palabra clave 'def', seguida del nombre de la función y paréntesis. Por ejemplo: def mi_funcion():",  
    "Las listas en Python se crean encerrando una secuencia de valores entre corchetes y separándolos con comas. Por ejemplo: mi_lista = [1, 2, 3, 4, 5].",  
    "En Python, un bucle for se realiza con la palabra clave 'for' seguida de una variable, la palabra 'in' y una secuencia. Por ejemplo: for i in range(5):",  
    "Un bucle while se realiza con la palabra clave 'while' seguida de una condición. Por ejemplo: while i < 5:",  
    "En Python, las excepciones se manejan con los bloques 'try' y 'except'. El código que puede lanzar una excepción se coloca dentro del bloque 'try', y el código que se ejecutará si ocurre una excepción se coloca dentro del bloque 'except'.",  
    "La función 'print' en Python imprime el argumento que se le pasa a la consola. Por ejemplo: print('Hola, mundo!') imprimirá 'Hola, mundo!' en la consola.",  
    "En Python, se importa un módulo usando la palabra clave 'import' seguida del nombre del módulo. Por ejemplo: import math.",
    "!De nada! No dudes en preguntar si tienes más preguntas."  
    ]

  
# Crear un vectorizador y entrenarlo en las intenciones
# Usa TfidfVectorizer para convertir las intenciones en vectores  
vectorizer = TfidfVectorizer().fit(intentions)  
  
def get_response(user_input):  
    # Convierte la entrada del usuario en un vector  
    user_vector = vectorizer.transform([user_input])  
  
    # Calcula la similitud de coseno entre la entrada del usuario y las intenciones  
    similarities = cosine_similarity(user_vector, vectorizer.transform(intentions))  
  
    # Encuentra la intención más similar  
    closest = np.argmax(similarities)

    # Si la similtud máxima está por debajo del umbral, devulve un mensaje por defecto
    if similarities[0, closest] < 0.6:
        return 'Lo siento, no entiendo tu pregunta. Por favor reformula tu pregunta'
  
    # De lo contrario, devuelve la respuesta correspondiente a esta intención  
    return responses[closest]  

  
# Probemos el chatbot con algunas preguntas  

for i, intention in enumerate(intentions, start=1):  
    print(str(i) + ".- " + intention + ": " + get_response(intention))  

print("¿Cómo te llamas?: " + get_response('¿Cómo te llamas?'))  
