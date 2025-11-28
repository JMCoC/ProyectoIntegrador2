import json
import os
import random
import re
import tempfile
from datetime import datetime

import requests
import sounddevice as sd
import speech_recognition as sr
from scipy.io.wavfile import write

SRATE = 16000  # tasa de muestreo
DUR = 5        # segundos
DATA_FILE = "recordatorios_personales.json"


def grabar_audio():
    print("Grabando... describe tu plan o petición ahora.")
    audio = sd.rec(int(DUR * SRATE), samplerate=SRATE, channels=1, dtype="int16")
    sd.wait()
    print("Listo, procesando...")
    tmp_wav = tempfile.mktemp(suffix=".wav")
    write(tmp_wav, SRATE, audio)
    return tmp_wav


def transcribir(path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(path) as source:
        data = recognizer.record(source)
    return recognizer.recognize_google(data, language="es-ES")


def cargar_recordatorios():
    if not os.path.exists(DATA_FILE):
        return []
    try:
        with open(DATA_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return []


def guardar_recordatorios(items):
    with open(DATA_FILE, "w", encoding="utf-8") as fh:
        json.dump(items, fh, ensure_ascii=False, indent=2)


def agregar_recordatorio(texto):
    recordatorios = cargar_recordatorios()
    recordatorios.append({
        "texto": texto.strip(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
    })
    guardar_recordatorios(recordatorios)
    print(f"✅ Recordatorio guardado: {texto.strip()}")


def listar_recordatorios():
    recordatorios = cargar_recordatorios()
    if not recordatorios:
        print("Aún no tienes recordatorios guardados.")
        return
    print("Tus recordatorios:")
    for idx, item in enumerate(recordatorios, 1):
        print(f" {idx}. {item['texto']} ({item['timestamp']})")


def limpiar_recordatorios():
    guardar_recordatorios([])
    print("Se limpiaron todos los recordatorios.")


def motivacion():
    frases = [
        "Respira hondo, enfoca tu objetivo y da el siguiente paso.",
        "Pequeños avances diarios construyen resultados gigantes.",
        "Tu voz marcó el plan, ahora actúa para cumplirlo.",
        "La disciplina de hoy es la libertad de mañana."
    ]
    print(random.choice(frases))


def obtener_definicion(termino):
    termino = termino.strip()
    if not termino:
        print("No escuché qué palabra definir.")
        return
    url = f"https://api.dictionaryapi.dev/api/v2/entries/es/{termino}"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print(f"No encontré la definición de '{termino}'.")
            return
        data = resp.json()
        definicion = data[0]["meanings"][0]["definitions"][0]["definition"]
        print(f"Definición de {termino}: {definicion}")
    except Exception as exc:
        print("No se pudo consultar la definición:", exc)


def obtener_clima(ciudad):
    ciudad = ciudad.strip()
    if not ciudad:
        print("No escuché la ciudad para el clima.")
        return
    url = f"https://wttr.in/{ciudad}?format=j1"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            print("No pude obtener el clima.")
            return
        data = resp.json()
        actual = data["current_condition"][0]
        desc = actual["weatherDesc"][0]["value"]
        temp = actual["temp_C"]
        sensacion = actual["FeelsLikeC"]
        print(f"Clima en {ciudad.title()}: {desc}, {temp} °C (sensación {sensacion} °C)")
    except Exception as exc:
        print("Error al consultar el clima:", exc)


def ejecutar_comando(texto):
    handled = False

    match_recordatorio = re.search(
        r"(?:agrega|agregar)\s+(?:un\s+)?recordatorio(?:\s+(?:de|para))?\s+(.*)",
        texto,
        flags=re.IGNORECASE
    )
    if match_recordatorio:
        agregar_recordatorio(match_recordatorio.group(1))
        return True

    if re.search(r"(?:listar|muestra)\s+recordatorios", texto, re.IGNORECASE):
        listar_recordatorios()
        return True

    if re.search(r"(?:limpia|borra)\s+recordatorios", texto, re.IGNORECASE):
        limpiar_recordatorios()
        return True

    if re.search(r"motívame|motivame|necesito motivación", texto, re.IGNORECASE):
        motivacion()
        return True

    match_def = re.search(r"definición de (.+)", texto, re.IGNORECASE)
    if match_def:
        obtener_definicion(match_def.group(1))
        return True

    match_clima = re.search(r"clima en (.+)", texto, re.IGNORECASE)
    if match_clima:
        obtener_clima(match_clima.group(1))
        return True

    if re.search(r"(?:plan|ruta) del día", texto, re.IGNORECASE):
        listar_recordatorios()
        motivacion()
        return True

    return handled


def main():
    tmp_wav = grabar_audio()
    try:
        texto = transcribir(tmp_wav)
        print("Dijiste:", texto)
        if not ejecutar_comando(texto):
            print(
                "No reconocí un comando de la rutina personalizada. "
                "Prueba con 'agregar recordatorio', 'listar recordatorios', "
                "'definición de', 'clima en' o 'motívame'."
            )
    except sr.UnknownValueError:
        print("No se entendió el audio.")
    except sr.RequestError as err:
        print("Error al contactar el servicio de reconocimiento:", err)
    finally:
        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)


if __name__ == "__main__":
    main()