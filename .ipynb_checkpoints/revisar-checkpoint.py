import os

def verificar_sincronizacion_rgb(ruta_base):
    carpetas = ["00_15", "00_16", "00_17"]
    listas_filtradas = {}

    # 1. Escanear carpetas y filtrar archivos
    for carpeta in carpetas:
        ruta_completa = os.path.join(ruta_base, carpeta)
        
        if not os.path.exists(ruta_completa):
            print(f"Error: La carpeta {carpeta} no existe en la ruta proporcionada.")
            return

        # Listar, filtrar por "rgb" y quitar el prefijo para comparar nombres base
        archivos = os.listdir(ruta_completa)
        # Filtramos y guardamos el nombre original para la ordenación final
        rgb_files = [f for f in archivos if "rgb" in f.lower()]
        
        # Guardamos el "nombre base" (sin el prefijo 00_XX) para comparar entre carpetas
        # Ejemplo: "00_15_imagen_1.jpg" -> "imagen_1.jpg"
        nombres_base = {f.split('_', 2)[-1] for f in rgb_files}
        listas_filtradas[carpeta] = nombres_base

    # 2. Encontrar la intersección (archivos que están en las 3)
    c15, c16, c17 = carpetas
    comunes = listas_filtradas[c15] & listas_filtradas[c16] & listas_filtradas[c17]

    # 3. Verificar si hay archivos faltantes en alguna carpeta
    todas_las_bases = listas_filtradas[c15] | listas_filtradas[c16] | listas_filtradas[c17]
    
    esta_sincronizado = True
    for carpeta in carpetas:
        faltantes = todas_las_bases - listas_filtradas[carpeta]
        if faltantes:
            esta_sincronizado = False
            print(f"--- Archivos faltantes en {carpeta} ---")
            for f in sorted(faltantes):
                print(f"Falta: {f}")

    # 4. Mostrar resultado final ordenado alfabéticamente
    print("\n" + "="*30)
    if esta_sincronizado:
        print("✅ ¡ÉXITO! Las tres carpetas están perfectamente sincronizadas.")
        print(f"Total de archivos RGB comunes: {len(comunes)}")
        print("\nLista ordenada de archivos comunes (nombre base):")
        for f in sorted(comunes):
            print(f"- {f}")
    else:
        print("❌ ERROR: Las carpetas no contienen los mismos archivos.")
        print(f"Archivos que sí coinciden en las tres: {len(comunes)}")

# --- Configuración ---
# Cambia '.' por la ruta de la carpeta que contiene las 3 subcarpetas
ruta_principal = "MultiViewVisibleThermalImagesHPE/test" 
verificar_sincronizacion_rgb(ruta_principal)