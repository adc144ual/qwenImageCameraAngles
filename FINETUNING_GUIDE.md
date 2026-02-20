# Fine-tuning Qwen-Image-Edit para Dataset Multi-Vista

Sistema completo para entrenar y usar un modelo Qwen-Image-Edit personalizado con tu dataset de 3 cámaras sincronizadas.

## 📋 Requisitos

### Hardware
- GPU con mínimo 16GB VRAM (recomendado: 24GB+)
- 32GB+ RAM del sistema
- ~50GB espacio en disco

### Software
```bash
pip install torch torchvision transformers diffusers accelerate peft pillow tqdm tensorboard
```

## 📁 Estructura del Dataset

Tu dataset debe seguir esta estructura:

```
/dataset_root/
    train_val/
        00_17/  # Cámara frontal (0 grados)
            00_17_timestamp_rgb.png
            ...
        00_16/  # Cámara derecha (+90 grados)
            00_16_timestamp_rgb.png
            ...
        00_15/  # Cámara izquierda (-90 grados)
            00_15_timestamp_rgb.png
            ...
    test/       # (Opcional) Estructura idéntica para validación/test
        00_17/
        00_16/
        00_15/
```
Notas:
- Las imágenes deben estar sincronizadas por timestamp.
- El script busca coincidencias de timestamp entre las carpetas de las cámaras.
- Se asume:
    - 00_17: Frontal (0°)
    - 00_16: Derecha (+90°)
    - 00_15: Izquierda (-90°)

**Importante:**
- Las imágenes con el mismo timestamp deben estar sincronizadas entre cámaras
- Los nombres de archivo deben seguir el patrón: `XX_YY_TIMESTAMP_rgb.png`
- Todas las personas deben tener imágenes en las 3 cámaras

## 🚀 Proceso Completo

### Paso 1: Verificar el Dataset

Antes de entrenar, verifica que tu dataset esté correctamente estructurado:

```bash
python src/prepare_dataset.py \
    --dataset_root "/home/jupyter-antoniodetoro/nas/qwen/datos/MultiViewVisibleThermalImagesHPE" \
    --show_commands
```

Este script:
- ✅ Verifica la estructura del dataset
- 📊 Genera estadísticas (personas, imágenes, sincronización)
- 💡 Proporciona recomendaciones
- 🚀 Muestra comandos de ejemplo para entrenar

**Salida esperada:**
```
📊 Analizando dataset...
  └─ Personas con todas las vistas: 50
  └─ Imágenes sincronizadas totales: 5,000
  └─ Pares aproximados: 30,000

✅ Dataset parece adecuado para entrenamiento
```

### Paso 2: Entrenar el Modelo

#### Entrenamiento Básico (Recomendado)

```bash
python src/train_multiview_finetuning.py \
    --dataset_root "../datos/MultiViewVisibleThermalImagesHPE" \
    --output_dir "../models/finetuned_multiview_v1" \
    --batch_size 1 \
    --epochs 10 \
    --learning_rate 1e-4 \
    --resolution 512 \
    --lora_rank 16 \
    --seed 42
```

**Parámetros importantes:**
- `--batch_size`: Tamaño de batch (1-2 para 16GB VRAM)
- `--epochs`: Épocas de entrenamiento (10-20 recomendado)
- `--learning_rate`: Learning rate (1e-4 funciona bien)
- `--resolution`: Resolución de imágenes (512 o 768)
- `--lora_rank`: Rango de LoRA (16-32, mayor = más capacidad pero más memoria)

#### Entrenamiento Rápido (Para Pruebas)

```bash
python src/train_multiview_finetuning.py \
    --dataset_root "../datos/MultiViewVisibleThermalImagesHPE" \
    --output_dir "../models/test_run" \
    --batch_size 2 \
    --epochs 3 \
    --learning_rate 5e-5 \
    --resolution 384 \
    --lora_rank 8
```

#### Entrenamiento de Alta Calidad

```bash
python src/train_multiview_finetuning.py \
    --dataset_root "../datos/MultiViewVisibleThermalImagesHPE" \
    --output_dir "../models/finetuned_multiview_hq" \
    --batch_size 2 \
    --epochs 20 \
    --learning_rate 1e-4 \
    --resolution 768 \
    --lora_rank 32
```

**Durante el entrenamiento:**
- Los logs se guardan en `output_dir/logs/` (visualiza con TensorBoard)
- Los checkpoints se guardan cada 500 steps en `output_dir/checkpoint-XXXX/`
- Las imágenes de validación se guardan en `output_dir/validation_*.png`

```bash
# Ver progreso con TensorBoard
tensorboard --logdir ../models/finetuned_multiview_v1/logs
```

### Paso 3: Usar el Modelo Entrenado

#### Modo Single (Una imagen)

```bash
python src/inference_finetuned.py \
    --lora_path "../models/finetuned_multiview_v1/final_lora" \
    --input_image "../datos/test_images/person_01.png" \
    --output "../resultados/person_01_rot90.png" \
    --angle 90 \
    --steps 12 \
    --resolution 512
```

#### Modo Batch (Múltiples imágenes)

```bash
python src/inference_finetuned.py \
    --lora_path "../models/finetuned_multiview_v1/final_lora" \
    --input_dir "../datos/test_images/" \
    --output "../resultados/rotaciones/" \
    --angles "90,-90,180" \
    --steps 12 \
    --resolution 512
```

#### Con Background y Máscara

```bash
python src/inference_finetuned.py \
    --lora_path "../models/finetuned_multiview_v1/final_lora" \
    --input_image "../datos/test_images/person_01.png" \
    --output "../resultados/person_01_composite.png" \
    --angle 90 \
    --background "../datos/background.png" \
    --mask "../datos/mask.png" \
    --steps 12
```

**Parámetros:**
- `--angle`: Ángulo único (90, -90, o 180)
- `--angles`: Múltiples ángulos separados por coma
- `--steps`: Pasos de inferencia (6-20, más = mejor calidad)
- `--resolution`: Resolución de salida
- `--seed`: Semilla para reproducibilidad

## 📊 Monitoreo y Evaluación

### Ver Métricas de Entrenamiento

```bash
tensorboard --logdir ../models/finetuned_multiview_v1/logs --port 6006
```

Métricas clave:
- **Loss**: Debe decrecer consistentemente
- **Learning Rate**: Sigue el schedule configurado
- **Imágenes de Validación**: Se generan cada 250 steps

### Evaluar Calidad

Después del entrenamiento, evalúa con:

1. **Coherencia de pose**: ¿La persona mantiene su postura?
2. **Realismo**: ¿La imagen se ve natural?
3. **Estabilidad MediaPipe**: ¿MediaPipe detecta landmarks consistentemente?

```python
import mediapipe as mp
from PIL import Image

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Probar con imagen generada
img = Image.open("resultado_rot90.png")
results = pose.process(np.array(img))

if results.pose_landmarks:
    print("✅ Pose detectada correctamente")
else:
    print("❌ Pose no detectada - necesita más entrenamiento")
```

## 🔧 Troubleshooting

### Error: Out of Memory (OOM)

**Solución 1: Reducir batch size**
```bash
--batch_size 1
```

**Solución 2: Reducir resolución**
```bash
--resolution 384
```

**Solución 3: Reducir LoRA rank**
```bash
--lora_rank 8
```

**Solución 4: Aumentar gradient accumulation**
```bash
# En train_multiview_finetuning.py, línea 50:
gradient_accumulation_steps: int = 8  # Aumentar de 4 a 8
```

### El modelo genera imágenes borrosas

- Aumenta `num_inference_steps` a 16-20
- Verifica que el dataset tenga buena calidad
- Entrena por más épocas
- Aumenta `lora_rank` a 32

### La pose no se mantiene coherente

- El dataset puede necesitar más ejemplos
- Considera aumentar épocas de entrenamiento
- Ajusta `learning_rate` (prueba 5e-5)
- Verifica que las imágenes estén bien sincronizadas

### Imágenes completamente negras

- Usa `torch.bfloat16` en lugar de `float16`
- Verifica la configuración de VAE
- Revisa que las imágenes de entrada estén normalizadas

## 💡 Tips de Optimización

### Para Mejor Calidad

1. **Dataset balanceado**: Asegura igual número de ejemplos por ángulo
2. **Limpieza de datos**: Elimina imágenes borrosas o mal sincronizadas
3. **Augmentation**: Considera flip horizontal para duplic ar datos
4. **Training largo**: 15-20 épocas suelen dar mejores resultados

### Para Menor Uso de Memoria

```python
# Modificar en TrainingConfig (línea 50):
gradient_accumulation_steps: int = 8      # De 4 a 8
train_batch_size: int = 1                 # Batch size 1
resolution: int = 384                     # Resolución menor
lora_rank: int = 8                        # LoRA rank menor
```

### Para Entrenamiento Más Rápido

```python
# Usar resolución menor temporalmente
--resolution 384 --epochs 5
```

## 📈 Ejemplo Completo de Flujo

```bash
# 1. Verificar dataset
python src/prepare_dataset.py \
    --dataset_root "../datos/MultiViewVisibleThermalImagesHPE" \
    --show_commands

# 2. Entrenar (10 épocas, ~4-6 horas en A30)
python src/train_multiview_finetuning.py \
    --dataset_root "../datos/MultiViewVisibleThermalImagesHPE" \
    --output_dir "../models/my_model_v1" \
    --batch_size 1 \
    --epochs 10 \
    --resolution 512

# 3. Monitorear
tensorboard --logdir ../models/my_model_v1/logs

# 4. Probar modelo
python src/inference_finetuned.py \
    --lora_path "../models/my_model_v1/final_lora" \
    --input_image "../datos/test.png" \
    --output "../resultados/test_rot90.png" \
    --angle 90 \
    --steps 12

# 5. Generar batch
python src/inference_finetuned.py \
    --lora_path "../models/my_model_v1/final_lora" \
    --input_dir "../datos/validation_set/" \
    --output "../resultados/batch/" \
    --angles "90,-90,180" \
    --steps 12
```

## 📝 Notas Adicionales

- **Checkpoints**: Se guardan automáticamente cada 500 steps
- **Resumir entrenamiento**: Usa `--resume_from_checkpoint path/to/checkpoint`
- **Multi-GPU**: El script usa Accelerate, soporta multi-GPU automáticamente
- **Validación**: Se ejecuta cada 2 épocas por defecto

## 🎯 Siguiente Paso: Integración con MediaPipe

Una vez entrenado el modelo, puedes usar las imágenes generadas con MediaPipe:

```python
import mediapipe as mp
from PIL import Image
import numpy as np

# Cargar imagen generada
generated_img = Image.open("resultado_rot90.png")

# Detectar pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
results = pose.process(np.array(generated_img))

# Extraer landmarks
if results.pose_landmarks:
    landmarks = results.pose_landmarks.landmark
    # Usar landmarks como si fueran de imagen real
    for idx, landmark in enumerate(landmarks):
        print(f"Landmark {idx}: x={landmark.x}, y={landmark.y}, z={landmark.z}")
```

---

## Resumen Visual del flujo:
```
INPUT IMAGE (person, camera 0°)
    ↓
[VAE Encoder] ❄️ Frozen
    ↓
SOURCE LATENTS + PROMPT → [Text Encoder] ❄️ Frozen → EMBEDDINGS
    ↓                                                      ↓
TARGET IMAGE (person, camera 90°)                         │
    ↓                                                      │
[VAE Encoder] ❄️ Frozen                                   │
    ↓                                                      │
TARGET LATENTS                                             │
    ↓                                                      │
Add noise (Flow Matching) → NOISY_LATENTS                 │
    ↓                           ↓                          │
    │      ┌──────────────────────────────────────────────┘
    │      ↓
    │  [TRANSFORMER 🔥 LoRA Trainable]
    │      ├─ Timestep embedding
    │      ├─ 60× QwenImageTransformerBlock
    │      │     ├─ Image + Text dual stream
    │      │     ├─ Joint Attention (Q/K/V ← LoRA)
    │      │     └─ FFN
    │      └─ Output projection
    │          ↓
    │   PREDICTED_VELOCITY
    │          ↓
    └──→ LOSS = MSE(predicted, target_velocity)
              ↓
         Backward() → Update LoRA weights only
```



**¿Preguntas o problemas?** Revisa los logs en `output_dir/logs/` o ajusta los hiperparámetros según las secciones de troubleshooting.
