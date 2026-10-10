# ds-project2

Proyecto 2 de Data Science (UVG) · Reto [MAP – Charting Student Math Misunderstandings](https://www.kaggle.com/competitions/map-charting-student-math-misunderstandings).

Clasificación de las explicaciones de estudiantes en la etiqueta `Category:Misconception` (64 clases).

## Instalación

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt
```

`scikit-learn` está fijo en la versión 1.7.2 porque el modelo M1 se guardó con ella.

## Orden de los notebooks

| Notebook | Qué hace | Dónde se corre |
|---|---|---|
| `00-eda-limpieza.ipynb` | Limpieza → `data/train_clean.csv` | Local |
| `01-eda-exploratorio.ipynb` | Análisis exploratorio | Local |
| `02-particion-y-preprocesamiento.ipynb` | Partición train/val/test y reglas derivadas | Local |
| `03-modelo-M1-tfidf.ipynb` | M1: TF-IDF + regresión logística | Local (CPU, ~15 min) |
| `04-modelo-M3-deberta.ipynb` | M3: DeBERTa-v3 con ajuste fino | **Kaggle con GPU** (~1 h) |

## Pesos de los modelos

Los modelos no se suben a GitHub (pesan demasiado). Descargarlos y colocarlos así:

```
models/
├── m1_tfidf_lr.joblib
└── m3_deberta/            (config.json, model.safetensors, tokenizer..., config_proyecto.json)
```

Enlace de descarga: **_pendiente (Google Drive)_**

## Cómo correr M3 en Kaggle

1. Hacer `git push` del repositorio con `data/splits/particion.csv` y las reglas generadas por el notebook `02`.
2. En Kaggle: *New Notebook → File → Import Notebook* y subir `04-modelo-M3-deberta.ipynb`.
3. Panel derecho: **Accelerator → GPU T4 x2** e **Internet → On** (requiere cuenta verificada con teléfono).
4. *Run All*. Para probar primero, poner `PRUEBA_RAPIDA = True` en la celda de configuración.
5. Descargar `salida_M3.zip` de la pestaña *Output* y descomprimirlo en la raíz del repositorio.

## Predecir desde código (o desde la app)

```python
from src.prediccion import predecir, predecir_todos, modelos_disponibles

modelos_disponibles()            # ['M1_tfidf_lr', 'M3_deberta']
predecir("M1_tfidf_lr", 31772, r"\( \frac{3}{6} \)", "3 triangles are white and 6 are blue")
# [{'etiqueta': '...', 'category': '...', 'misconception': '...', 'prob': 0.95}, ...]  (top 3)
```

La respuesta puede ir en LaTeX (como en los datos) o normalizada (`"3/6"`). El preprocesamiento ocurre dentro de la función.

## Estructura

```
data/            datos, partición (splits/particion.csv) y reglas derivadas del split de entrenamiento
src/             preprocesamiento, predicción, métricas y utilidades compartidas
models/          pesos de los modelos (fuera de Git)
results/         predicciones sobre test, tuning y tiempos
documentacion/   documentos del proyecto
```
