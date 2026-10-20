"""
Reto AgroCebada 2026 - Modelo final y predicciones

Modelo elegido (ver validacion cruzada):
  1) Si la zona de clima de la parcela tiene parcelas de entrenamiento:
       prediccion = promedio suavizado del rendimiento de esa zona
  2) Si no tiene: prediccion con Ridge sobre las 9 variables (set C)

Entrada : dataset_final.csv
Salidas : predicciones_59_parcelas.csv
          modelo_final.joblib   (lo cargara la aplicacion web)
"""
import pandas as pd
import numpy as np
import joblib                                       # guardar/cargar objetos de Python

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# ---------------------------------------------------------------
# 0. PARAMETROS DEL MODELO (los mismos de la validacion cruzada)
# ---------------------------------------------------------------
K = 1        # "parcelas imaginarias" con el promedio general que se mezclan con la zona
ALPHA = 10   # fuerza de la penalizacion de Ridge
COLS_C = ["elevacion_m", "pendiente_grados", "prec_ciclo_mm", "tmin_ciclo", "tmax_ciclo",
          "ndvi_may_jun", "ndvi_jul_sep", "ndvi_oct", "caida_pico_oct"]

# ---------------------------------------------------------------
# 1. DATOS Y ZONAS
# ---------------------------------------------------------------
df = pd.read_csv("/Users/mayra_190/Documents/5toSemestre/cebada/code/datos_shapefile_extraidos/dataset_final.csv")

# Zona = mismo pixel de clima = mismos valores de tmin y precipitacion.
# Se calcula sobre las 197 parcelas para que entrenamiento y prediccion
# compartan la misma numeracion de zonas.
df["zona"] = df.groupby(["tmin_ciclo", "prec_ciclo_mm"]).ngroup()

trn = df[df.CONJUNTO == "ENTRENAMIENTO"].reset_index(drop=True)   # 138
prd = df[df.CONJUNTO == "PREDICCION"].reset_index(drop=True)      # 59
y = trn.RENDIMIENTO_T_HA

# ---------------------------------------------------------------
# 2. ENTRENAR (con las 138 parcelas completas)
# ---------------------------------------------------------------
mu = y.mean()   # promedio general del rendimiento

# Respaldo: Ridge con imputacion de nulos (pendiente) y estandarizacion
ridge_c = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        Ridge(alpha=ALPHA)).fit(trn[COLS_C], y)

# Tabla de zonas: para cada zona, suma y numero de parcelas de entrenamiento.
# Se guardan tambien tmin y prec para poder identificar la zona despues.
tabla_zona = trn.groupby("zona").agg(
    tmin_ciclo=("tmin_ciclo", "first"),
    prec_ciclo_mm=("prec_ciclo_mm", "first"),
    suma=("RENDIMIENTO_T_HA", "sum"),
    n=("RENDIMIENTO_T_HA", "count"),
)

# ---------------------------------------------------------------
# 3. FUNCION DE PREDICCION
# ---------------------------------------------------------------
def predice(nuevos):
    # Para cada parcela nueva: suma y n de su zona (0 si la zona no existe en train)
    s = nuevos.zona.map(tabla_zona["suma"]).fillna(0)
    n = nuevos.zona.map(tabla_zona["n"]).fillna(0)

    # Promedio de la zona suavizado: (suma + K*mu) / (n + K)
    pred_zona = (s + K * mu) / (n + K)
    pred_ridge = ridge_c.predict(nuevos[COLS_C])

    # Si la zona tiene datos -> promedio de zona; si no -> Ridge de respaldo
    pred = np.where(n > 0, pred_zona, pred_ridge)
    metodo = np.where(n > 0, "promedio_zona", "ridge_respaldo")
    return pred, metodo, n.astype(int)

# ---------------------------------------------------------------
# 4. PREDECIR LAS 59 PARCELAS Y GUARDAR
# ---------------------------------------------------------------
pred, metodo, n_zona = predice(prd)

salida = pd.DataFrame({
    "ID_POLIGONO": prd.ID_POLIGONO,
    "rendimiento_pred_t_ha": pred.round(2),
    "metodo": metodo,
    "n_train_zona": n_zona,     # cuantas parcelas de entrenamiento respaldan la prediccion
})
salida.to_csv("/Users/mayra_190/Documents/5toSemestre/cebada/code/predicciones_finales/predicciones_59_parcelas.csv", index=False)

# ---------------------------------------------------------------
# 5. GUARDAR EL MODELO PARA LA APLICACION WEB
# ---------------------------------------------------------------
joblib.dump({"mu": mu, "K": K, "cols_ridge": COLS_C,
             "ridge_c": ridge_c, "tabla_zona": tabla_zona},
            "modelo_final.joblib")

# ---------------------------------------------------------------
# 6. REVISIONES DE SENTIDO COMUN
# ---------------------------------------------------------------
print("Filas de salida:", len(salida), "(esperado 59)")
print("Nulos en predicciones:", salida.rendimiento_pred_t_ha.isna().sum(), "(esperado 0)")
print("\nMetodo usado:")
print(salida.metodo.value_counts())
print("\nPredicciones (59 parcelas):")
print(salida.rendimiento_pred_t_ha.describe().round(2))
print("\nRendimiento real, entrenamiento (138 parcelas):")
print(y.describe().round(2))
print("\nParcelas con respaldo Ridge o zona de 1 sola parcela:")
print(salida[salida.n_train_zona <= 1].to_string(index=False))