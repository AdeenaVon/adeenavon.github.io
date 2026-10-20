"""
Reto AgroCebada 2026 - Construccion del dataset final (1 fila por parcela)

Entradas (en DATA_DIR):
  - ID_area_rendimiento_70_30_Reto_AgroCebada.csv   (tabla maestra)
  - topografia_por_parcela.csv                      (elevacion, pendiente)
  - precipitacion_por_parcela.csv                   (mensual)
  - tmin_por_parcela.csv                            (mensual)
  - tmax_por_parcela.csv                            (mensual)
  - Conjunto_datos_PRO_AgroCebada.csv               (Planet 2025)

Salidas:
  - features_satelital.csv
  - dataset_final.csv   (197 filas, 1 por parcela)

Ciclo del cultivo con rendimiento asociado: abril-octubre 2025.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = "/Users/mayra_190/Documents/5toSemestre/cebada/code/datos_shapefile_extraidos"  # carpeta con los CSV de entrada 
CICLO_MESES = range(4, 11)  # abril (4) a octubre (10)
ANIO = 2025
NUBOSIDAD_MAX = 10      # regla de limpieza: excluir porcentaje_nubosidad > 10


def p(nombre):
    return os.path.join(DATA_DIR, nombre)


# ---------------------------------------------------------------
# 1. Tabla maestra
# ---------------------------------------------------------------
base = pd.read_csv("/Users/mayra_190/Documents/5toSemestre/cebada/DATASET_RETO_AGRO_2026/ID_area_rendimiento_70_30_Reto_AgroCebada.csv", encoding="utf-8-sig")
base = base[["ID_POLIGONO", "AREA_HA", "RENDIMIENTO_T_HA", "CONJUNTO"]]  # quita columna vacia

# ---------------------------------------------------------------
# 2. Topografia (ya viene 1 fila por parcela)
# ---------------------------------------------------------------
topo = pd.read_csv(p("topografia_por_parcela.csv"), encoding="utf-8-sig")

# ---------------------------------------------------------------
# 3. Clima del ciclo (abr-oct 2025)
# ---------------------------------------------------------------
def ciclo(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    return d[(d.anio == ANIO) & (d.mes.isin(CICLO_MESES))]

pr = ciclo(p("precipitacion_por_parcela.csv"))
tn = ciclo(p("tmin_por_parcela.csv"))
tx = ciclo(p("tmax_por_parcela.csv"))

clima = pd.DataFrame({
    "prec_ciclo_mm": pr.groupby("ID_POLIGONO").precipitacion_mm.sum(),
    "tmin_ciclo": tn.groupby("ID_POLIGONO").tmin_c.mean(),
    "tmax_ciclo": tx.groupby("ID_POLIGONO").tmax_c.mean(),
}).reset_index()
# (amplitud_termica = tmax - tmin se descarta: no agrega informacion nueva)

df = (base.merge(topo, on="ID_POLIGONO", how="left", validate="one_to_one")
          .merge(clima, on="ID_POLIGONO", how="left", validate="one_to_one"))

print("Base + topografia + clima:", df.shape)  # esperado (197, 9)

# ---------------------------------------------------------------
# 4. Features satelitales (Planet PRO, solo NDVI)
# ---------------------------------------------------------------
sat = pd.read_csv("/Users/mayra_190/Documents/5toSemestre/cebada/DATASET_RETO_AGRO_2026/Conjunto_datos_PRO_AgroCebada.csv", encoding="utf-8-sig")
sat = sat[sat.porcentaje_nubosidad <= NUBOSIDAD_MAX].copy()
sat["fecha"] = pd.to_datetime(sat["fecha_captura"])
sat["mes"] = sat["fecha"].dt.month
print("Filas satelitales limpias:", sat.shape[0])  # esperado 31708

# Promedio mensual de NDVI por parcela -> tabla parcela x mes
mensual = sat.groupby(["ID_POLIGONO", "mes"])["ndvi_promedio"].mean().reset_index()
ndvi = mensual.pivot(index="ID_POLIGONO", columns="mes", values="ndvi_promedio")

f = pd.DataFrame(index=ndvi.index)
f["ndvi_may_jun"] = ndvi[[5, 6]].mean(axis=1)      # arranque del cultivo
f["ndvi_jul_sep"] = ndvi[[7, 8, 9]].mean(axis=1)   # parte media del ciclo
f["ndvi_oct"] = ndvi[10]                           # final del ciclo
f["ndvi_pico"] = ndvi[list(CICLO_MESES)].max(axis=1)   # (solo auxiliar)
f["caida_pico_oct"] = f["ndvi_pico"] - f["ndvi_oct"]   # velocidad de senescencia
f = f.reset_index()

# Se guarda el set completo auxiliar, pero al dataset final solo van 4 columnas
f.to_csv(p("features_satelital.csv"), index=False)

sat_feat = f[["ID_POLIGONO", "ndvi_may_jun", "ndvi_jul_sep", "ndvi_oct", "caida_pico_oct"]]

# ---------------------------------------------------------------
# 5. Union final y validaciones
# ---------------------------------------------------------------
final = df.merge(sat_feat, on="ID_POLIGONO", how="left", validate="one_to_one")

print("\n=== VALIDACIONES ===")
print("shape:", final.shape)  # esperado (197, 12)
print("IDs duplicados:", final.ID_POLIGONO.duplicated().sum())  # 0
print(final.CONJUNTO.value_counts())  # 138 / 59
print("Nulos (solo RENDIMIENTO=59 y pendiente=3):")
print(final.isna().sum()[lambda s: s > 0])
print("Rendimiento nulo en ENTRENAMIENTO:",
      final[(final.CONJUNTO == "ENTRENAMIENTO") & final.RENDIMIENTO_T_HA.isna()].shape[0])  # 0

feats = ["elevacion_m", "pendiente_grados", "prec_ciclo_mm", "tmin_ciclo",
         "tmax_ciclo", "ndvi_may_jun", "ndvi_jul_sep", "ndvi_oct", "caida_pico_oct"]
trn = final[final.CONJUNTO == "ENTRENAMIENTO"]
print("\nCorrelaciones (solo entrenamiento):")
print(trn[feats + ["RENDIMIENTO_T_HA"]].corr().round(2))

final.to_csv(p("dataset_final.csv"), index=False)
print("\nGuardado: dataset_final.csv")