########################################################################
## SCRIPT PARA EL CÁLCULO DE LAG DE FORMA GLOBAL CON DTW  ###############
###########################################################################
## Recibe el fichero de datos iniciales y devuelve in fichero de datos que incluye 
## la corrección de y_pred en función de lo calculado por CCF
##############################################################################

# LIBRERÍAS NECESARIAS

import numpy as np
import pandas as pd
from collections import defaultdict

#####################################################################################
#####################################################################################

# ELECCIÓN DE LOS FICHEROS A ANALIZAR

lista_modelo = ['LSTM'] #['CNN','Linear','LSTM']
lista_horizonte = [4] # [2,4,6,8] # [2,4,6,8,12,16]
dataset = 'DiaTrend'

min_muestras = 50
freq = "15min"

#################################################################################
################################################################################
# FUNCIÓN QUE IMPLEMENTA DTW
#################################################
# Recibe: series temporales a comparar
# Ofrece: warp óptimo y distancia asociada

def dtw_subsequence_y(x, y):
    # Asegurar que sean arrays de numpy
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Dimensiones de las series a comparar
    N = len(x)
    M = len(y)
    
    # Matriz de costes locales
    cost_matrix = (x[:, np.newaxis] - y[np.newaxis, :]) ** 2
            
    # Matriz de costes acumulados
    accumulated_cost = np.zeros((N, M))
    
    # CONDICIÓN DE CONTORNO: Inicio libre para Y.
    # Puede empezarse desde cualquier punto de la fila 0
    accumulated_cost[0, :] = cost_matrix[0, :]
    
    # CONDICIÓN DE CONTORNO: X con extremos fijos
    for i in range(1, N):
        accumulated_cost[i, 0] = accumulated_cost[i-1, 0] + cost_matrix[i, 0]
        
    # Llenar el resto de la matriz
    for i in range(1, N):
        for j in range(1, M):
            accumulated_cost[i, j] = cost_matrix[i, j] + min(
                accumulated_cost[i-1, j],    # Inserción
                accumulated_cost[i, j-1],    # Borrado
                accumulated_cost[i-1, j-1]   # Coincidencia
            )
            
    # CONDICIÓN DE CONTORNO: Final libre para Y.
    last_row = accumulated_cost[-1, :]
    # Buscamos en la última fila de matrices de coste, la celda con menor coste acumulado
    best_j_end = np.argmin(last_row)
    
    # Distancia final
    min_distance = np.sqrt(last_row[best_j_end])
    
    # Reconstrucción del camino (Backtracking)
    # desde la celda final elegida
    path = []
    i, j = N - 1, best_j_end
    path.append((i, j))
    
    # Nos detenemos cuando i llega a 0 (X se ha recorrido por completo)
    while i > 0:
        if j == 0:
            i -= 1
        else:
            # Evaluamos los 3 movimientos posibles hacia atrás
            steps = [
                accumulated_cost[i-1, j],   # diagonal hacia arriba (i-1, j)
                accumulated_cost[i, j-1],   # izquierda (i, j-1)
                accumulated_cost[i-1, j-1]  # diagonal (i-1, j-1)
            ]
            best_step = np.argmin(steps)
            
            if best_step == 0:
                i -= 1
            elif best_step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1
        path.append((i, j))
        
    # El punto de inicio de la subsecuencia en Y será el j final tras el bucle
    # RETORNA: distancia del warp y los emparejamientos del warp
    return min_distance, path[::-1]

####################################################################################
##################################################################################
# FUNCIÓN PARA CORREGIR LA PREDICCIÓN CON DTW SUBSEQUENCE
######################################
# Recibe: series temporales a comparar y corregir
# Devuelve: serie de predicciones corregida según el método

def corregir_con_dtw_subsequence(y_test, y_pred):

    # INTERPOLACIÓN LINEAL EN LOS GAPS DE MISSING VALUES
    y_pred = y_pred.interpolate(method='linear').ffill().bfill()
    y_test = y_test.interpolate(method='linear').ffill().bfill()

    resultado = pd.Series(np.nan, index=y_test.index, dtype=object)

    x = y_test.to_numpy()
    y = y_pred.to_numpy()

    # Cálculo el warp con la función anterior
    _, path = dtw_subsequence_y(x, y)

    mapping = defaultdict(list)

    for i_x, j_y in path:
        mapping[i_x].append(y[j_y])

    idx_originales = y_test.index.to_numpy()

    # reindexo en función de las parejas del warp
    for i_x, valores in mapping.items():

        idx_df = idx_originales[i_x]

        # Guardar toda la lista
        resultado.loc[idx_df] = valores

    return resultado

#################################################################################
######################################################################################
###############################################################################

# CÓDIGO PARA OBTENER RESULTADOS (dataframe con y_corr incorporado)

min_muestras = 50

for modelo in lista_modelo:
    for hp in lista_horizonte:
        
        filepath = '/CARPETA/result_parquet_{}/df_test_results_vectors_RMSE_{}_H{}.parquet'.format(dataset,modelo, hp)

        # Leer datos
        df = pd.read_parquet(filepath)

        # Intervalo de muestras a partir del que se considera que se pasa a un nuevo bloque de serie temporal
        salto = 2*8 + hp   # 2 * ventana usada para calcular predicciones + hp

        #################################################################
        # PREPROCESAMIENTO
        #################################################################
        # Obtener una columna fecha con el datetime del suceso 
        # (a partir del datetime del momento en el que se produjo la predicción)
        ###########################################################################

        df['x_date_7'] = pd.to_datetime(df['x_date_7'], format='%Y-%m-%d')
        df['x_time_7'] = pd.to_datetime(df['x_time_7'], format='%H:%M:%S').dt.time
        df['datetime'] = pd.to_datetime(df['x_date_7'].astype(str) + ' ' + df['x_time_7'].astype(str))
        df['datetime'] = df['datetime'] + pd.Timedelta(minutes=hp * 15)
        df = df.drop(columns=['x_date_7', 'x_time_7'])

        # Rellenar con NaNs cada 15 min si no see tienen valores
        # Se pretende tener una fila de datos cada 15 min

        df = df.sort_values(['patient_id', 'datetime'])

        out = []
        for pid, g in df.groupby('patient_id'):  # Se rellena dentro de cada paciente de forma independiente

            idx = pd.date_range(
                start=g['datetime'].min(),
                end=g['datetime'].max(),
                freq='15min')    

            g = g.set_index('datetime').reindex(idx)

            g['patient_id'] = pid
            g.index.name = 'datetime'

            out.append(g.reset_index())

        df_resultado = pd.concat(out, ignore_index=True) 
        df_resultado["valid_block"] = np.nan

        ##################################################
        # Identificar y crear los bloques de muestras
        ##################################################
        #   Una racha de NaNs de longitud >= salto:
        #   - no pertenece a ningún bloque
        #   - separa dos bloques consecutivos
        
        is_nan = df_resultado['y_test'].isna()

        # Identificar rachas de NaNs consecutivos
        run_id = (is_nan.ne(is_nan.shift())
            | df_resultado['patient_id'].ne(df_resultado['patient_id'].shift())
        ).cumsum()

        # Longitud de cada racha
        run_length = run_id.groupby(run_id).transform('size')

        # Rachas de huecos largas
        gap_mask = is_nan & (run_length >= salto)

        # Filas que sí pertenecen a bloques (las demás)
        valid = ~gap_mask

        # Inicio de bloque:
        # - cambio de paciente
        # - fila posterior a un hueco largo
        new_block = (df_resultado['patient_id'].ne(df_resultado['patient_id'].shift())
            | gap_mask.shift(fill_value=False))

        # Inicializar la columna de bloques
        df_resultado['block_id'] = pd.NA

        # Determinar bloques
        df_resultado.loc[valid, 'block_id'] = (
            new_block[valid]
            .groupby(df_resultado.loc[valid, 'patient_id'])
            .cumsum())

        df_resultado['block_id'] = df_resultado['block_id'].astype('Int64')

        # Orden final
        df_resultado = (df_resultado.sort_values(['patient_id', 'datetime'])
            .reset_index(drop=True))

        #############################################################
        # CALCULAR EL LAG Y CORREGIRLO
        #########################################################
    
        # Inicialización de las columnas de salida en el DataFrame
        df_resultado["y_corr"] = None
        df_resultado["y_corr"] = df_resultado["y_corr"].astype(object)
        df_resultado["y_corr"] = [[v] if pd.notna(v) else [] for v in df_resultado["y_predict"]] # Vamos a guardar lista de posibles correcciones
        df_resultado["lag_valid"] = 0


        # Realizamos el análisis en cada bloque
        grouped = df_resultado.groupby(["patient_id", "block_id"])
        total = grouped.ngroups
        
        for i, ((patient_id, block_id), df_bloque) in enumerate(grouped, start=1):

            # Marcador de progreso
            if i % 200 == 0 or i == grouped.ngroups:
                print(
                    f"Procesando paciente {patient_id} "
                    f"({100 * i / total:.2f}%)")

            y_test = df_bloque["y_test"]
            y_pred = df_bloque["y_predict"]

            muestras_disponibles = y_test.notna().sum()
            is_valid = muestras_disponibles >= min_muestras

            # SI NO ES VÁLIDO: Ya no hacemos nada, porque 'y_corr' ya tiene la predicción original
            if not is_valid:
                continue

            # SI ES VÁLIDO: Ejecutamos la corrección del bloque por DTW
            y_corr = corregir_con_dtw_subsequence(y_test, y_pred)

            # Asignamos los resultados usando .values para evitar errores de Pandas
            df_resultado.loc[df_bloque.index, "y_corr"] = y_corr.values
            df_resultado.loc[df_bloque.index, "lag_valid"] = 1


        
        # Escribir en un parquet el dataframe resultados     
        output_file_name = '/CARPETA/resultadosdtwglobal/df_time_lags_results_{}_{}_H{}.parquet'.format(
        dataset, modelo, hp)
        df_resultado.to_parquet(output_file_name, index=False)
