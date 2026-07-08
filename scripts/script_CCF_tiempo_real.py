########################################################################
## SCRIPT PARA EL CÁLCULO DE LAG EN TIEMPO REAL CON CCF     ###############
###########################################################################
## Recibe el fichero de datos iniciales y devuelve in fichero de datos que incluye 
## la corrección de y_pred en función de lo calculado por CCF
##############################################################################

# LIBRERÍAS NECESARIAS

import pandas as pd
import numpy as np


########################################################################################

# ELECCIÓN DE LOS FICHEROS A ANALIZAR

lista_modelo = ['LSTM'] #['CNN','Linear','LSTM']
lista_horizonte = [4] # [2,4,6,8] # [2,4,6,8,12,16]
dataset = 'DiaTrend'
window = 50

min_muestras = 50
freq = "15min"

##########################################################################################
######################################################################################

# FUNCIÓN QUE CALCULA LOS COEFICIENTES DE CCF 

def calcular_ccf_completa(y_real, y_pred, max_lag):

    y_real = pd.Series(y_real)
    y_pred = pd.Series(y_pred)

    lags = np.arange(-max_lag, max_lag + 1)
    coeficientes = []

    # Caso degenerado: y_real constante, no existe correlación útil
    if y_real.dropna().nunique() <= 1:
        coeficientes = np.zeros(len(lags))
        # forzar máximo en lag = 0
        idx_lag0 = np.where(lags == 0)[0][0]
        coeficientes[idx_lag0] = 1

        return lags, coeficientes

    for l in lags:
        correlacion = y_real.corr(y_pred.shift(l))
        coeficientes.append(correlacion)

    coeficientes = np.array(coeficientes)

    # Si toda la CCF es NaN: forzar lag = 0
    if np.all(np.isnan(coeficientes)):
        coeficientes = np.zeros(len(lags))
        idx_lag0 = np.where(lags == 0)[0][0]
        coeficientes[idx_lag0] = 1

    return lags, np.array(coeficientes)  # retorna los posibles lags y la correlación asociada a ellos


######################################################################################
########################################################################################
########################################################################################

# CÓDIGO PARA OBTENER RESULTADOS (dataframe con y_corr incorporado)

for modelo in lista_modelo:
    for hp in lista_horizonte:
        
        filepath = '/opt/datasets/prediction_vectors/2026-04/result_parquet_{}/df_test_results_vectors_RMSE_{}_H{}.parquet'.format(dataset,modelo, hp)

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
        df_resultado["y_corr"] = [[] for _ in range(len(df_resultado))] # Vamos a guardar lista de posibles correcciones
        df_resultado["lag_online"] = np.nan
        df_resultado["lag_valid"] = 0  

        # Realizamos el análisis en cada bloque
        grouped = df_resultado.groupby(["patient_id", "block_id"])
        total = len(grouped)

        for i, (combo_id, df_bloque) in enumerate(grouped, 1):
            patient_id = combo_id[0]

            # Marcador de progreso
            if i % 200 == 0:
                print(f"Procesando Paciente {patient_id} ({i/total*100:.2f}%) - ")

            # Extraer arrays de numpy para lectura rápida
            idx_global = df_bloque.index.to_numpy()
            y_test = df_bloque["y_test"].to_numpy()
            y_pred = df_bloque["y_predict"].to_numpy()
            n = len(df_bloque)

            # Listas de python para la escritura
            y_corr_bloque = [[] for _ in range(n)]
            lag_online_bloque = np.full(n, np.nan)
            lag_valid_bloque = np.zeros(n, dtype=int)

            # Cálculo de valores diponibles
            is_test = ~np.isnan(y_test)  

            # Corrijo punto a punto la serie de predicciones
            for t in range(n):
                # Mediciones hasta ese momento (el futuro es desconocido)
                valid_idx = np.where(is_test[:t+1])[0]

                # CASO 1: Sin VENTANA suficiente (menos de 50 muestras disponibles)
                if len(valid_idx) < window:
                    y_corr_bloque[t].append(y_pred[t])   # no corrijo la predicción
                    lag_online_bloque[t] = np.nan
                    lag_valid_bloque[t] = 0
                    continue

                # Índice que marca el comienzo de la ventana
                inicio_ventana = valid_idx[-window]

                # Control de seguridad (la ventana debe empezar antes de t-hp)
                if (t - hp + 1) <= inicio_ventana:
                    y_corr_bloque[t].append(y_pred[t])
                    lag_online_bloque[t] = np.nan
                    lag_valid_bloque[t] = 0
                    continue

                # CASO 2: VENTANA admisible 
                y_test_win = y_test[inicio_ventana : t - hp + 1] # fragmento con los 50 últimos valores disponibles
                y_pred_win = y_pred[inicio_ventana : t + 1] # predicciones correspondientes y además, las más recientes hasta t

                # CÁLCULO DE LAG CONCCF
                lags, ccf = calcular_ccf_completa(
                    y_test_win, y_pred_win,
                    max_lag=hp)
                lag_max = lags[np.nanargmax(ccf)]
                
                # Se considera que el punto tiene un lag igual al ofrecido por la CCF
                lag_online_bloque[t] = lag_max
                # Índice que realmente y_pred pronostica según el lag (shift de y_pred):
                destino = t + lag_max

                # Validar si el índice de destino es válido
                if 0 <= destino < n:
                    y_corr_bloque[destino].append(y_pred[t])   # llevo la predicción a la fecha donde le más conviene estar según el lag calculado
                    lag_valid_bloque[destino] = 1             # esa fila queda entonces corregida por al menos un valor
                else:
                    y_corr_bloque[t].append(y_pred[t])       
                    # lag_valid_bloque[t] se queda en 0 (valor que indica 'valor no corrgido')

            # ==============================================================================
            # Volcar los resultados del bloque al DataFrame de golpe utilizando .loc
            # ==============================================================================
            # Convertimos y_corr_bloque en una Serie de Pandas con el índice correcto.
            df_resultado.loc[idx_global, "y_corr"] = pd.Series(y_corr_bloque, index=idx_global)
            # Guardo el resto de resultados
            df_resultado.loc[idx_global, "lag_online"] = lag_online_bloque
            df_resultado.loc[idx_global, "lag_valid"] = lag_valid_bloque

        # Conversión final de tipo para mantener la consistencia en Pandas
        df_resultado['lag_valid'] = df_resultado['lag_valid'].astype('Int64')


        # Escribir en un parquet el dataframe resultados     
        output_file_name = '/home/diegosmc/resultadosccflocal/df_time_lags_results_{}_{}_H{}.parquet'.format(
        dataset, modelo, hp)
        df_resultado.to_parquet(output_file_name, index=False)
