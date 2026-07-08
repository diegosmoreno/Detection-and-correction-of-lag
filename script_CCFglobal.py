########################################################################
## SCRIPT PARA EL CÁLCULO DE LAG DE FORMA GLOBAL CON CCF     ###############
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
lista_horizonte = [4] # [2,4,6,8] 
dataset = 'DiaTrend'

# Otros valores
min_muestras = 50   # Mínimo de muestras del bloque para ejecutar CCF
freq = "15min"

##########################################################################################
######################################################################################

# FUNCIÓN QUE CALCULA LOS COEFICIENTES DE CCF 


def calcular_ccf_completa(y_real, y_pred, max_lag):
    
    # Recibe las series a comparar y_real, y _pred
    # Recibe el límite del conjunto {-maxlag,...,maxlag} de valores de k 
    # para los que se calcula un coeficiente 

    y_real = pd.Series(y_real)
    y_pred = pd.Series(y_pred)

    # Lags para los que se calcula un coeficiente
    lags = np.arange(-max_lag, max_lag + 1)
    coeficientes = []

    # Caso degenerado: y_real constante, no existe correlación útil
    if y_real.dropna().nunique() <= 1:
        coeficientes = np.zeros(len(lags))
        # Forzar máximo en lag = 0
        idx_lag0 = np.where(lags == 0)[0][0]
        coeficientes[idx_lag0] = 1

        return lags, coeficientes

    # En el resto de casos se calcula la correlación
    for l in lags:
        correlacion = y_real.corr(y_pred.shift(l))
        coeficientes.append(correlacion)

    coeficientes = np.array(coeficientes)

    # Si todos los coeficientes son NaN: forzar lag = 0
    if np.all(np.isnan(coeficientes)):
        coeficientes = np.zeros(len(lags))
        idx_lag0 = np.where(lags == 0)[0][0]
        coeficientes[idx_lag0] = 1

    # Devuelve los k posibles y el valor del coeficiente correspondiente
    return lags, np.array(coeficientes)


######################################################################################
######################################################################################
######################################################################################

# Código para el cálculo de y_corr

# Se recorren todos los archivos de entrada
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

        # Inicialización de la corrección
        # (si no hubiera lag, no habría modificación de la predicción)
        df_resultado["y_corr"] = df_resultado["y_predict"] 
    
        # Calculo el lag en cada bloque de cada paciente
        grouped = df_resultado.groupby(["patient_id", "block_id"])
        total = grouped.ngroups
        for i, ((patient_id, block_id), df_bloque) in enumerate(grouped, start=1):

            # Marcador de progreso
            if i % 200 == 0 or i == total:
                print(f"Procesando paciente {patient_id} "
                        f"({100 * i / total:.2f}%)")

            y_test = df_bloque["y_test"]
            y_pred = df_bloque["y_predict"]

            muestras_disponibles = y_test.notna().sum()

            # Mínimo de muestras por bloque para calcular el lag
            is_valid = muestras_disponibles >= min_muestras
            df_resultado.loc[df_bloque.index, "valid_block"] = int(is_valid)

            # Si no hay un mínimo, no se calcula lag
            if not is_valid:
                continue

            # Llamada a la función CCF
            lags, full_ccf = calcular_ccf_completa(y_test,y_pred,max_lag=hp)
            # ARGMAX para calcular el lag
            lag_max = lags[np.nanargmax(full_ccf)]

            # Muevo la predicción conforme al lag
            df_resultado.loc[df_bloque.index, "y_corr"] = (
                y_pred.shift(lag_max))

        # Relleno con NaN lo restante
        df_resultado.loc[df_resultado["block_id"].isna(), "y_corr"] = np.nan

        # Escribir en un csv el dataframe resultados     
        output_file_name = '/home/diegosmc/resultados0106/df_time_lags_results_{}_{}_H{}_minmuest_{}.csv'.format(dataset, modelo, hp,min_muestras)
        df_resultado.to_csv(output_file_name, index=False)