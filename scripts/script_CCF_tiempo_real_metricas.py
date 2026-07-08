########################################################################

### SCRIPT PARA EL CÁLCULO DE MÉTRICAS DESPUÉS DE LA CORRECCIÓN POR CCF TIEMPO REAL

###########################################################################
## Recibe el fichero con las series originales y la de correcciones y devuelve 
## ficheros con las métricas de evaluación calculadas
##############################################################################

# LIBRERÍAS NECESARIAS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow import keras
from pandas import DataFrame
import os
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator

#####################################################################################

# ELECCIÓN DE LOS FICHEROS A ANALIZAR

lista_modelo = ['LSTM'] #['CNN','Linear','LSTM']
lista_horizonte = [4] # [2,4,6,8] # [2,4,6,8,12,16]
dataset = 'DiaTrend'

min_muestras = 50
freq = "15min"

#################################################################################
######################################################################################

## CÓDIGO PARA EL CÁLCULO DE MÉTRICAS 
## (código de Ciro Rodríguez-León)


mpl.rcParams['svg.fonttype'] = 'none'

def clarke_error_grid_as_heatmap(ref_values,
                                 pred_values,
                                 title_string,
                                 plot_dir='ceg_as_heatmap',
                                 vmin_manual=None,
                                 vmax_manual=None,
                                 show_plot=False,
                                 minimum_sensor_reading=40,
                                 maximum_sensor_reading=500):

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    # Saturate values
    ref_values = ref_values.clip(0, 500)
    pred_values = pred_values.clip(0, 500)

    # Create numpy arrays with float32 type to reduce memory usage
    ref_values = np.asarray(ref_values, dtype=np.float32)
    pred_values = np.asarray(pred_values, dtype=np.float32)

    # Checking to see if the lengths of the reference and prediction arrays are the same
    assert (len(ref_values) == len(
        pred_values)), "Unequal number of values (reference : {0}) (prediction : {1}).".format(len(ref_values),
                                                                                               len(pred_values))

    # Checks to see if the values are within the normal sensor measurement range, otherwise it gives a warning
    # Reference values
    # if min(ref_values) < minimum_sensor_reading:
        # print(
        #     f'Input Warning: the minimum reference value ({min(ref_values):.2f}) is below the limit of sensor design (Inferior limit: {minimum_sensor_reading} mg/dL).')
        # number_ref_values_low_limit = sum(value < minimum_sensor_reading for value in ref_values)
        # print(f"Number of reference values below the limit: {number_ref_values_low_limit}")
        # print()

    # if max(ref_values) > maximum_sensor_reading:
    #     # print(
    #     #     f'Input Warning: the maximum reference value ({max(ref_values):.2f}) is above the limit of sensor design (Superior limit: {maximum_sensor_reading} mg/dL).')
    #     number_ref_values_up_limit = sum(value > maximum_sensor_reading for value in ref_values)
    #     # print(f"Number of reference values above the limit: {number_ref_values_up_limit}")
    #     # print()

    # Predicted values
    min_ref_value = min(pred_values)
    min_ref_value = round(min_ref_value, 2)
    number_pred_values_low_limit = 0
    # if min_ref_value < minimum_sensor_reading:
    #     # print(
    #     #     f'Input Warning: the minimum predicted value ({min_ref_value:.2f}) is below the minimum input value (Inferior limit: {minimum_sensor_reading} mg/dL).')
    #     number_pred_values_low_limit = sum(value < minimum_sensor_reading for value in pred_values)
    #     # print(f"Number of predicted values below the limit: {number_pred_values_low_limit}")
    #     # print()

    max_ref_value = max(pred_values)
    max_ref_value = round(max_ref_value, 2)
    number_pred_values_up_limit = 0
    # if max_ref_value > maximum_sensor_reading:
    #     # print(
    #     #    f'Input Warning: the maximum predicted value ({max_ref_value:.2f}) is above the maximum input value (Superior limit: {maximum_sensor_reading} mg/dL).')
    #     number_pred_values_up_limit = sum(value > maximum_sensor_reading for value in pred_values)
    #     # list_of_values_up_limit = [value for value in pred_values if value > maximum_sensor_reading]
    #     #print(f"Number of predicted values above the limit: {number_pred_values_up_limit}")
    #     # print()

    # list of out of range values
    out_range_values = [min_ref_value, number_pred_values_low_limit, max_ref_value, number_pred_values_up_limit]

    if show_plot:
        # ---- PLOT CONFIGURATION START ----
        fig, ax = plt.subplots()

        # --- BINS CONFIGURATION ---
        n_bins = 500
        my_range = [[0, 500], [0, 500]]
        my_bins = [n_bins, n_bins]

        # --- LIMITS CALCULATION (AUTOMATIC OR MANUAL) ---
        # If manual limits (vmin_manual) are provided, we use them. Otherwise, we calculate them.
        if vmin_manual is not None and vmax_manual is not None:
            vmin_global = vmin_manual
            vmax_global = vmax_manual
        else:
            # Standard local calculation if not sharing scale
            h_temp = np.histogram2d(ref_values, pred_values, bins=my_bins, range=my_range)
            # The minimum count is always 1 (for log), the maximum is the max of the histogram
            vmin_global = 1
            vmax_global = np.max(h_temp[0]) if np.max(h_temp[0]) > 0 else 10

        # Shared normalization across all histograms ensures that the color intensity is comparable between them.
        global_norm = LogNorm(vmin=vmin_global, vmax=vmax_global)

        # --- COLORS DEFINITION ---
        cmap_tbr2 = LinearSegmentedColormap.from_list("custom_indian", ["#ffffff", "indianred", "#8B3A3A"])
        cmap_tbr1 = LinearSegmentedColormap.from_list("custom_red", ["#ffffff", "red", "#8B0000"])
        cmap_tir = LinearSegmentedColormap.from_list("custom_lime", ["#ffffff", "limegreen", "#006400"])
        cmap_tar1 = LinearSegmentedColormap.from_list("custom_yellow", ["#ffffff", "yellow", "#808000"])
        cmap_tar2 = LinearSegmentedColormap.from_list("custom_orange", ["#ffffff", "orange", "#8B4513"])
        cmaps_ordered = [cmap_tbr2, cmap_tbr1, cmap_tir, cmap_tar1, cmap_tar2]

        # --- Draw the histograms for each range with the corresponding colormap ---
        ranges = [
            (ref_values < 54, cmap_tbr2),
            ((ref_values >= 54) & (ref_values < 70), cmap_tbr1),
            ((ref_values >= 70) & (ref_values <= 180), cmap_tir),
            ((ref_values > 180) & (ref_values <= 250), cmap_tar1),
            (ref_values > 250, cmap_tar2)
        ]

        for mask, cmap in ranges:
            if np.any(mask):
                # Important: we pass norm=global_norm to respect the shared scale across all histograms
                ax.hist2d(ref_values[mask], pred_values[mask],
                          bins=my_bins, range=my_range, cmap=cmap, cmin=1, norm=global_norm, zorder=2,)

        # --- Legend bar configuration (striped) ---
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)

        n_steps = 200
        Y_bounds = np.logspace(np.log10(vmin_global), np.log10(vmax_global), n_steps + 1)
        Z_values = np.logspace(np.log10(vmin_global), np.log10(vmax_global), n_steps)
        Z_slice = Z_values.reshape(-1, 1)

        for i in range(5):
            X_bounds = np.array([i, i + 1])
            X_mesh, Y_mesh = np.meshgrid(X_bounds, Y_bounds)
            cax.pcolormesh(X_mesh, Y_mesh, Z_slice, cmap=cmaps_ordered[i], norm=global_norm, shading='flat')

        cax.set_yscale('log')
        cax.set_xlim(0, 5)
        cax.yaxis.tick_right()
        cax.set_xticks([])  # No labels on the x-axis of the legend

        # --- Clarke Error Grid Lines ---
        lp = {'c': 'black', 'linewidth': 0.8, 'zorder': 10, 'alpha': 0.5}
        ax.plot([0, 500], [0, 500], ':', **lp)

        ax.plot([0, 175 / 3], [70, 70], '-', **lp)
        ax.plot([175 / 3, 500 / 1.2], [70, 500], '-', **lp)

        ax.plot([70, 70], [84, 500], '-', **lp)

        ax.plot([0, 70], [180, 180], '-', **lp)
        ax.plot([70, 390], [180, 500], '-', **lp)
        ax.plot([70, 70], [0, 56], '-', **lp)

        ax.plot([70, 500], [56, 400], '-', **lp)

        ax.plot([180, 180], [0, 70], '-', **lp)
        ax.plot([180, 500], [70, 70], '-', **lp)
        ax.plot([240, 240], [70, 180], '-', **lp)
        ax.plot([240, 500], [180, 180], '-', **lp)
        ax.plot([130, 180], [0, 70], '-', **lp)

        zones_ceg = [(380, 440, "A"), (440, 380, "A"), (170, 100, "B"), (100, 170, "B"),
                     (150, 400, "C"), (155, 15, "C"), (30, 120, "D"), (370, 120, "D"), (30, 340, "E"), (370, 15, "E")]
        for x, y, txt in zones_ceg:
            ax.text(x, y, txt, fontsize=12, fontweight='bold', color='black', alpha=0.6, zorder=11)

        ax.set_xlabel("Reference Concentration (mg/dl)")
        ax.set_ylabel("Prediction Concentration (mg/dl)")
        ax.set_xticks(np.arange(0, 551, 50));
        ax.set_yticks(np.arange(0, 551, 50))
        ax.xaxis.set_minor_locator(MultipleLocator(10));
        ax.yaxis.set_minor_locator(MultipleLocator(10))
        ax.set_xlim([-1, 501]);
        ax.set_ylim([-1, 500])
        ax.set_aspect('equal')

        base = f'{plot_dir}/CEG_{title_string}'
        plt.savefig(base + '.png', dpi=900, bbox_inches='tight')
        plt.savefig(base + '.pdf', dpi=900, bbox_inches='tight')
        # plt.savefig(base + '.svg', dpi=900, bbox_inches='tight')
        plt.savefig(base + '.svgz') # This will save the SVG in a compressed format, which is more efficient for large files.

        plt.close('all')
        print(f"Generado: {base}")
        # --- PLOT CONFIGURATION END ---

    # Statistics from the data
    zone = [0] * 5
    for i in range(len(ref_values)):
        # ---------------------------------------------------------------------------------
        # ZONE A (Clinical Accurate)
        # ---------------------------------------------------------------------------------
        if ((ref_values[i] < 70 and pred_values[i] < 70)  # Condition 1 for Zone A (LEFT BOTTOM)
                or
                (pred_values[i] <= 1.2 * ref_values[i] and pred_values[i] >= 0.8 * ref_values[
                    i])):  # Condition 2 for Zone A (RIGHT DIAGONAL)
            zone[0] += 1

        # EXPLANATION:
        # (ref & pred < 70) - EXCLUSIVE: True to the clinical definition in original Clarke paper text.
        # (<= 1.2*ref and >= 0.8*ref) - INCLUSIVE: Captures the ±20% range in original Clarke paper text.

        # End of Zone A conditions --------------------

        # ---------------------------------------------------------------------------------
        # ZONE E (Erroneous Treatment)
        # ---------------------------------------------------------------------------------
        elif ((ref_values[i] > 180 and pred_values[i] < 70)  # Condition 1 for Zone E (BOTTOM RIGHT)
              or
              (ref_values[i] < 70 and pred_values[i] > 180)):  # Condition 2 for Zone E (TOP LEFT)

            zone[4] += 1  # Zone E

        # EXPLANATION:
        # (< 70 and > 180) - EXCLUSIVE: True to the original Clarke paper text.
        # A real value of 70 or 180 is considered "in range," not "hypo" or "hyper."
        # Therefore, (180, 69) or (70, 181) are not Zone E.

        # End of Zone E conditions --------------------

        # ---------------------------------------------------------------------------------
        # ZONE C (Overcorrection)
        # ---------------------------------------------------------------------------------
        elif (((ref_values[i] >= 70 and ref_values[i] <= 290) and pred_values[i] >= ref_values[
            i] + 110)  # Condition 1 for Zone C (TOP)
              or
              ((ref_values[i] >= 130 and ref_values[i] <= 180) and (
                      pred_values[i] <= (7 / 5) * ref_values[i] - 182))):  # Condition 2 for Zone C (BOTTOM)

            zone[2] += 1  # Zone C

        # EXPLANATION:
        # Community consensus to use inclusive limits for Zone C conditions

        # End of Zone C conditions --------------------

        # ---------------------------------------------------------------------------------
        # ZONE D (Failure to Detect)
        # #---------------------------------------------------------------------------------
        elif ((ref_values[i] > 240 and (
                pred_values[i] >= 70 and pred_values[i] <= 180))  # Condition 1 for Zone D (RIGHT) # DUDA
              or
              (ref_values[i] <= 175 / 3 and pred_values[i] <= 180 and pred_values[
                  i] >= 70)  # Condition 2 for Zone D (LEFT)
              or
              ((ref_values[i] >= 175 / 3 and ref_values[i] < 70) and pred_values[i] >= (6 / 5) * ref_values[
                  i])):  # Condition 3 for Zone D (LEFT)

            zone[3] += 1

            # EXPLANATION OF THE CHANGES:
            # The original code used (ref <= 70) in condition D3.
            # This caused a point like (70, 90) to be classified as Zone D.
            # But (70, 90) is clinically a Zone B error,
            # since 70 is the edge of the target range, not hypoglycemia.
            # By changing it to (ref < 70), (70, 90) fails this condition and
            # correctly falls through to the 'else' (Zone B).
            #
            # Likewise, condition D1 was changed from (ref >= 240) to (ref > 240).
            # This change maintains consistency with the original Clarke paper text,
            # which textually defines this specific clinical failure as (ref > 240).
            #
            # The other prediction value limits (>= 70, <= 180) in D1 and D2 are correct because
            # they define the "target range" that the meter erroneously predicts.

            # End of Zone D conditions --------------------

        else:
            zone[1] += 1  # Zone B

    return zone, out_range_values


def test_by_range(df_result_vector: pd.DataFrame,
                  grid_name: str,
                  show_plot: bool = False,
                  plot_dir: str = 'plots',
                  minimum_sensor_reading=40,
                  maximum_sensor_reading=500) -> tuple[DataFrame, DataFrame, DataFrame]:

    df_result_vector_TBR_2 = df_result_vector[df_result_vector['y_test'] < 54]

    df_result_vector_TBR_1 = df_result_vector[
        (df_result_vector['y_test'] >= 54) & (df_result_vector['y_test'] < 70)]

    df_result_vector_TIR = df_result_vector[(df_result_vector['y_test'] >= 70) & (df_result_vector['y_test'] < 181)]

    df_result_vector_TAR_1 = df_result_vector[
        (df_result_vector['y_test'] >= 181) & (df_result_vector['y_test'] < 251)]

    df_result_vector_TAR_2 = df_result_vector[(df_result_vector['y_test'] >= 251)]

    list_dataframe_by_range = [df_result_vector,
                               df_result_vector_TBR_2, df_result_vector_TBR_1,
                               df_result_vector_TIR,
                               df_result_vector_TAR_1, df_result_vector_TAR_2]
    list_dataframe_by_range_name = ['ENTIRE', 'TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

    df_metrics_summary = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E', 'A + B', 'RMSE', 'MSE', 'MAE', 'MAPE'])
    df_zones_values = pd.DataFrame(columns=['Range', 'A', 'B', 'C', 'D', 'E'])
    df_limits = pd.DataFrame(
        columns=['Range', 'Minimum value', 'Number values below limit', 'Maximum value', 'Number values above limit'])

    for i in range(6):
        reference_values = list_dataframe_by_range[i]['y_test']
        pred_values = list_dataframe_by_range[i]['y_predict']
        # print(f'------- Zone: {list_dataframe_by_range_name[i]} -------')
        if list_dataframe_by_range_name[i] == 'ENTIRE':  # Plot the graph for the ENTIRE zone
            zone, out_of_range_values = clarke_error_grid_as_heatmap(ref_values=reference_values.values,
                                                                     pred_values=pred_values.values,
                                                                     title_string=grid_name,
                                                                     plot_dir=plot_dir,
                                                                     show_plot=show_plot,
                                                                     minimum_sensor_reading=minimum_sensor_reading,
                                                                     maximum_sensor_reading=maximum_sensor_reading)
        else:  # Do not plot the graph for the other zones
            zone, out_of_range_values = clarke_error_grid_as_heatmap(ref_values=reference_values.values,
                                                                     pred_values=pred_values.values,
                                                                     title_string=grid_name + list_dataframe_by_range_name[i],
                                                                     plot_dir=plot_dir,
                                                                     show_plot=False,
                                                                     minimum_sensor_reading=minimum_sensor_reading,
                                                                     maximum_sensor_reading=maximum_sensor_reading)

        # Clinical metrics
        # print(f'Zones: [A, B, C, D, E]')
        # print(f'Number of points by zone: {zone}')
        zone_percentages = round(pd.Series(zone) / reference_values.shape[0] * 100, 2)

        new_row_zone_values = {'Range': list_dataframe_by_range_name[i],
                               'A': zone[0],
                               'B': zone[1],
                               'C': zone[2],
                               'D': zone[3],
                               'E': zone[4]}
        new_index_zone_values = len(df_zones_values)
        df_zones_values.loc[new_index_zone_values] = new_row_zone_values

        # print(f'Percentage of point by zone: {zone_percentages.tolist()}')
        # print()

        # Non-clinical metrics
        mse = mean_squared_error(reference_values, pred_values)
        rmse = np.sqrt(mean_squared_error(reference_values, pred_values))
        mae = mean_absolute_error(reference_values, pred_values)
        mape = mean_absolute_percentage_error(reference_values, pred_values)

        #print(f'Mean Squared Error (MSE): {mse:.2f}')
        #print(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
        #print(f'Mean absolute Error (MAE): {mae:.2f}')
        #print(f'Mean absolute percentage Error (MAPE): {mape:.2f}')

        new_row_metrics = {'Range': list_dataframe_by_range_name[i],
                           'A': zone_percentages[0],
                           'B': zone_percentages[1],
                           'C': zone_percentages[2],
                           'D': zone_percentages[3],
                           'E': zone_percentages[4],
                           'A + B': zone_percentages[0] + zone_percentages[1],
                           'RMSE': rmse,
                           'MSE': mse,
                           'MAE': mae,
                           'MAPE': mape}
        new_index_metrics = len(df_metrics_summary)
        df_metrics_summary.loc[new_index_metrics] = new_row_metrics

        # Save the limits of the values
        new_row_limits = {'Range': list_dataframe_by_range_name[i],
                          'Minimum value': out_of_range_values[0],
                          'Number values below limit': out_of_range_values[1],
                          'Maximum value': out_of_range_values[2],
                          'Number values above limit': out_of_range_values[3]}
        new_index_limits = len(df_limits)
        df_limits.loc[new_index_limits] = new_row_limits

        # print()
        # print('-' * 100)
        # print()

    return df_metrics_summary, df_zones_values, df_limits

def get_train_plots_loss(hist: keras.callbacks,
                         name: str):
    fig, ax = plt.subplots()

    # data
    x_epoch = hist.epoch
    y_val_loss = hist.history['val_loss']
    y_train_loss = hist.history['loss']

    # Create a line plots
    ax.plot(x_epoch, y_val_loss, label='Validation loss', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_loss, label='Train loss', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and validation loss')

    ax.legend()

    fig.savefig(name + '_Loss.pdf', dpi=350, bbox_inches='tight')

def get_train_plots_RMSE(hist: keras.callbacks,
                         name: str):
    fig, ax = plt.subplots()

    # Sample data
    x_epoch = hist.epoch
    y_val_rmse = hist.history['val_root_mean_squared_error']
    y_train_rmse = hist.history['root_mean_squared_error']

    # Create a line plot
    ax.plot(x_epoch, y_val_rmse, label='Validation RMSE', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_rmse, label='Train RMSE', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Training and validation RMSE')

    # Add a legend
    ax.legend()

    # Display the plot
    fig.savefig(name + '_RMSE.pdf', dpi=350, bbox_inches='tight')

def get_train_plots_MSE(hist: keras.callbacks,
                        name: str):
    fig, ax = plt.subplots()

    # Sample data
    x_epoch = hist.epoch
    y_val_mse = hist.history['val_mean_squared_error']
    y_train_mse = hist.history['mean_squared_error']

    # Create a line plot
    ax.plot(x_epoch, y_val_mse, label='Validation MSE', color='orange', linestyle='-')
    ax.plot(x_epoch, y_train_mse, label='Train MSE', color='blue', linestyle='-')

    # Add labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE')
    ax.set_title('Training and validation MSE')

    # Add a legend
    ax.legend()

    # Display the plot
    fig.savefig(name + '_MSE.pdf', dpi=350, bbox_inches='tight')

def fold_results_aggregation(df:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function takes a DataFrame with results from multiple folds and aggregates the results by calculating mean and
    standard deviation by 'Range'.
    :param df: dataframe with results from multiple folds
    :return: two DataFrames: one with flat structure (suitable for CSV) and another formatted for reports
    """
    logical_order = ['ENTIRE', 'TBR_2', 'TBR_1', 'TIR', 'TAR_1', 'TAR_2']

    # Select only numeric columns for calculation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # ---------------------------------------------------------
    # APPROACH A: Flat CSV Structure (Numeric)
    # Best for saving to CSV and further analysis
    # ---------------------------------------------------------

    # 1. Group and calculate Mean and Std
    df_grouped = df.groupby('Range')[numeric_cols].agg(['mean', 'std'])

    # 2. FLATTEN THE COLUMNS (The key step)
    # This changes MultiIndex ('A', 'mean') to single index 'A_Mean'
    df_grouped.columns = [f"{col}_{stat.capitalize()}" for col, stat in df_grouped.columns]

    # 3. Reorder rows logically and reset index to make 'Range' a normal column
    df_flat = df_grouped.reindex(logical_order).reset_index()

    # print("\n=== OPTION A: Flat Structure (Ready for CSV) ===")

    # To save:
    # df_flat.to_csv('statistical_analysis.csv', index=False)

    # ---------------------------------------------------------
    # APPROACH B: Report Structure (Text format "Mean ± Std")
    # Best for presentations/publications
    # ---------------------------------------------------------

    df_report = pd.DataFrame(index=logical_order)
    df_report.index.name = 'Range'

    # Calculate base stats again (or reuse from above)
    grouped = df.groupby('Range')[numeric_cols]
    means = grouped.mean()
    stds = grouped.std()

    # Format columns
    for col in numeric_cols:
        df_report[col] = (
                means[col].map('{:.2f}'.format) +
                " ± " +
                stds[col].map('{:.2f}'.format)
        )

    df_report = df_report.reset_index()

    # print("\n=== OPTION B: Report Structure (Formatted) ===")

    # To save:
    # df_report.to_csv('final_report.csv', index=False)

    return df_flat, df_report


####################################################################################
######################################################################################
# CÓDIGO PARA RESULTADOS FINALES

resultados = []

for modelo in lista_modelo:
    for hp in lista_horizonte:
        
        filepath = 'CARPETA/resultadosccflocal/df_time_lags_results_{}_{}_H{}.csv'.format(dataset, modelo, hp)

        # Cargar los datos
        df = pd.read_parquet(filepath)
        
        # CRITERIO PARA ELEGIR SOLO UN y_corr de la lista de posibles correcciones
        # (criterio del valor más reciente)
        df['y_corr_ultimo'] = df['y_corr'].apply(
            lambda x: x[-1] if isinstance(x, (list, np.ndarray)) and len(x) > 0 else np.nan)

        y_test = df["y_test"]
        y_pred = df["y_predict"]
        y_corr = df["y_corr_ultimo"]
        valid_blocks = df['lag_valid']

        #######################################################
        # Cálcular métricas de evaluación sin corregir el lag
        #########################################################
        # Variable names
        algorithm_name = modelo
        grid_name = f'{algorithm_name}_H{hp}'
        show_plot = False
        plot_dir = 'prueba_clarke'
        minimum_sensor_reading = 40
        maximum_sensor_reading = 500

        mask_orig = y_test.notna() & y_pred.notna()
        df_metrics = pd.DataFrame({'y_test': y_test[mask_orig], 'y_predict': y_pred[mask_orig]})
        metrics_by_range, aux1, aux2 = test_by_range( df_result_vector=df_metrics,
                                                grid_name=grid_name,
                                                show_plot=show_plot,
                                                plot_dir=plot_dir,
                                                minimum_sensor_reading=minimum_sensor_reading,
                                                maximum_sensor_reading=maximum_sensor_reading)
        
        ######################################################################################
        # Cálcular métricas de evaluación considerando bloques corregidos y bloques sin corregir
        #######################################################################################
        mask_shift = (y_test.notna() & y_corr.notna() & ((valid_blocks ==1) | (valid_blocks==0)) )
        df_metrics_corr = pd.DataFrame({'y_test': y_test[mask_shift], 'y_predict': y_corr[mask_shift]})
        
        metrics_by_range_corr, aux1, aux2 = test_by_range(df_result_vector=df_metrics_corr,
                                                     grid_name=grid_name,
                                                     show_plot=show_plot,
                                                     plot_dir=plot_dir,
                                                     minimum_sensor_reading=minimum_sensor_reading,
                                                     maximum_sensor_reading=maximum_sensor_reading)
        
        ######################################################################################
        # Cálcular métricas de evaluación considerando solo bloques corregidos
        #######################################################################################
        mask_shift = (y_test.notna() & y_corr.notna() & (valid_blocks ==1))
        df_metrics_solo_corr = pd.DataFrame({'y_test': y_test[mask_shift], 'y_predict': y_corr[mask_shift]})
        
        metrics_by_range_solo_corr, aux1, aux2 = test_by_range( df_result_vector=df_metrics_solo_corr,
                                                    grid_name=grid_name,
                                                    show_plot=show_plot,
                                                    plot_dir=plot_dir,
                                                    minimum_sensor_reading=minimum_sensor_reading,
                                                    maximum_sensor_reading=maximum_sensor_reading)


        output_file_name_1 = '/CARPETA/resultadosccflocal/metricas/df_time_lags_results_ORIGINAL_{}_{}_H{}_minmuest_{}.csv'.format(dataset, modelo, hp,min_muestras)
        metrics_by_range.to_csv(output_file_name_1, index=False)

        output_file_name_2 = '/CARPETA/resultadosccflocal/metricas/df_time_lags_results_CORREGIDO_{}_{}_H{}_minmuest_{}.csv'.format(dataset, modelo, hp,min_muestras)
        metrics_by_range_corr.to_csv(output_file_name_2, index=False)

        output_file_name_3 = '/CARPETA/resultadosccflocal/metricas/df_time_lags_results_SOLOCORREGIDO_{}_{}_H{}_minmuest_{}.csv'.format(dataset, modelo, hp,min_muestras)
        metrics_by_range_solo_corr.to_csv(output_file_name_3, index=False)



