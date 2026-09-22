// ============================================================================
// TALLER GUIADO: REDES NEURONALES EN SCILAB CON EL DATASET CCPP (V3 - ITERATIVO)
// Curso: Matemática & Simulación con Software Científico
// ============================================================================

// 1. CARGA OBLIGATORIA DE LIBRERÍA AL INICIO (EVITA ERROR DE VARIABLE UNDEFINED)
atomsLoad("ANN_Toolbox");

clear;
clc;

// ----------------------------------------------------------------------------
// FASE 1: CARGA DE DATOS Y TRANSPOSICIÓN MATRICIAL
// ----------------------------------------------------------------------------
disp("=== FASE 1: Carga y Preparación de Datos ===");

if ~fileinfo("ccpp.csv") then
    error("El archivo ccpp.csv no se encuentra en el directorio actual.");
end

// Cargamos los datos crudos
raw_data = csvRead("ccpp.csv", ",", [], "double");

// Si la primera fila contiene valores no numéricos (encabezados), se descarta
if isnan(raw_data(1, 1)) then
    raw_data = raw_data(2:$, :);
end

// Separación de Entradas (X: cols 1 a 4) y Salida (T: col 5 - PE)
X_raw = raw_data(:, 1:4);
T_raw = raw_data(:, 5);

// TRANSPOSICIÓN: ANN_Toolbox exige patrones en COLUMNAS (4xN y 1xN)
X_transposed = X_raw'; 
T_transposed = T_raw'; 

mprintf("Registros cargados correctamente: %d\n", size(X_transposed, 2));

// ----------------------------------------------------------------------------
// FASE 2: NORMALIZACIÓN MIN-MAX Y DIVISIÓN TRAINING/TESTING
// ----------------------------------------------------------------------------
disp("=== FASE 2: Normalización Min-Max y División Training/Testing ===");

// Guardamos min/max globales de la variable objetivo (PE) para des-escalamiento
y_min = min(T_transposed);
y_max = max(T_transposed);

// Normalización Min-Max por fila para las características de entrada [0, 1]
n_filas = size(X_transposed, 1);
n_cols = size(X_transposed, 2);

X_norm = zeros(n_filas, n_cols);

for i = 1:n_filas
    mi = min(X_transposed(i, :));
    ma = max(X_transposed(i, :));
    X_norm(i, :) = (X_transposed(i, :) - mi) / (ma - mi);
end

// Normalización Min-Max para la salida objetivo [0, 1]
T_norm = (T_transposed - y_min) / (y_max - y_min);

// Partición del dataset: 80% Entrenamiento, 20% Prueba
N_train = floor(0.8 * n_cols);

X_train = X_norm(1:4, 1:N_train);
Y_train = T_norm(1, 1:N_train);

X_test = X_norm(1:4, (N_train + 1):n_cols);
Y_test = T_norm(1, (N_train + 1):n_cols);

// ----------------------------------------------------------------------------
// FASE 3: CONFIGURACIÓN Y ESTRUCTURA DE LA RED NEURONAL
// ----------------------------------------------------------------------------
disp("=== FASE 3: Configuración de la Red Neuronal ===");

// Definición de arquitectura: 4 Entradas -> 10 Neuronas Ocultas -> 1 Salida
N_capas = [4, 10, 1];

// Inicialización de pesos sinápticos
W = ann_FFN_init(N_capas);

// Hiperparámetros de entrenamiento
eta = 0.05;          // Tasa de aprendizaje
alpha = 0.1;         // Término de momento
max_epochs = 100;    // Límite máximo de épocas
tol_mse = 15.0;      // Criterio de parada temprana (MSE en MW^2)

historial_MSE = zeros(1, max_epochs);

// ----------------------------------------------------------------------------
// FASE 4: BUCLE DE ENTRENAMIENTO ITERATIVO Y EVALUACIÓN DE CONVERGENCIA
// ----------------------------------------------------------------------------
disp("=== FASE 4: Entrenando la Red Época por Época ===");

tic();
convergencia_alcanzada = %f;
epoca_parada = max_epochs;

for epoch = 1:max_epochs
    // Entrenamiento online (1 época completa)
    W = ann_FFN_mmult_train(X_train, Y_train, W, N_capas, 'online', 1, eta, alpha);

    // Evaluación en el conjunto de prueba (Generalización)
    t_pred_norm = ann_FFN_run(X_test, W, N_capas);

    // Des-escalamiento a MW reales
    Y_pred_real = t_pred_norm * (y_max - y_min) + y_min;
    Y_test_real = Y_test * (y_max - y_min) + y_min;

    // Métricas de error cuadrático medio
    errores = Y_test_real - Y_pred_real;
    mse_val = mean(errores.^2);
    
    historial_MSE(epoch) = mse_val;

    // Verificación del criterio de convergencia
    if mse_val <= tol_mse then
        convergencia_alcanzada = %t;
        epoca_parada = epoch;
        mprintf("Convergencia alcanzada en la época %d con MSE = %.6f MW^2\n", epoch, mse_val);
        break;
    end
end

tiempo_ejecucion = toc();

// ----------------------------------------------------------------------------
// FASE 5: GRAFICACIÓN DE RESULTADOS
// ----------------------------------------------------------------------------
disp("=== FASE 5: Generando Gráfica de Convergencia ===");

scf(0);
clf();
plot(1:epoca_parada, historial_MSE(1:epoca_parada), 'r-o', 'LineWidth', 2);
xtitle("Convergencia del Error Cuadrático Medio (MSE) en Prueba", "Épocas", "MSE (MW^2)");
xgrid();

// ============================================================================
// PREGUNTAS DEL TALLER Y CONCLUSIONES
// ============================================================================
// 1. ¿Cómo influye la transposición matricial en el ANN_Toolbox?
//    R: El toolbox requiere que las características/variables sean filas y 
//    las muestras sean columnas (dimensión 4xN). Sin la transposición, 
//    Scilab toma cada muestra como variable generando errores de dimensión.
//
// 2. ¿Por qué es vital des-escalar antes de calcular el MSE?
//    R: Porque durante la red las salidas están en rango [0, 1]. Calcular el
//    MSE sin des-escalar daría valores artificialmente diminutos. Des-escalar
//    permite evaluar el error en unidades físicas reales (MW^2).
//
// 3. ¿Cómo se comporta la curva de convergencia?
//    R: Muestra una reducción drástica del error en las primeras épocas y 
//    luego una estabilización suave, alcanzando la convergencia deseada.
// ============================================================================