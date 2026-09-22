// ============================================================================
// TALLER GUIADO: REDES NEURONALES EN SCILAB CON EL DATASET CCPP (V3 - ITERATIVO)
// Curso: Matemática & Simulación con Software Científico
// ============================================================================

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

// Separar características (columnas 1 a 4) y objetivo (columna 5)
X_raw = raw_data(:, 1:4);
T_raw = raw_data(:, 5);

printf("Registros cargados correctamente: %d\n", size(X_raw, 1));

// Transposición matricial
X_transposed = X_raw'; // Dimensión: [4 x N]
T_transposed = T_raw'; // Dimensión: [1 x N]

// ----------------------------------------------------------------------------
// FASE 2: NORMALIZACIÓN Y DIVISIÓN DEL DATASET
// ----------------------------------------------------------------------------
disp("=== FASE 2: Normalización Min-Max y División Training/Testing ===");

// 1. Normalización Min-Max sobre la matriz completa [0, 1]
y_min = min(T_transposed);
y_max = max(T_transposed);

n_filas = size(X_transposed, 1);
n_cols = size(X_transposed, 2);

X_norm = zeros(n_filas, n_cols);

for i = 1:n_filas
    mi = min(X_transposed(i, :));
    ma = max(X_transposed(i, :));
    X_norm(i, :) = (X_transposed(i, :) - mi) / (ma - mi);
end

T_norm = (T_transposed - y_min) / (y_max - y_min);

// 2. División 80% Entrenamiento / 20% Prueba con índices calculados
N_train = floor(0.8 * n_cols);

X_train = X_norm(1:4, 1:N_train);
Y_train = T_norm(1, 1:N_train);

X_test = X_norm(1:4, (N_train + 1):n_cols);
Y_test = T_norm(1, (N_train + 1):n_cols);

// ----------------------------------------------------------------------------
// FASE 3: ARQUITECTURA E INICIALIZACIÓN DE LA RED
// ----------------------------------------------------------------------------
disp("=== FASE 3: Configuración de la Red Neuronal ===");

N_capas = [4, 10, 1];
f_act = ['sigm', 'sigm'];

W = ann_FFN_init(N_capas);

eta = 0.05;      // Tasa de aprendizaje
alpha = 0.1;     // Término de momento
max_epochs = 100;
tol_mse = 15.0;  // Tolerancia MSE en MW^2

historial_MSE = zeros(1, max_epochs);

// ----------------------------------------------------------------------------
// FASE 4: BUCLE DE ENTRENAMIENTO ITERATIVO Y EVALUACIÓN
// ----------------------------------------------------------------------------
disp("=== FASE 4: Entrenando la Red Época por Época ===");

tic();
convergencia_alcanzada = %f;
epoca_parada = max_epochs;

for epoch = 1:max_epochs
    // Entrenamiento por 1 época
    W = ann_FFN_mmult_train(X_train, Y_train, W, N_capas, 'online', 1, eta, alpha);

    // Propagación hacia adelante
    t_pred_norm = ann_FFN_run(X_test, W, N_capas);

    // Des-escalamiento a MW reales
    Y_pred_real = t_pred_norm * (y_max - y_min) + y_min;
    Y_test_real = Y_test * (y_max - y_min) + y_min;

    // Métricas de error
    errores = Y_test_real - Y_pred_real;
    mse_val = mean(errores.^2);
    
    historial_MSE(epoch) = mse_val;

    if mse_val <= tol_mse then
        convergencia_alcanzada = %t;
        epoca_parada = epoch;
        mprintf("Convergencia alcanzada en la época %d con MSE = %.6f\n", epoch, mse_val);
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