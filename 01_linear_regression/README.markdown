# <center><b> Regresión Lineal </b></center>

---

## <b>¿Qué es?</b>

La **regresión lineal** es un modelo estadístico y de machine learning que se utiliza para **predecir valores numéricos continuos** (por ejemplo, ventas, ingresos, peso, temperatura, etc.) a partir de un conjunto de variables explicativas o características.

Lo que hace este modelo es establecer una **relación lineal** entre una variable dependiente (lo que queremos predecir) y una o varias variables independientes (los factores que influyen en la predicción).

Es especialmente útil cuando queremos responder preguntas como:

- ¿Cuál será el precio de una vivienda en función de su tamaño y ubicación?
- ¿Cuánto aumentarán las ventas si se incrementa el presupuesto de publicidad?
- ¿Cuál será el ingreso de una persona según su edad y nivel educativo?

En este caso, a diferencia de la regresión logística, el modelo **sí predice valores numéricos continuos** y no probabilidades.

---

## <b>Formulación Matemática</b>

### **Modelo Básico**

En la **regresión lineal múltiple**, el objetivo es predecir un valor numérico $Y$ a partir de varias variables explicativas $X = (x_1, x_2, ..., x_p)$.

La formulación del modelo es:

$$
Y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon
$$

Donde:

- $Y$ → variable dependiente (lo que queremos predecir).
- $\beta_0$ → intercepto o término independiente.
- $\beta_j$ → coeficiente que mide cuánto cambia $Y$ cuando $x_j$ aumenta en 1 unidad (manteniendo las demás variables constantes).
- $\epsilon$ → término de error aleatorio (lo que el modelo no puede explicar).

En forma matricial, para $N$ observaciones:

$$
\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}
$$

Donde:

- $\mathbf{Y}$ es un vector $N \times 1$ con los valores de salida.
- $\mathbf{X}$ es la matriz de diseño $N \times (p+1)$ (incluyendo una columna de unos para el intercepto).
- $\boldsymbol{\beta}$ es el vector de parámetros $(p+1) \times 1$.
- $\boldsymbol{\epsilon}$ es el vector de errores.

### **Estimación de los Parámetros**

Los coeficientes $\boldsymbol{\beta}$ se estiman mediante el **método de mínimos cuadrados ordinarios (OLS)**:

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{Y}
$$

Esto significa que buscamos los parámetros que **minimizan la suma de los errores al cuadrado** entre las predicciones y los valores reales.

### **Función de Pérdida: Error Cuadrático Medio (MSE)**

El modelo se ajusta minimizando el **Error Cuadrático Medio**:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
$$

Donde:

- $y_i$ es el valor real.
- $\hat{y}_i$ es el valor predicho por el modelo.

### **Interpretación**

- Si $\beta_j > 0$, significa que un aumento en $x_j$ tiende a **incrementar** el valor esperado de $Y$.
- Si $\beta_j < 0$, significa que un aumento en $x_j$ tiende a **disminuir** el valor esperado de $Y$.
- El tamaño absoluto de $\beta_j$ refleja la importancia relativa de esa variable (siempre considerando la escala de las variables).

En resumen, la regresión lineal múltiple busca encontrar la **mejor recta (o hiperplano en varias dimensiones)** que minimice la distancia entre los valores predichos y los observados.

---

## <b>Supuestos del Modelo</b>

Para que la **regresión lineal** funcione correctamente y produzca resultados confiables, se deben considerar los siguientes supuestos:

- **Relación lineal** entre las variables independientes y la variable dependiente.  
  > *Se asume que existe una relación lineal en los parámetros.*

- **Independencia de los errores**.  
  > *Los residuos deben ser independientes entre sí (ejemplo: no se cumple en series temporales sin ajustes).*

- **Homoscedasticidad** (varianza constante de los errores).  
  > *La varianza de los residuos debe ser aproximadamente constante en todos los niveles de los predictores.*

- **Normalidad de los errores**.  
  > *Los residuos deben seguir una distribución normal, especialmente importante para la inferencia estadística (pruebas de hipótesis e intervalos de confianza).*

- **Ausencia de multicolinealidad severa** entre las variables predictoras.  
  > *La alta correlación entre predictores puede inflar las varianzas de los coeficientes. Revisar VIF.*

- **No hay outliers influyentes o leverage points excesivos**.  
  > *Los valores atípicos extremos pueden distorsionar la recta de regresión. Se recomienda revisar Cook's Distance o leverage.*

- **Medición precisa de las variables independientes**.  
  > *Errores de medición en X generan estimaciones sesgadas.*

---

## <b>Interpretación</b>

Una vez entrenado el modelo de **regresión lineal múltiple**, es clave **interpretar correctamente sus salidas**. A continuación, se detallan los elementos principales:

### **Coeficientes ($\beta_j$)**

Cada coeficiente $\beta_j$ representa el **efecto promedio de un incremento de 1 unidad en la variable $x_j$ sobre la variable respuesta $Y$**, manteniendo constantes las demás variables.

$$
Y = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p + \varepsilon
$$

- Si $\beta_j > 0$: a mayor valor de $x_j$, **aumenta $Y$**.  
- Si $\beta_j < 0$: a mayor valor de $x_j$, **disminuye $Y$**.  
- Si $\beta_j = 0$: $x_j$ no tiene efecto en $Y$ (condicional a las demás variables).  

### **Intercepto ($\beta_0$)**

El intercepto es el valor esperado de $Y$ cuando **todas las variables $x_j = 0$**.  
> Puede o no tener interpretación práctica, dependiendo del contexto.

### **Error estándar y significancia estadística**

Cada coeficiente $\beta_j$ viene acompañado de:

- Un **error estándar (SE)**  
- Un **estadístico t**:  
  $$
  t_j = \dfrac{\beta_j}{SE(\beta_j)}
  $$
- Un **p-valor**: mide si el efecto de $x_j$ es significativamente distinto de 0.  

> Si el p-valor < 0.05, se considera que $x_j$ tiene un efecto significativo sobre $Y$.

### **Bondad de ajuste: $R^2$ y $R^2$ ajustado**

El **coeficiente de determinación $R^2$** mide qué proporción de la variabilidad de $Y$ es explicada por el modelo:

$$
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$$

- $SS_{res}$: suma de cuadrados de los residuos  
- $SS_{tot}$: suma de cuadrados total  

Valores cercanos a 1 indican mejor ajuste.  

El **$R^2$ ajustado** corrige por el número de variables, evitando inflar el ajuste cuando se añaden predictores irrelevantes:

$$
R^2_{ajustado} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
$$

### **ANOVA y F-Statistic**

Se utiliza la prueba F para contrastar si **al menos uno de los coeficientes $\beta_j$ es diferente de cero**.

- **Hipótesis nula ($H_0$):** todos los $\beta_j = 0$  
- **Hipótesis alternativa ($H_1$):** al menos un $\beta_j \neq 0$  

Un p-valor bajo en la prueba F sugiere que el modelo es globalmente significativo.

### **Residuos**

El análisis de residuos permite evaluar la validez de los supuestos del modelo:

- Deben tener **media cercana a 0**  
- Deben presentar **varianza constante (homocedasticidad)**  
- Deben seguir **distribución aproximadamente normal**  
- No deben mostrar **patrones sistemáticos** al graficarse contra los valores ajustados

### **En resumen:**

| Elemento        | Qué representa                                                                 |
|-----------------|---------------------------------------------------------------------------------|
| $\beta_j$       | Efecto de $x_j$ sobre $Y$ (manteniendo constantes los demás predictores)        |
| $\beta_0$       | Valor esperado de $Y$ cuando todos los $x_j = 0$                                |
| p-valor         | Significancia del efecto de cada variable                                       |
| $R^2$           | Proporción de variabilidad de $Y$ explicada por el modelo                       |
| $R^2$ ajustado  | Versión corregida por número de predictores                                     |
| Estadístico F   | Significancia global del modelo                                                 |
| Residuos        | Evaluación de supuestos y ajuste del modelo                                     |

---

## <b>Implementación en `scikit-learn`</b>

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression(
    fit_intercept=True,
    copy_X=True,
    n_jobs=None,
    positive=False
)

model.fit(X_train, y_train)
pred = model.predict(X_test)
```

---

## <b>Parámetros Cruciales</b>

A continuación, se explican los hiperparámetros más importantes que afectan directamente el rendimiento y la precisión de la **regresión lineal** en `scikit-learn`.

### **fit_intercept — Inclusión del intercepto**

Controla si el modelo estima el intercepto (β₀).

- `True` → Se calcula el intercepto (lo más común).
- `False` → No se calcula intercepto; la recta pasa por el origen.

> Si tus datos no están centrados en torno a 0, **desactivar el intercepto generará un modelo sesgado**.

### **Resumen gráfico mental**

| Parámetro        | Afecta...                       | Cuándo ajustarlo                                     |
|------------------|---------------------------------|------------------------------------------------------|
| `fit_intercept`  | Si se ajusta el término β₀      | Cuando los datos están centrados o no en el origen   |

---

## <b>Validaciones Numéricas Internas</b>

Cuando llamas al método `.fit()` del modelo de **regresión lineal** en `scikit-learn`, el algoritmo aplica un procedimiento matemático para encontrar los coeficientes que mejor ajustan los datos.

### **¿Qué significa "entrenar" el modelo?**

Significa **estimar los coeficientes óptimos** ($\beta_0, \beta_1, ..., \beta_p$) de la recta o hiperplano que mejor explica la relación entre las variables predictoras $X$ y la variable respuesta $y$.

### **¿Qué función se minimiza?**

La regresión lineal busca minimizar el **Error Cuadrático Medio (MSE)**, equivalente a minimizar la **suma de los residuos al cuadrado** (RSS):

$$
RSS = \sum_{i=1}^N (y_i - \hat{y}_i)^2
$$

Donde:
- $y_i$ = valor real observado  
- $\hat{y}_i$ = valor predicho por el modelo  

En forma matricial, se resuelve:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

### **¿Qué hace internamente `.fit()`?**

1. **Prepara la matriz de diseño $X$** (incluyendo el intercepto si `fit_intercept=True`).
2. **Calcula la solución cerrada** mediante álgebra matricial (mínimos cuadrados ordinarios).
   - Usa descomposiciones numéricas como **SVD (Singular Value Decomposition)** para mejorar la estabilidad y precisión.
3. **Obtiene los coeficientes** que minimizan el error cuadrático.
4. **Valida convergencia numérica** para asegurar que la solución sea estable.

### **¿Qué devuelve al final?**

Después del ajuste, puedes acceder a:

- `model.coef_` → coeficientes estimados ($\beta_1, \dots, \beta_p$)  
- `model.intercept_` → el término independiente ($\beta_0$)  
- `model.rank_` → rango de la matriz $X$  
- `model.singular_` → valores singulares de la descomposición (útil para detectar multicolinealidad)  

### **Importante:**

- Si hay **multicolinealidad severa** ($X^T X$ casi no invertible), los coeficientes pueden volverse inestables.  
- En esos casos, se recomienda usar **Ridge** o **Lasso**, que agregan regularización.  
- A diferencia de la regresión logística, la regresión lineal **no requiere un proceso iterativo de optimización**: la solución es directa y exacta en términos matemáticos.

### **En resumen**

Entrenar una regresión lineal en `scikit-learn` equivale a resolver un **problema de mínimos cuadrados ordinarios**, donde el objetivo es **encontrar la recta que minimiza la suma de errores al cuadrado** entre los valores reales y los predichos.

---

## <b>Casos de uso</b>

Aunque hoy en día existen modelos más sofisticados como Random Forest, XGBoost o redes neuronales, la **regresión lineal** sigue siendo una de las técnicas más utilizadas y valiosas en múltiples disciplinas, especialmente cuando se busca **simplicidad e interpretabilidad**.

### **Predicción de precios (inmuebles, autos, etc.)**

- Relaciona características como superficie, ubicación o antigüedad con el precio.
- Fácil de interpretar: cada coeficiente muestra cómo influye cada variable en el precio.

> Ideal para reportes inmobiliarios y modelos de valoración inicial.

### **Análisis económico y financiero**

- Usada para modelar **relaciones macroeconómicas** (ej. inflación, consumo, inversión).
- Permite explicar tendencias con base en predictores claros.

> Economistas la prefieren por su **transparencia y solidez teórica**.

### **Estudios científicos y sociales**

- Útil para identificar relaciones entre variables (ej. nivel educativo vs. ingresos).
- Entregables interpretables para publicaciones académicas.

> Cada coeficiente tiene un significado estadístico claro.

### **Forecasting básico en series temporales**

- Puede aplicarse para tendencias lineales en el tiempo.
- Sirve como **modelo base** antes de usar métodos más avanzados como ARIMA o LSTM.

> Siempre es el primer escalón antes de complejizar.

### **Modelos de riesgo y predicción cuantitativa**

- Común en seguros y finanzas para estimar costos esperados.
- Fácil de auditar y validar ante reguladores.

> Transparencia y trazabilidad > complejidad excesiva.

### **¿Cuándo *NO* usar regresión lineal?**

- Cuando la relación entre variables es **no lineal** y no puede corregirse con transformaciones.
- Si hay **alta multicolinealidad**, lo que vuelve inestables los coeficientes.
- Cuando los datos tienen **outliers extremos** que distorsionan los resultados.
- En contextos donde la **precisión predictiva máxima** es más importante que la interpretabilidad.

### **Conclusión**

> La regresión lineal no es un modelo “obsoleto”: sigue siendo **la mejor opción cuando buscas interpretabilidad, rapidez y una primera aproximación sólida a los datos**.

---

## <b>Profundización matemática</b>

Esta sección explica el **fundamento matemático detrás de la regresión lineal**, más allá de la implementación práctica, mostrando cómo se construye y entrena el modelo.

### **Ecuación del modelo**

La regresión lineal busca modelar la relación entre una variable dependiente $y$ y un conjunto de variables independientes $x_1, x_2, \dots, x_p$:

$$
\hat{y}_i = \beta_0 + \sum_{j=1}^{p} \beta_j x_{ij}
$$

donde:
- $\hat{y}_i$ = valor predicho para la observación $i$
- $\beta_0$ = intercepto
- $\beta_j$ = coeficientes de regresión
- $x_{ij}$ = valor de la variable $j$ en la observación $i$

### **Función de pérdida (mínimos cuadrados ordinarios, OLS)**

El objetivo es **minimizar el error cuadrático medio** (MSE). La función de costo es:

$$
\mathcal{L}(\beta) = \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{N} \left( y_i - \beta_0 - \sum_{j=1}^{p} \beta_j x_{ij} \right)^2
$$

> OLS encuentra los $\beta$ que **minimizan la suma de errores al cuadrado**.

### **Solución matricial cerrada**

La regresión lineal **sí tiene solución analítica exacta** (a diferencia de la logística). En forma matricial:

$$
\mathbf{y} = \mathbf{X}\beta + \epsilon
$$

donde:
- $\mathbf{y}$ es un vector $N \times 1$ de valores observados
- $\mathbf{X}$ es la matriz $N \times (p+1)$ de predictores (incluyendo columna de 1's para el intercepto)
- $\beta$ es el vector de coeficientes
- $\epsilon$ es el error

La solución óptima para $\beta$ es:

$$
\hat{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
$$

### **Derivación del gradiente**

La derivada de la función de pérdida respecto a un coeficiente $\beta_j$ es:

$$
\frac{\partial \mathcal{L}}{\partial \beta_j} = -2 \sum_{i=1}^{N} (y_i - \hat{y}_i) x_{ij}
$$

Esto significa que el ajuste busca **reducir la diferencia entre los valores predichos y los observados**, ponderada por la variable correspondiente.

### **Regularización**

Cuando hay muchos predictores o multicolinealidad, se usan variantes:

- **Ridge (L2):**

$$
\mathcal{L}_{\text{ridge}} = \mathcal{L} + \lambda \sum_{j=1}^{p} \beta_j^2
$$

- **Lasso (L1):**

$$
\mathcal{L}_{\text{lasso}} = \mathcal{L} + \lambda \sum_{j=1}^{p} |\beta_j|
$$

- **Elastic Net (combinación de L1 y L2):**

$$
\mathcal{L}_{\text{EN}} = \mathcal{L} + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2
$$

### **Propiedades matemáticas clave**

- La función de pérdida de OLS es **convexa**, lo que garantiza un mínimo global.
- Con regularización L1, algunos coeficientes pueden volverse exactamente **cero** (selección de variables).
- El error $\epsilon$ se asume con **media cero y varianza constante** ($\epsilon \sim \mathcal{N}(0, \sigma^2)$) para que los estimadores sean óptimos (BLUE: Best Linear Unbiased Estimators).

### **Conclusión**

> La regresión lineal tiene la ventaja de una **solución cerrada y exacta** vía álgebra matricial.  
> Su formulación matemática es la base para métodos más avanzados como Ridge, Lasso y Elastic Net, y también inspira técnicas de optimización en modelos no lineales como redes neuronales.

---

## <b>Recursos para profundizar</b>

**Libros**  
- *Applied Linear Regression Models* – Kutner, Nachtsheim, Neter  
- *Introduction to Linear Regression Analysis* – Montgomery, Peck, Vining  
- *The Elements of Statistical Learning* – Hastie, Tibshirani, Friedman  

**Cursos**  
- MIT OpenCourseWare – Linear Regression  
- Coursera – Regression Models (Johns Hopkins University)  
- StatQuest (YouTube) – Linear Regression  

**Documentación oficial**  
- [scikit-learn: LinearRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

---
