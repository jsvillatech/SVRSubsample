# Implementación de algoritmo de submuestreo para Support Vector Regression en conjuntos de grandes volúmenes de datos​

_Estaba pensando en poner algo del abstract o de la introducción de tu texto_

## Contenido del repositorio 📌

_El proyecto está compuesto por los notebooks en los que se realizó la aplicación del algoritmo de submuestreo, el cual se encuentra definido en un archivo _**.py**_. Detalladamente, cada uno se enfoca en un conjunto de dato específicos que se describen de la siguiente manera:_

- El conjuto de datos _Temp Electric Motor_ es un archivo csv que comprende varios datos de sensores recopilados de un motor síncrono de imanes permanentes (PMSM). Las características objetivo más interesantes son la temperatura del rotor ("pm"), las temperaturas del estator ("stator_*") y el par.
    Tomado de: <https://www.kaggle.com/wkirgsn/electric-motor-temperature>
- El conjuto de datos _UK Used Car_ contiene 100.000 listados de autos usados y cada columna representa la información de precio, transmisión, kilometraje, tipo de combustible, impuesto de circulación, millas por galón (mpg) y tamaño del motor. La variable objetivo es el precio de venta del automóvil, para predecirlo a partir de sus características.
    Tomado de: <https://www.kaggle.com/adityadesai13/used-car-dataset-ford-and-mercedes>
- El conjuto de datos _Beijing PM 2.5_ tiene como variable objetivo la partícula contaminante PM 2.5 que se encuentra en el aire. Está compuesto por los datos de PM2.5 de la Embajada de EE. UU. En Beijing por hora.
    Tomado de: <https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data>
    

### Pre-requisitos 🔧

_Inicialmente, para realizar alguna aplicación del algoritmo de submuestreo es importante confirmar que se podrán importar adecuadamente las siguientes librerias en Python: pandas, sklearn, time, math, from skopt: BayesSearchCV._
_Para importar el archivo **.py** y hacer uso del algoritmo, se debe definir los parámetros que recibe: 
```
svr_subsample_bayes_train(trainD,testD,sig,ep,y_label_name,kernel_type,num_neighbors,thres,rs=45)
```

|  Parámetros      | Descripción de parámetro |
| -----------------|:-------------:|
| **trainD**       | Train data     |
| **testD**        | Test data     |
| **sig**          | Percentage of the subsampleT (0.01 or  0.1)   |
| **ep**           | Percentage of R set (usually 0.1)    |
| **y_label_name** | Column name of the target variable     |
| **kernel_type**  | Type of kernel ('linear', 'poly', 'rbf')   |
| **num_neighbors**| Number of neighbors to look for (5 when sig=0.01 or 3 when sig=0.1)     |
| **thres**        | Threshold for the callback function on_step     |
| **rs**           | random_state seed for the reproducibility of the experiment   |

## Descripción del algoritmo de submuestreo 🛠️

La implementación del algoritmo tiene como base _Nearest neighbors methods for support vector machines, Camelo, S. A.,González-Lima, M. D.,Quiroz, A. J. Adapted for SVR._ Teniendo en cuenta los siguientes pasos, que dependiendo de los resultados obtenidos puede requerir más de una iteración:

1. En la primera iteración, selecciona una submuestra aleatoria 𝑻(𝟎) de tamaño δn del conjunto de datos de entrenamiento elegido de tamaño n, que en este caso será llamado D.
```
submuestraT = trainD.sample (frac = sig, random_state = rs)
```
2. Inicializa un nuevo conjunto 𝑺(𝟎) que sea la diferencia entre D y 𝑻(𝟎)

```
submuestra_index = lista (submuestraT.index)
SetS = trainD.drop (subsample_index, eje = 0)
```
3. Resolver el problema de RVS para 𝑻(𝟎) e identifique los vectores de soporte 𝑺𝑽(0)
> En este paso, se aplica la optomización bayesiana para la optimización de hiperparámetros. 
> Si es una nueva iteración, se realiza la intersección del anterior 𝑹(𝒋 − 𝟏) con el actual 𝑺𝑽(𝒋)
4. Definir un nuevo conjunto 𝑵(𝒋) encontrando los k vecinos más cercanos en 𝑺(j) para cada Support Vector.
5. Crear un nuevo conjunto 𝑹(𝒋) dejando caer todos los N ubicados en 𝑺(𝒋).
6. Unifica 𝑹(𝒋), 𝑵(𝒋) y 𝑺𝑽 (𝒋) en un solo conjunto llamado 𝑻(𝒋 + 𝟏).
7. Resuelva el problema de SVR para 𝑻(𝒋 + 𝟏).
> Adicionalmente, se define 𝑺 (𝒋 + 𝟏) que es el actual 𝑺(𝒋) menos (𝐒𝐕∪𝑹).
8. Si no hay una mejora significativa del error de regresión, regrese al paso 3.
> En la nueva iteración: 𝒋 ← 𝒋+𝟏 

## Bibliografía 📖

_Nearest neighbors methods for support vector machines, Camelo, S. A.,González-Lima, M. D.,Quiroz, A. J._


## Expresiones de Gratitud 🎁

* Gracias...



---
Proyecto por [jsvillatech](https://github.com/jsvillatech)
