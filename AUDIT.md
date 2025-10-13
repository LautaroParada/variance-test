# Código auditado

Este repositorio implementa simulaciones de trayectorias de precios y la prueba de razón de varianzas (Variance Ratio Test) para contrastar la Hipótesis de Caminata Aleatoria. A continuación se documentan los hallazgos principales encontrados durante la revisión del módulo `code/variance_test.py`.

## Hallazgos

### 1. Estimador `sigma_a^2` ignora la agregación `q`
*Severidad: Alta*

La función `__vol_a` debería calcular la varianza de las diferencias de primer orden cuando \(q = 1\) y, para \(q > 1\), debería usar incrementos agregados de longitud `q`. Sin embargo, el bucle suma únicamente diferencias consecutivas `X[k] - X[k-1]`, sin utilizar `q`. La línea comentada justo debajo muestra la intención original, pero actualmente el estimador no cambia con `q`, lo que invalida la estadística de la prueba para horizontes distintos a un día.【F:code/variance_test.py†L31-L69】

**Recomendación:** Reescribir el bucle utilizando `X[k * q] - X[k * q - q]` o, alternativamente, aplicar `np.diff` con un paso `q` para capturar los retornos agregados.

### 2. Posible división por cero en el ajuste insesgado de `sigma_b^2`
*Severidad: Media*

Cuando `len(X) == q`, el parámetro `n` se redondea a 1 y el cálculo de `m = q * (n * q - q + 1) * (1 - (q / (n * q)))` produce exactamente cero, provocando una división por cero. El mismo riesgo aparece para otras combinaciones pequeñas de tamaño de muestra y `q` elevado.【F:code/variance_test.py†L88-L106】

**Recomendación:** Validar que `m > 0` antes de dividir y, en caso contrario, devolver `np.nan` o elevar una excepción que indique que la muestra es insuficiente para el `q` solicitado.

### 3. Estimación heterocedástica `v_hat` degenera cuando `q < 3`
*Severidad: Media*

El estimador `__v_hat` acumula términos para `j` en `range(1, q-1)`. Para `q = 1` o `q = 2`, el rango es vacío y la función retorna cero, lo que causa una división por cero al calcular la estadística `z2`. Incluso para `q = 3`, el numerador puede anularse si los retornos son constantes, dejando la prueba indefinida.【F:code/variance_test.py†L212-L252】

**Recomendación:** Incluir una comprobación explícita que garantice `q >= 3` y que `v_hat` sea estrictamente positivo antes de invertirlo. De lo contrario, devolver una advertencia o un valor nulo.

## Observaciones adicionales

* El módulo mezcla desviaciones estándar y varianzas dependiendo del parámetro `annualize`, lo que puede inducir errores silenciosos en usuarios que esperan siempre varianzas. Documentar claramente la unidad de retorno o separar las funciones para varianza y desviación estándar ayudaría a evitar malentendidos.【F:code/variance_test.py†L65-L176】
* El estilo de indentación irregular dificulta la lectura y puede inducir a errores al mantener el código. Adoptar una convención uniforme (PEP 8) incrementaría la mantenibilidad.【F:code/variance_test.py†L6-L269】

## Conclusión

Los problemas identificados afectan directamente la validez estadística del test de razón de varianzas, especialmente para horizontes de agregación mayores a uno y para tamaños de muestra reducidos. Se recomienda priorizar la corrección del estimador `sigma_a^2` y endurecer las validaciones de tamaño de muestra antes de usar la biblioteca en producción.
