---
excerpt: Te explicamos cómo procesar múltiples activos financieros para agregarlos a un reporte que destaque el rendimiento anual de cada activo.
id_slug: C-BILELLO-1
slug: procesando-reportes-anuales-sobre-activos-financieros-en-python
tags:  
  ghost:
    - Finance
    - Descriptive Analysis
    - Pandas
    - Tutorial
title: Procesando reportes anuales sobre activos financieros en Python
---

Si comparamos el rendimiento de activos financieros como Bitcoin, ETFs y acciones, ¿cuál fue el activo con mayor rentabilidad, desde el año 2010?

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_asset_performance.png" alt="{alt_text}">
    <figcaption>{caption}</figcaption>
</figure>

En este tutorial, te explicamos cómo descargar, procesar y reportar el rendimiento de activos financieros usando Python.

El reporte está inspirado los [análisis de Charlie Bilello](https://twitter.com/charliebilello/status/1643699214914822144).

## Data

Definimos la lista de tickers de los activos a analizar.

```python
tickers = ['BTC-USD', 'QQQ', 'IWF', 'SPY', ...]
```

Descargamos movimientos diarios de precios históricos usando la librería `yfinance`, que descarga datos de Yahoo Finance.

```python
import yfinance as yf
df = yf.download(tickers)
```

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_hist.png" alt="{alt_text}">
    <figcaption>{caption}</figcaption>
</figure>

## Preguntas

1. ¿Cómo descargar datos históricos de precios de múltiples activos financieros?
2. ¿Cómo calcular el retorno acumulado anual de cada activo?
3. ¿Por qué es necesario agrupar los datos para tener cálculos acumulados?
4. ¿Cómo seleccionar el último día de retorno acumulado en cada año?
5. ¿Cómo identificar los valores máximos y mínimos de retorno en cada año?
6. ¿Cómo calcular el porcentaje de retornos positivos para cada activo?

## Metodología

### Retorno acumulado anual

Seleccionamos los precios de cierre ajustados `Adj Close` desde el año 2010 y calculamos el retorno acumulado anual, seleccionando el último día hábil de cada año.

```python
(df
 .loc['2010':, 'Adj Close']
 .groupby(df.index.year).pct_change().add(1)
 .groupby(df.index.year).cumprod().sub(1)
 .resample('YE').last().T
)
```

`BTC-USD` (Bitcoin) presenta valores perdidos en los primeros años debido a su regulación y adopción (ver historia completa en [Wikipedia](https://en.wikipedia.org/wiki/History_of_bitcoin#2014)).

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_etf_returns.png" alt="{alt_text}">
    <figcaption>{caption}</figcaption>
</figure>

{{ engagement }}

### Resumen por columnas

¿Cómo se comportaron los activos durante todos los años?

Calculamos el retorno promedio anual y el retorno acumulado anual para cada activo.

> Para más detalles sobre el cálculo de los retornos, visita [este tutorial](https://datons.ai/preprocess-and-analyze-stock-returns-with-python/).

```python
t_avg = df.mean(axis=1).mul(100)
t_cum = df.add(1).cumprod(axis=1).sub(1).mul(100).iloc[:,[-1]]

pd.DataFrame({'AVG': t_avg, 'CAGR': t_cum})
```

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_etf_summary.png" alt="{alt_text}">
    <figcaption>{caption}</figcaption>
</figure>

### Resumen por filas

¿Cuáles fueron los valores máximos y mínimos en cada año? ¿Cuál fue el porcentaje de retornos positivos?

```python
positive_pct = lambda x: (x > 0).mean() * 100
dfr.agg(['max', 'min', positive_pct])
```

Todos los activos presentan retornos positivos acumulados al final del período, siendo `BTC-USD` el activo con mayor rentabilidad.

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_etf_summary_rows.png" alt="{alt_text}">
    <figcaption>{caption}</figcaption>
</figure>

### Combinar y aplicar estilo

Finalmente, combinamos todas las tablas y aplicamos un estilo para resaltar los valores máximos y mínimos.

> En [este tutorial](https://datons.ai/style-pandas-pivot-table-to-create-heat-matrix/) se explica cómo aplicar estilos a tablas de pandas.

<figure>
    <img src="https://images.datons.ai/C-BILELLO-1/D_asset_performance.png" alt="{alt_text}">
    <figcaption>{caption}</figcaption>
</figure>

¿Por qué durante el año 2022 casi todos los activos presentan retornos negativos?

Te leo en los comentarios.

## Conclusiones

Para las secciones {QUESTIONS} y {CONCLUSIONS}, aquí tienes una propuesta que encaja con el resto del contenido:

Entendido, aquí tienes una versión más directa y enfocada:

Entendido, voy a ajustar esa parte para reflejar la importancia de `groupby` en el cálculo de los acumulados.

## Conclusiones

1. **Descargar datos históricos:** `yf.download(tickers)` para obtener datos de múltiples activos financieros de Yahoo Finance.
2. **Calcular el retorno acumulado anual:** `.pct_change().add(1).cumprod().sub(1)` para determinar el rendimiento de un activo a lo largo del tiempo.
3. **Agrupar datos para cálculos acumulados:** `groupby.cumprod` para resetear los cálculos acumulados al inicio de cada año.
4. **Seleccionar el último día de retorno acumulado:** `.resample('Y').last()` para obtener el valor final de cada año, útil para análisis anual.
5. **Identificar valores máximos y mínimos:** `.agg(['max', 'min'])` para encontrar los extremos en el rendimiento anual de los activos.
6. **Calcular el porcentaje de retornos positivos:** `lambda x: (x > 0).mean() * 100` para evaluar la frecuencia de ganancias de un activo.

{{ feedback }}