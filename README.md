# Do the reported demand drivers survive a different training history?

**In progress — the 2011 validation search and 52 model freezes are complete; the 2012 test has not been scored.** This study will test whether demand explanations remain stable across whole-week training resamples, and whether grouping two correlated temperature measurements changes the interpretation.

The decision is whether an operations analyst should present a single ranked weather driver as robust evidence. Four model families, complete forward-validation grids and independent squared-error checks will expose the accuracy and explanation-stability tradeoff. A negative or mixed finding is useful.

The [pre-fit protocol](PROTOCOL.md) defines the full experiment. [Source reconciliation](results/INDEPENDENT_SOURCE_CHECK.json) verifies every value in the original UCI hourly file, including the total-count identity. The source has 17,379 rows, rather than the 17,389 stated on its landing page. The 165 missing hours remain absent, not invented zero demand.

This is retrospective explanation with contemporaneous observed weather, not a next-day forecast or a causal weather-intervention estimate. Month, hour and working-day are conditioning strata; their own within-stratum permutations are unchanged by design. The weather analysis will retain this limitation and every zero or negative result.

Source: Hadi Fanaee-T (2013), [Bike Sharing, UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset), DOI [10.24432/C5W894](https://doi.org/10.24432/C5W894), CC BY 4.0. This independent study is not endorsed by the data creators. Code and eligible aggregate data may be public; manuscript drafts and source papers are excluded.
