Got it — you’re referring to the fact that in SARIMA, residuals at the beginning of the series (especially the first max(p, d, q, P×s, D×s, Q×s) steps) are unreliable due to initial conditions (like backcasting or conditional sum of errors). These are not valid for residual diagnostics.

⸻

✅ To exclude initial residuals, compute the effective burn-in:

Given your config:

'order': (1, 1, 1),
'seasonal_order': (1, 1, 1, 24),  # s = 24

You should exclude:

burn_in = max(p + d + q, s * (P + D + Q))  # conservative offset
        = max(1+1+1, 24*(1+1+1))
        = max(3, 72)
        = 72

So start analyzing residuals from index 72 onward in the training set.

⸻

Example:

resid_valid = model_fit.resid[72:]
plot_acf(resid_valid, lags=50)

This ensures you’re not contaminating diagnostics with unreliable early residuals due to lag structure and differencing.

Let me know if you’re splitting explicitly and want help slicing out only the post-training residuals.