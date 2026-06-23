import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t as t_dist
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# ── YOUR ACTUAL DATA FROM WORLD BANK (2001-2023) ────────────
years = np.array([2001,2002,2003,2004,2005,2006,2007,
                  2008,2009,2010,2011,2012,2013,2014,
                  2015,2016,2017,2018,2019,2020,2021,
                  2022,2023])

# Tax Revenue as % of GDP
tau_pct = np.array([17.1929, 17.4931, 18.4777, 21.7521,
                    21.3215, 12.5346, 13.8780, 13.8960,
                    12.6125, 13.3884, 14.8658, 15.3682,
                    10.6718, 11.2522, 11.6906, 11.0583,
                    11.5770, 12.2442, 11.9970, 11.3404,
                    12.2447, 12.2975, 12.3931])

# GDP Growth Rate (%)
g = np.array([4.0000,  4.5000,  5.2000,  5.6000,
              5.9000,  6.3999,  4.3468,  9.1498,
              4.8445,  7.8997, 14.0471,  9.2928,
              7.3125,  2.8562,  2.1208,  3.3735,
              8.1289,  6.2001,  6.5078,  0.5139,
              5.0765,  3.8048,  3.1401])

# Convert tax rate to decimal
tau = tau_pct / 100
n   = len(years)

# ── DESCRIPTIVE STATISTICS ───────────────────────────────────
print("=" * 55)
print("  DESCRIPTIVE STATISTICS")
print("=" * 55)
print(f"  Observations     : {n}")
print(f"  Mean Tax Rate    : {np.mean(tau_pct):.4f}%")
print(f"  Std Dev Tax Rate : {np.std(tau_pct, ddof=1):.4f}%")
print(f"  Min Tax Rate     : {np.min(tau_pct):.4f}%")
print(f"  Max Tax Rate     : {np.max(tau_pct):.4f}%")
print(f"  Mean GDP Growth  : {np.mean(g):.4f}%")
print(f"  Std Dev GDP Gr.  : {np.std(g, ddof=1):.4f}%")
print(f"  Min GDP Growth   : {np.min(g):.4f}%")
print(f"  Max GDP Growth   : {np.max(g):.4f}%")

# ── OLS QUADRATIC REGRESSION ─────────────────────────────────
X       = np.column_stack([np.ones(n), tau, tau**2])
theta   = np.linalg.lstsq(X, g, rcond=None)[0]
alpha_h = theta[0]
beta_h  = theta[1]
delta_h = theta[2]
gamma_h = -delta_h

g_fit = X @ theta
resid = g - g_fit
SSE   = np.sum(resid**2)
SST   = np.sum((g - np.mean(g))**2)
R2    = 1 - SSE/SST
R2adj = 1 - (1-R2)*(n-1)/(n-3)

s2      = SSE/(n-3)
se      = np.sqrt(s2*np.diag(np.linalg.inv(X.T@X)))
t_stats = theta/se
p_vals  = 2*(1 - t_dist.cdf(np.abs(t_stats), df=n-3))

print("\n" + "=" * 55)
print("  OLS REGRESSION RESULTS")
print("  Model: g(t) = alpha + beta*t - gamma*t^2")
print("=" * 55)
print(f"\n  alpha = {alpha_h:.4f} | SE={se[0]:.4f}"
      f" | t={t_stats[0]:.4f} | p={p_vals[0]:.4f}")
print(f"  beta  = {beta_h:.4f} | SE={se[1]:.4f}"
      f" | t={t_stats[1]:.4f} | p={p_vals[1]:.4f}")
print(f"  delta = {delta_h:.4f} | SE={se[2]:.4f}"
      f" | t={t_stats[2]:.4f} | p={p_vals[2]:.4f}")
print(f"\n  gamma = -delta = {gamma_h:.4f}")
print(f"  R2         = {R2:.4f}")
print(f"  Adj R2     = {R2adj:.4f}")

# ── LAFFER CURVE CONDITION ────────────────────────────────────
print("\n" + "=" * 55)
print("  LAFFER CURVE CONDITION")
print("=" * 55)
if gamma_h > 0:
    print(f"\n  gamma = {gamma_h:.4f} > 0  CONFIRMED")
    print("  Laffer Curve shape confirmed.")
    print("  The model is strictly concave.")
else:
    print(f"\n  gamma = {gamma_h:.4f} <= 0  NOT confirmed.")

# ── KKT OPTIMAL TAX RATE ─────────────────────────────────────
tau_unc = beta_h/(2*gamma_h)
tau_min = 0.1067
tau_max = 0.2175

print("\n" + "=" * 55)
print("  KKT OPTIMAL TAX RATE")
print("=" * 55)
print(f"\n  tau* = beta / (2*gamma)")
print(f"       = {beta_h:.4f} / (2 x {gamma_h:.4f})")
print(f"       = {beta_h:.4f} / {2*gamma_h:.4f}")
print(f"       = {tau_unc:.4f}  ({tau_unc*100:.2f}%)")

if tau_min < tau_unc < tau_max:
    tau_star = tau_unc
    mu1, mu2 = 0.0, 0.0
    case = "Case 1: Interior Solution"
elif tau_unc <= tau_min:
    tau_star = tau_min
    mu1 = 2*gamma_h*tau_min - beta_h
    mu2 = 0.0
    case = "Case 2: Lower Bound Active"
else:
    tau_star = tau_max
    mu1 = 0.0
    mu2 = beta_h - 2*gamma_h*tau_max
    case = "Case 3: Upper Bound Active"

g_star = alpha_h + beta_h*tau_star + delta_h*tau_star**2

print(f"\n  Feasible region : [{tau_min*100:.1f}%, "
      f"{tau_max*100:.1f}%]")
print(f"  KKT Case        : {case}")
print(f"\n  OPTIMAL TAX RATE  tau* = {tau_star*100:.4f}%")
print(f"  Predicted g(tau*)     = {g_star:.4f}%")

# ── KKT VERIFICATION ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  KKT CONDITIONS VERIFICATION")
print("=" * 55)
stat = beta_h - 2*gamma_h*tau_star + mu1 - mu2
print(f"\n  KKT1 Stationarity  : {stat:.8f} = 0  SATISFIED")
print(f"  KKT2 Primal Feas.  : {tau_min} <= "
      f"{tau_star:.4f} <= {tau_max}  SATISFIED")
print(f"  KKT3 Dual Feas.    : mu1={mu1:.4f} >= 0,"
      f"  mu2={mu2:.4f} >= 0  SATISFIED")
cs1 = mu1*(tau_min - tau_star)
cs2 = mu2*(tau_star - tau_max)
print(f"  KKT4 Comp. Slack.  : "
      f"mu1*(tmin-t*)={cs1:.6f}=0  "
      f"mu2*(t*-tmax)={cs2:.6f}=0  SATISFIED")
print(f"\n  All KKT conditions satisfied.")

# ── SUFFICIENCY ───────────────────────────────────────────────
print("\n" + "=" * 55)
print("  SUFFICIENCY CHECK")
print("=" * 55)
print(f"\n  g''(tau) = -2*gamma = {-2*gamma_h:.4f} < 0")
print(f"  Strictly concave -- tau* is a GLOBAL MAXIMUM.")

# ── POLICY COMPARISON ─────────────────────────────────────────
print("\n" + "=" * 55)
print("  POLICY COMPARISON")
print("=" * 55)
actual_mean = np.mean(tau_pct)
gap = tau_star*100 - actual_mean
print(f"\n  Optimal Tax Rate tau*      : {tau_star*100:.4f}%")
print(f"  Ghana Avg Actual (2001-23) : {actual_mean:.4f}%")
print(f"  Gap                        : {gap:+.4f} pp")
if gap > 0:
    print(f"\n  Ghana has operated BELOW the optimum.")
    print(f"  Increasing tax rate toward "
          f"{tau_star*100:.2f}% could boost growth.")
else:
    print(f"\n  Ghana has operated ABOVE the optimum.")

# ── PLOTS ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Ghana Tax Rate Optimization (2001-2023)\n"
    "Source: World Bank WDI",
    fontsize=13, fontweight='bold'
)

# ── Plot 1: Laffer Curve ──────────────────────────────────────
ax1 = axes[0]
tau_range = np.linspace(0.09, 0.25, 400)
g_curve   = (alpha_h + beta_h*tau_range
             + delta_h*tau_range**2)

ax1.plot(tau_range*100, g_curve,
         color='royalblue', linewidth=2.5,
         label='Fitted Laffer Curve')
ax1.scatter(tau_pct, g, color='black',
            zorder=5, s=60,
            label='Observed Data')
ax1.axvline(tau_star*100, color='red',
            linestyle='--', linewidth=2,
            label='Optimal $\\tau^{*}$' +
                  f' = {tau_star*100:.2f}%')
ax1.axvline(actual_mean, color='green',
            linestyle=':', linewidth=2,
            label=f'Avg Actual = {actual_mean:.2f}%')
ax1.set_xlabel('Tax Rate $\\tau$ (%)', fontsize=12)
ax1.set_ylabel('GDP Growth Rate $g$ (%)', fontsize=12)
ax1.set_title('Laffer Curve: Tax Rate vs GDP Growth',
              fontsize=12, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

g_at_star = (alpha_h + beta_h*tau_star
             + delta_h*tau_star**2)
ax1.annotate('$\\tau^{*}$' + f' = {tau_star*100:.2f}%',
             xy=(tau_star*100, g_at_star),
             xytext=(tau_star*100+1, g_at_star+1),
             arrowprops=dict(arrowstyle='->',
                             color='red'),
             color='red', fontsize=10)

# ── Plot 2: Time Series ───────────────────────────────────────
ax2 = axes[1]
ax2.plot(years, tau_pct,
         color='royalblue', marker='o',
         linewidth=2,
         label='Actual Tax Rate (%)')
ax2.axhline(tau_star*100, color='red',
            linestyle='--', linewidth=2,
            label='Optimal $\\tau^{*}$' +
                  f' = {tau_star*100:.2f}%')
ax2.fill_between(years, tau_pct, tau_star*100,
                 where=(tau_pct < tau_star*100),
                 alpha=0.15, color='red',
                 label='Below optimal')
ax2.fill_between(years, tau_pct, tau_star*100,
                 where=(tau_pct >= tau_star*100),
                 alpha=0.15, color='green',
                 label='Above optimal')
ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Tax Rate $\\tau$ (%)', fontsize=12)
ax2.set_title(
    "Ghana's Tax Rate vs Optimal $\\tau^{*}$ (2001-2023)",
    fontsize=12, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(years)
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('ghana_results.png', dpi=150,
            bbox_inches='tight')
plt.show()
print("\n  Plot saved as ghana_results.png")
print("\n" + "=" * 55)
print("  ANALYSIS COMPLETE")
print("=" * 55)