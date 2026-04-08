import scipy as sp
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint  
from src.utils.Registry import *

os.chdir("/Users/canderson/dev/registry-analysis/")
Path("out/plots").mkdir(exist_ok=True)
# registry list
tab = pd.read_csv("out/amazon-registry.csv")

# invites info
invites = pd.read_csv("in/invites.csv")
invites[np.isnan(invites.close)] = 0

#\\\\\\
#\\\\\\
# Analyze
#\\\\\\
#\\\\\\

plt.figure()
sns.histplot(tab.price, kde=True, bins = 8)
plt.title("Price Distribution")
plt.savefig('out/plots/registry_item_price_hist.pdf')

Prices =  tab['price'].sort_values().values

Z = Prices
# Z[Z==0] = Z.mean()
# Z = Z[Z<300]
Z = Z[Z>0]
Z = np.sort(Z)

# def show_beta(a,b):
#     n = 100
#     # Create points in [0, 1000]
#     x_scaled = np.linspace(0, 1000, n)
#     # Map to [0, 1] for Beta distribution
#     x_beta = x_scaled / 1000
#     # Get PDF values from Beta distribution
#     beta_dist = sp.stats.beta(a, b)
#     pdf_values = beta_dist.pdf(x_beta)
#     # Scale PDF to account for the transformation
#     # When you scale x by factor c, PDF scales by 1/c
#     pdf_scaled = pdf_values / 1000
#     pdf_normalized = pdf_scaled / pdf_scaled.sum()
#     plt.plot(pdf_normalized)

# def resample_beta(X, a, b, size, replace = True):
    
#     _min = X.min()
#     _max = X.max()

#     # Create points range
#     x_scaled = np.linspace(_min,_max, X.size)

#     # Map to [0, 1] for Beta distribution
#     x_beta = (x_scaled - _min) / (_max - _min)

#     # Get PDF values from Beta distribution
#     beta_dist = sp.stats.beta(a, b)
#     pdf_values = beta_dist.pdf(x_beta)
#     # Scale PDF to account for the transformation
#     # When you scale x by factor c, PDF scales by 1/c
#     pdf_scaled = pdf_values / (_max - _min)

#     # make tails non zero
#     pdf_scaled[pdf_scaled == 0] = 1e-6

#     pdf_normalized = pdf_scaled / pdf_scaled.sum()

#     if not replace:
#         size = X.size
    
#     Z = np.random.choice(a=Prices, size=size, replace=replace , p=pdf_normalized)
#     Z = np.sort(Z)
#     return Z


# # # Resample Prices with Beta
# # Beta parameters 
# # a, b = 5, 2
# # a, b = 4,2
# # a, b = 5, 5
# # show_beta(a,b)
# # Z = resample_beta(Prices, a,b, size = 200, replace = False)
# # sns.histplot(Z, kde = True)
# # plt.title("Resampled Prices Distribution")
# # plt.show()

# # Simulate Prices from beta
# def sim_beta(a,b,n):
#     np.random.seed(1)
#     beta = sp.stats.beta(a = a,b = b)
#     beta_sample = beta.rvs(n)
#     return beta_sample

# a, b = 1,8
# beta_sample = sim_beta(a,b,100)
# sns.kdeplot(beta_sample, clip = (0,1))
# plt.show()

# # rescale to price range
# _max, _min = (Prices.max()+0), (Prices.min() - 0)
# Z_beta = (beta_sample) * (_max - _min) + _min
# sns.histplot(Z_beta, kde = True)
# plt.title("Simulated Prices Distribution")
# plt.show()







#\\\\
#\\\\
#\\\\
# Price Tier Completion 1
#\\\\
#\\\\
#\\\\

# Under $50: ~60–75% purchase rate
# $50–$150: ~30–50% purchase rate
# $150–$300: ~15–25% purchase rate
# $300+: ~5–15% purchase rate

def completion(amt):
    if amt <= 50:
        return (.60,.75)
    elif amt > 50 and amt <= 150:
        return (.30,.50)
    elif amt > 150 and amt <= 300:
        return (.15,.25)
    elif amt >300: 
        return (.05,.15)
    else:
        return np.nan
    
lwr = np.array([completion(round(x))[0] for x in Z])
uppr = np.array([completion(round(x))[1] for x in Z])

# proportion of total number of items
item_completion = lwr.mean(), uppr.mean()

# proportion of total possible value
dollar_completion = ((Z*lwr).sum() / Z.sum()), ((Z*uppr).sum() / Z.sum())

print(f"Expected proportion of items completed = ({item_completion[0]:.3f}, {item_completion[1]:.3f})")
print(f"Expected proportion of total value completed = ({dollar_completion[0]:.3f}, {dollar_completion[1]:.3f})")
##----
# Proportion of items completed = (0.452, 0.601)
# Proportion of total value completed = (0.235, 0.360)
##----


#\\\\
#\\\\
# Price Tier Completion 2
#\\\\
#\\\\

# https://guides.myregistry.com/gift-list/the-2026-formula-for-the-perfect-registry-size/
# < $50	    30–35%
# $50–$100	30–35%
# $100–$250	20–25%
# $250–$500	8–12%
# $500<	    5–8%	
# funds	    10–15%

def percents(amt):
    return np.array([
        np.mean(amt <= 50),
        np.mean((amt > 50) & (amt <= 100)),
        np.mean((amt > 100) & (amt <= 250)),
        np.mean((amt > 250) & (amt <= 500)),
        np.mean(amt >500 )
        ])   


ideal_prop= np.array([.30,.35, .20,.12,.05])
actual_prop = percents(Z) + 0.001


prices = np.array([50,100,250,500,750])
# plt.figure()
# plt.plot(prices, actual_prop, label = "Actual")
# plt.plot(prices,ideal_prop,  label = "Ideal")
# plt.xlim(0,500)
# plt.legend()
# plt.title("Price Tier Proportion of Registry")
# plt.xlabel("Price")
# plt.ylabel("Proportion")
# plt.show()

rmse = np.sum((actual_prop- ideal_prop)**2)
cross_ent = -np.sum(actual_prop*np.log(ideal_prop))
BC = np.sum(np.sqrt(actual_prop*ideal_prop))
KS = np.max( np.abs(np.cumsum(actual_prop) - np.cumsum(ideal_prop)))
WASS = sp.stats.wasserstein_distance(prices, prices, actual_prop, ideal_prop)
print(f"RMSE = {rmse:.3f}")
print(f"CE = {cross_ent:.3f}")
print(f"BC = {BC:.3f}")
print(f"1-KS = {1-KS:.3f}")
print(f"WASS = {WASS:.3f}")
# RMSE = 0.159
# CE = 1.360
# BC = 0.933
# 1-KS = 0.679
# WASS = 52.682



# \\\\
# \\\\
# Number of items near mean
# Int. J. Electronic Marketing and Retailing, Vol. 5, No. 4, 2013 359Copyright © 2013 Inderscience Enterprises Ltd. How do external reference prices influence online gift giving? Yun Kyung Oh* 
# \\\\
# \\\\
#      so to summarize. I specify the probability of item being purchased as coming from a bimodal distribution centered at the prices avg. further positive or negative away from this
#   center incrfeases the probability of purchase where the high max is at 1.7 (170% of price_avg) and .6 (60% of price average). i set the mixture of these two pdfs using a ratio
#   of .55 low + .45 high and scale their probabilites to lie between .4 and .7 purchase probability. with these i then model the probability that all items are purchased by
#   running a MC simulation multiple times for all guests where i specify their probability of any purchase, if they 'choose' to, then i model their choice as a draw from the
#   available items using their calculated probabilities. I use the proportion of close invites as an estimate of the probability a single guest will buy a gift. i then report the
#   average purchase comletion rates and its 95% confidence intervals 
    
fulfillment = pd.DataFrame({
    "store":["bloomingdales", "macys", "kohls", "sears", "williamsonoma", "potterybarn","crateandbarrel"],
    "no_requested_categories": [3.41, 3.91, 4.18, 3.56, 3.63, 3.51, 4.13], 
    "no_requested_categories_sd": [1.22, 1.11, 1.09, 1.13, .97, 1.03, 1.05], 
    "avg_requested_price":[69.94, 47.74, 19.11, 51.62, 54.84, 54.28, 21.69],
    "avg_requested_price_sd":[35.93, 2.17, 11.13, 45.70, 36.14, 41.77, 26.5],
    "item_fulfillment_rate": [.63, .47, .45, .22, .48, .52, .63], 
    "dollar_fulfillment_rate": [.57, .41, .40, .15, .48, .42, .55]
})


# fig, ax = plt.subplots(3,1, figsize = (5,10))
# sns.scatterplot(fulfillment, x = "avg_requested_price", y = "item_fulfillment_rate", ax=ax[0])
# ax[0].set_title("Price Versus Fulfillment")
# sns.scatterplot(fulfillment, x = "avg_requested_price_sd", y = "item_fulfillment_rate", ax=ax[1])
# ax[1].set_title("Price Stdev Versus Fulfillment")
# sns.scatterplot(fulfillment, x = "no_requested_categories", y = "item_fulfillment_rate", ax=ax[2])
# ax[2].set_title("No. Requested Categories Versus Fulfillment")
# plt.tight_layout()
# plt.show()

# # estimate monetary and social benefit
# soc_ben = (Z - Z.mean()) / Z.max()
# # mon_ben = (Z.mean() - Z) / Z.max()
# mon_ben = -1*soc_ben

# plt.axhline(0 , linestyle='--', color = "black")
# plt.plot(Z, mon_ben, label = 'Monetary Benefit')
# plt.plot(Z, soc_ben, label = 'Social Benefit')
# plt.legend()
# plt.show()


# Create registry
registry = Registry(Z)

# Calculate probabilities
lo_hi_mx = (1 - invites.close.mean(), invites.close.mean())
registry.calculate_item_probabilities(low_high_mix=lo_hi_mx)

# Run simulation
results = registry.simulate_registry_completion(
    num_guests=invites.shape[0],
    guest_p_buy=fulfillment['item_fulfillment_rate'].mean(),
    num_simulations=500,
    seed=124
)

# Get summary
summary = registry.get_summary_stats(results)

print(f"Average number of expected buyers: {np.array(summary['expected_buyers']).mean():.2f} out of {invites.shape[0]} ({np.array(summary['expected_buyers']).mean()/invites.shape[0]*100:.0f}%)")
print(f"Completion rate: {summary['completion_rate']:.1%}")
print(f"Average items purchased: {summary['avg_items_purchased']:.1f} / {summary['n_items']}")
print(f"Average fulfillment rate: {summary['fulfillment_rate_mean']:.4f} ")


def interp(x, y, by=5):
    # Handle duplicates by averaging probabilities for duplicate prices
    sorted_idx = np.argsort(x)
    x_sorted = x[sorted_idx]
    sorted_probs = y[sorted_idx]
    # Get unique prices and average their probabilities
    unique, inverse_indices = np.unique(x_sorted, return_inverse=True)
    unique_probs = np.array([sorted_probs[inverse_indices == i].mean() for i in range(len(unique))])
    cs = sp.interpolate.CubicSpline(unique, unique_probs, bc_type="natural")
    x_range = np.arange(x.min(), x.max(), by)
    y_interp = cs(x_range)
    return x_range, y_interp


price_range, price_probs = interp(registry.prices, registry._item_probs)

plt.figure()
sns.histplot(registry.prices, stat="probability", bins=15)
plt.plot(price_range, price_probs, color="orange", label="Bimodal Purchase Probability")
plt.axvline(registry.prices.mean(), linestyle = '--', color = 'red', label = "Average Price")
plt.xlabel('Price')
plt.ylabel('Purchase Probability')
plt.title('Item Price Distribution and Purchase Probability')
plt.legend()
plt.savefig('out/plots/price_range_purchase_probabilities.pdf')

plt.figure()
sns.histplot(results['items_purchased_per_sim']/registry.N, kde = True)
plt.axvspan(*summary['fulfillment_rate_ci'], alpha = .5, color = "orange")
plt.axvline(np.mean(summary['fulfillment_rate_ci']), color = "green", linestyle = '--')
plt.title("Item Purchase Rate Per Simulation (#bought/total items)")
plt.savefig('out/plots/MC_sim_fulfillment_rate.pdf')


# \\\\
# \\\\
# What items to keep?
# \\\\
# \\\\

def fulfillmentRate(prices):
    registry = Registry(prices)
    # Calculate probabilities
    registry.calculate_item_probabilities(low_high_mix=lo_hi_mx)
    results = registry.simulate_registry_completion(
        num_guests=invites.shape[0],
        guest_p_buy=fulfillment['item_fulfillment_rate'].mean(),
        num_simulations=500,
        seed=42  
    )
    # Get summary
    summary = registry.get_summary_stats(results)
    return {"completion_rate": summary['completion_rate'],
            "fulfillment_rate_mean": summary['fulfillment_rate_mean'],
            "lower": summary['fulfillment_rate_ci'][0],
            "upper": summary['fulfillment_rate_ci'][1]
            } 
    
res = []
thresholds = np.arange(-300,200, 10)
for t in thresholds:
    if t<0:
        prices = Z[Z<=np.abs(t)]
        res.append(fulfillmentRate(prices))
    else:
        prices = Z[Z>=t]
        res.append(fulfillmentRate(prices))
        
res = pd.DataFrame(res)
res["threshold"] = thresholds


# plot thresholds against fulfillment rate
## yvals where x< 0 show rate for where prices > x value are excluded; vice verssa for x > 0
fig, ax = plt.subplots(figsize = (15,8))
ax.fill_between(res['threshold'], res.lower, res.upper, alpha = .4, color = 'orange')
ax.plot(res.threshold, res.fulfillment_rate_mean, color = 'orange')
plt.xlabel("Threshold (negative = prices[price < x], positive = prices[price > x])")
plt.ylabel("Average Fulfillment Rate of Filtered Prices")
plt.title("Filter effect on fulfillment rate")
# plt.ylim((0,None))
plt.savefig('out/plots/filter_effect_on_fulfillment_rate.pdf')

res.loc[(res.threshold > 0) & (res['fulfillment_rate_mean'] >.8 ), :]
fulfillmentRate(Z[(Z>=50)])
# {'completion_rate': 0.916,
#  'fulfillment_rate_mean': np.float64(0.9905263157894738),
#  'fulfillment_rate_ci': (np.float64(0.9870882752216066),
#   np.float64(0.993964356357341))}
## 
## Theoretically, setting the registry to have items worth > $50 dollars would maximize the chances of fulfillment
## 

# show price distribution with this threshold
price_thresh = (50,np.inf)
reg = Registry(Z[(Z>=price_thresh[0]) & (Z< price_thresh[1])])
reg.calculate_item_probabilities( low_high_mix= lo_hi_mx)

filt_price_range, filt_price_probs = interp(reg.prices, reg._item_probs)

plt.figure()
sns.histplot(reg.prices, stat="probability", bins=10, kde=True)
plt.plot(filt_price_range, filt_price_probs, color="orange", label="Bimodal Purchase Probability")
plt.axvline(reg.prices.mean(), linestyle = '--', color = 'red', label = "Average Price")
plt.xlabel('Price')
plt.ylabel('Count')
plt.title(f'Item Price Distribution and Purchase Probability for prices [{price_thresh[0]}, {price_thresh[1]})')
plt.legend()
plt.savefig('out/plots/optimal_price_range_purchase_probabilities.pdf')

# which items to remove?
tab.loc[tab.price < 30][["full_title", "price"]].to_csv("out/items_lsthn_30.csv", index = False)

