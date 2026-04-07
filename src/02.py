import scipy as sp
from pathlib import Path
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint  
from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any

os.chdir("/Users/canderson/dev/amazon-registry-analysis/")
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


class Registry:
    def __init__(self, prices: np.ndarray):
        """
        Initialize registry with item prices.
        Args:
            prices: Array of item prices
        """
        self.prices = np.asarray(prices)
        self._item_probs = None
        self.N = len(prices)
    
    def __len__(self):
        return len(self.prices)
    
    def __iter__(self):
        return iter(self.prices)
    
    def calculate_item_probabilities(self, 
                                     locs: Tuple[float,float] = (0.6,1.7),
                                     vars: Tuple[float,float] = ( 0.15**2, 0.30**2),
                                     low_high_mix: Tuple[float, float] = (0.55, 0.45),
                                     p_range: Tuple[float,float] = (0.40,0.70),
                                     clip: Tuple[float,float] = (0.2, 0.75)) -> np.ndarray:
        """
        Calculate purchase probabilities using a bimodal distribution.
        Args:
            low_high_mix: Tuple of (low_weight, high_weight) for mixture
            high_loc: Center of high-price peak (ratio to average)
            low_loc: Center of low-price peak (ratio to average)
            high_var: Variance of high-price peak
            low_var: Variance of low-price peak
            p_floor: Minimum probability at the dip
            p_ceiling: Maximum probability at peaks
            abs_min: Absolute minimum probability (clipping)
            abs_max: Absolute maximum probability (clipping)
        Returns:
            Array of purchase probabilities for each item
        """
        avg_price = np.mean(self.prices)
        ratios = self.prices / avg_price
        
        # Smooth bimodal probability
        low_peak = np.exp(-((ratios - locs[0])**2) / (2 * vars[0]))
        high_peak = np.exp(-((ratios - locs[1])**2) / (2 * vars[1]))
        
        mixture = low_high_mix[0] * low_peak + low_high_mix[1] * high_peak
        
        # Scale to empirical range
        p_base = p_range[0] + (mixture / mixture.max()) * (p_range[1] - p_range[0])
        
        self._item_probs = np.clip(p_base, clip[0], clip[1])
        return self._item_probs
    
    def simulate_registry_completion(self,num_guests: int,guest_p_buy: float = 0.8,num_simulations: int = 1000,item_probs: Optional[np.ndarray] = None,seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Monte Carlo simulation of registry completion.
        Args:
            num_guests: Number of guests invited
            guest_p_buy: Probability that a guest purchases a gift
            num_simulations: Number of Monte Carlo simulations
            item_probs: Pre-calculated item probabilities (if None, uses cached)
            seed: Random seed for reproducibility
        Returns:
            Dictionary with simulation results
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Use provided or cached probabilities
        if item_probs is None:
            if self._item_probs is None:
                raise ValueError("Must call calculate_item_probabilities first or provide item_probs")
            item_probs = self._item_probs
        
        n_items = len(self.prices)
        
        # Run simulations
        completion_count = 0
        items_purchased_per_sim = []
        buyers = []
        for _ in range(num_simulations):
            avail = np.ones(n_items, dtype=bool)
            # model guest buying with binomial: choice ~ Binom(trials, p_success)
            expected_buyers = np.random.binomial(num_guests, guest_p_buy)
            buyers.append(expected_buyers)
            
            for _ in range(expected_buyers):
                if not avail.any():
                    break
                
                available_probs = item_probs * avail
                prob_sum = available_probs.sum()
                
                if prob_sum > 0:  # Safety check
                    choice = np.random.choice(
                        n_items, 
                        p=available_probs / prob_sum
                    )
                    avail[choice] = False
            
            purchased = ~avail
            items_purchased_per_sim.append(purchased.sum())
            if purchased.all():
                completion_count += 1
        
        items_purchased_array = np.array(items_purchased_per_sim)
        
        return {
            "expected_buyers": buyers,
            "completion_rate": completion_count / num_simulations,
            "completion_count": completion_count,
            "items_purchased_per_sim": items_purchased_array,
            "avg_items_purchased": np.mean(items_purchased_array),
            "std_items_purchased": np.std(items_purchased_array, ddof=1),
            "item_probabilities": item_probs
        }
    
    @staticmethod
    def confint(loc: float, std: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """
        Calculate confidence interval using t-distribution.
        """
        alpha = 1 - confidence
        q = 1 - alpha / 2
        t = sp.stats.t.ppf(q, df=n-1)
        se = std / np.sqrt(n)
        err = t * se
        return loc - err, loc + err
    
    def get_summary_stats(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate summary statistics from simulation results.
        Args:
            simulation_results: Output from simulate_registry_completion
        Returns:
            Dictionary with summary statistics
        """
        items_purchased = simulation_results['items_purchased_per_sim']
        n_items = len(self.prices)
        
        item_rate = items_purchased / n_items
        rate_mean = item_rate.mean()
        rate_std = np.std(item_rate, ddof=1)
        
        rate_ci = self.confint(rate_mean, rate_std, len(item_rate))
        items_ci = self.confint(
            items_purchased.mean(), 
            simulation_results['std_items_purchased'], 
            len(items_purchased)
        )
        
        return {
            "n_items": n_items,
            "expected_buyers": simulation_results['expected_buyers'],
            "completion_rate": simulation_results['completion_rate'],
            "avg_items_purchased": simulation_results['avg_items_purchased'],
            "items_ci": items_ci,
            "fulfillment_rate_mean": rate_mean,
            "fulfillment_rate_std": rate_std,
            "fulfillment_rate_ci": rate_ci
        }

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

print(f"Number of expected buyers: {summary['expected_buyers']}")
print(f"Completion rate: {summary['completion_rate']:.1%}")
print(f"Average items purchased: {summary['avg_items_purchased']:.1f} / {summary['n_items']}")
print(f"Fulfillment rate: {summary['fulfillment_rate_mean']:.4f} "
        f"[{summary['fulfillment_rate_ci'][0]:.3f}, {summary['fulfillment_rate_ci'][1]:.3f}]")


plt.figure()
sns.histplot(registry.prices, stat="probability", bins=15)  
plt.plot(registry.prices, registry._item_probs,color = "orange", label = "Bimodal Purchase Probability")
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

plt.figure()
sns.histplot(reg.prices, stat="probability", bins = 10, kde = True)  
plt.plot(reg.prices, reg._item_probs ,color = "orange", label = "Bimodal Purchase Probability")
plt.axvline(reg.prices.mean(), linestyle = '--', color = 'red', label = "Average Price")
plt.xlabel('Price')
plt.ylabel('Count')
plt.title(f'Item Price Distribution and Purchase Probability for prices [{price_thresh[0]}, {price_thresh[1]})')
plt.legend()
plt.savefig('out/plots/optimal_price_range_purchase_probabilities.pdf')

# which items to remove?
tab.loc[tab.price < 30][["full_title", "price"]].to_csv("out/items_lsthn_30.csv", index = False)

