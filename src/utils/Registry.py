from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any
import numpy as np
import scipy as sp

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
                                     low_high_mix: Tuple[float, float] = (0.5, 0.5),
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

