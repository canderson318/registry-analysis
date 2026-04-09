from numpy.typing import NDArray
from typing import Optional, Tuple, Dict, Any
import numpy as np
import scipy as sp


class Registry2:
    """
    Extended registry model with per-gifter-type purchase probabilities.

    Adds two attenuation effects grounded in gift-giving literature
    (Wang & van der Lans, 2018):

      1. Preference-match sharpening (closeness-driven, positive)
         Close gifters have lower residual uncertainty about which registry
         items the receiver actually wants, so their probability mass is more
         tightly concentrated at the bimodal peaks.

      2. Price attenuation (price-driven, moderated by closeness, negative)
         Both gifter types experience a price penalty beyond a threshold, but:
         - Close gifters behave like self-purchasers: stronger price sensitivity
         - Non-close gifters use price as a relationship signal: weaker penalty,
           so expensive items remain relatively attractive to them.

    The simulation draws a gifter type at each purchase step and samples from
    that type's probability vector, rather than a single shared vector.
    """

    def __init__(self, prices: np.ndarray):
        self.prices = np.asarray(prices, dtype=float)
        self.N = len(prices)
        self._probs_close: Optional[np.ndarray] = None
        self._probs_nonclose: Optional[np.ndarray] = None

    def __len__(self):
        return self.N

    def __iter__(self):
        return iter(self.prices)

    # ------------------------------------------------------------------
    # Probability model
    # ------------------------------------------------------------------

    def calculate_item_probabilities(
        self,
        # bimodal shape (same as Registry v1)
        locs: Tuple[float, float] = (0.6, 1.7),
        vars: Tuple[float, float] = (0.15**2, 0.30**2),
        p_range: Tuple[float, float] = (0.40, 0.70),
        clip: Tuple[float, float] = (0.20, 0.75),
        # --- new: preference sharpening ---
        pref_sharpening: float = 0.50,
        # --- new: price attenuation ---
        gamma_close: float = 0.45,
        gamma_nonclose: float = 0.10,
        attenuation_onset: float = 1.2,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute two item probability vectors — one per gifter type.

        Parameters
        ----------
        locs : (low_peak, high_peak)
            Bimodal peak centers as multiples of the average price.
            Defaults from Oh et al. (2013): 0.6× and 1.7× avg price.
        vars : (low_var, high_var)
            Variance of each Gaussian peak for non-close gifters.
        p_range : (min_prob, max_prob)
            Scale the bimodal mixture to this probability range before
            applying attenuation.
        clip : (abs_min, abs_max)
            Hard clip applied after attenuation.
        pref_sharpening : float in [0, 1)
            Fraction by which to reduce peak variance for close gifters.
            Higher → more concentrated probability at bimodal peaks.
            Reflects lower residual uncertainty about receiver preferences.
        gamma_close : float >= 0
            Exponential price-attenuation rate for close gifters.
            Larger → steeper drop-off for expensive items (more price sensitive).
            Prior: N(0.45, 0.15), from paper H3: close gifters ≈ self-purchasers.
        gamma_nonclose : float >= 0
            Attenuation rate for non-close gifters.
            Smaller → expensive items stay attractive (price signaling motive).
            Prior: N(0.10, 0.10), from paper H2: gifters are less price sensitive.
        attenuation_onset : float
            Price ratio above which attenuation begins. Items below this
            multiple of average price are not penalized. Default 1.2× avg.

        Returns
        -------
        probs_close, probs_nonclose : np.ndarray
            Per-item purchase probabilities for each gifter type.
        """
        avg_price = np.mean(self.prices)
        ratios = self.prices / avg_price  # normalized price ratios

        # ----------------------------------------------------------
        # 1. Bimodal preference-match signal
        # ----------------------------------------------------------
        # Non-close: standard peak widths (more diffuse — higher uncertainty)
        low_nc  = np.exp(-((ratios - locs[0])**2) / (2 * vars[0]))
        high_nc = np.exp(-((ratios - locs[1])**2) / (2 * vars[1]))

        # Close: sharpened peaks — lower residual uncertainty → more targeted
        sharp_vars = (vars[0] * (1 - pref_sharpening),
                      vars[1] * (1 - pref_sharpening))
        low_cl  = np.exp(-((ratios - locs[0])**2) / (2 * sharp_vars[0]))
        high_cl = np.exp(-((ratios - locs[1])**2) / (2 * sharp_vars[1]))

        # Registry = full information condition from Wang & van der Lans (2018):
        # both types treat registry items as the receiver's stated preferences,
        # so we use equal mixture weights (0.5 / 0.5) for the bimodal.
        # Closeness drives attenuation, not mixture shape.
        mix_nc = 0.5 * low_nc + 0.5 * high_nc
        mix_cl = 0.5 * low_cl + 0.5 * high_cl

        # Scale bimodal to the empirical probability range
        def scale(m):
            return p_range[0] + (m / m.max()) * (p_range[1] - p_range[0])

        base_nc = scale(mix_nc)
        base_cl = scale(mix_cl)

        # ----------------------------------------------------------
        # 2. Price attenuation
        # ----------------------------------------------------------
        # Exponential decay applied only above `attenuation_onset` ratio.
        # exp(-gamma * max(ratio - onset, 0))
        #   gamma_close    large  → close gifters heavily penalize expensive items
        #   gamma_nonclose small  → non-close gifters stay willing to buy expensive
        excess = np.maximum(ratios - attenuation_onset, 0.0)
        att_nc = np.exp(-gamma_nonclose * excess)
        att_cl = np.exp(-gamma_close    * excess)

        # ----------------------------------------------------------
        # 3. Combine and clip
        # ----------------------------------------------------------
        probs_nc = np.clip(base_nc * att_nc, clip[0], clip[1])
        probs_cl = np.clip(base_cl * att_cl, clip[0], clip[1])

        self._probs_nonclose = probs_nc
        self._probs_close    = probs_cl
        return probs_cl, probs_nc

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate_registry_completion(
        self,
        num_guests: int,
        close_fraction: float,
        guest_p_buy: float = 0.80,
        num_simulations: int = 1000,
        probs_close: Optional[np.ndarray] = None,
        probs_nonclose: Optional[np.ndarray] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation of registry completion with two gifter types.

        At each purchase step, a gifter type (close / non-close) is drawn
        from Bernoulli(close_fraction), then the item is sampled from that
        type's probability vector over still-available items.

        Parameters
        ----------
        num_guests : int
            Total number of invited guests.
        close_fraction : float in [0, 1]
            Proportion of guests who are close (invites.close.mean()).
        guest_p_buy : float
            Probability that any given guest purchases a gift.
        num_simulations : int
            Number of Monte Carlo runs.
        probs_close, probs_nonclose : np.ndarray, optional
            Pre-computed probability vectors. Uses cached values if None.
        seed : int, optional
        """
        if seed is not None:
            np.random.seed(seed)

        if probs_close is None:
            if self._probs_close is None:
                raise ValueError("Call calculate_item_probabilities first.")
            probs_close = self._probs_close
        if probs_nonclose is None:
            probs_nonclose = self._probs_nonclose

        n_items = len(self.prices)
        completion_count = 0
        items_purchased_per_sim = []
        buyers_per_sim = []
        # track which type claimed each item
        claimed_by_close    = np.zeros(n_items, dtype=int)
        claimed_by_nonclose = np.zeros(n_items, dtype=int)

        for _ in range(num_simulations):
            avail = np.ones(n_items, dtype=bool)
            n_buyers = np.random.binomial(num_guests, guest_p_buy)
            buyers_per_sim.append(n_buyers)

            # draw all gifter types up front for this simulation
            is_close = np.random.binomial(1, close_fraction, size=n_buyers).astype(bool)

            for buyer_is_close in is_close:
                if not avail.any():
                    break

                probs = probs_close if buyer_is_close else probs_nonclose
                available_probs = probs * avail
                prob_sum = available_probs.sum()

                if prob_sum > 0:
                    choice = np.random.choice(n_items, p=available_probs / prob_sum)
                    avail[choice] = False
                    if buyer_is_close:
                        claimed_by_close[choice] += 1
                    else:
                        claimed_by_nonclose[choice] += 1

            purchased = ~avail
            items_purchased_per_sim.append(purchased.sum())
            if purchased.all():
                completion_count += 1

        items_purchased_array = np.array(items_purchased_per_sim)

        return {
            "completion_rate":           completion_count / num_simulations,
            "completion_count":          completion_count,
            "items_purchased_per_sim":   items_purchased_array,
            "avg_items_purchased":       np.mean(items_purchased_array),
            "std_items_purchased":       np.std(items_purchased_array, ddof=1),
            "expected_buyers":           buyers_per_sim,
            "probs_close":               probs_close,
            "probs_nonclose":            probs_nonclose,
            "claimed_by_close":          claimed_by_close,
            "claimed_by_nonclose":       claimed_by_nonclose,
        }

    # ------------------------------------------------------------------
    # Summary & CI (identical to v1)
    # ------------------------------------------------------------------

    @staticmethod
    def confint(loc: float, std: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        alpha = 1 - confidence
        t = sp.stats.t.ppf(1 - alpha / 2, df=n - 1)
        se = std / np.sqrt(n)
        return loc - t * se, loc + t * se

    def get_summary_stats(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        items_purchased = simulation_results["items_purchased_per_sim"]
        item_rate = items_purchased / self.N
        rate_mean = item_rate.mean()
        rate_std  = np.std(item_rate, ddof=1)
        return {
            "n_items":               self.N,
            "expected_buyers":       simulation_results["expected_buyers"],
            "completion_rate":       simulation_results["completion_rate"],
            "avg_items_purchased":   simulation_results["avg_items_purchased"],
            "items_ci":              self.confint(items_purchased.mean(),
                                                  simulation_results["std_items_purchased"],
                                                  len(items_purchased)),
            "fulfillment_rate_mean": rate_mean,
            "fulfillment_rate_std":  rate_std,
            "fulfillment_rate_ci":   self.confint(rate_mean, rate_std, len(item_rate)),
        }
