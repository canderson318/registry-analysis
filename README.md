# Registry Analysis
Optimizing an Amazon wedding registry to maximize item fulfillment rate.

Inspired by Oh et al. [paper](other/documents/oh2013.pdf), which models how external reference prices (relative to the gift-giver's anchor) influence online gift selection, this project applies a bimodal purchase-probability model to a real registry and uses Monte Carlo simulation to estimate fulfillment rates under different price-filtering strategies.

## Purpose

Given a fixed guest list, determine which registry items to keep (or remove) in order to maximize the probability that every item gets purchased. The core insight is that items priced near 60% or 170% of the registry's average price attract the highest purchase probability — one cluster driven by perceived affordability, the other by perceived generosity.

## Project Outline

```
registry-analysis/
├── in/
│   ├── grid.html          # Raw Amazon registry HTML export
│   └── invites.csv        # Guest list with RSVP/close-contact flags
├── src/
│   ├── 01.py              # Parse registry HTML → out/amazon-registry.csv
│   ├── 02.py              # Analysis, simulation, and plotting
│   └── utils/
│       └── Registry.py    # Registry class: probability model + MC simulation
└── out/
    ├── amazon-registry.csv
    ├── items_lsthn_30.csv  # Items flagged for removal (price < $30)
    └── plots/
        ├── registry_item_price_hist.pdf
        ├── price_range_purchase_probabilities.pdf
        ├── MC_sim_fulfillment_rate.pdf
        ├── filter_effect_on_fulfillment_rate.pdf
        └── optimal_price_range_purchase_probabilities.pdf
```

## Methodology

1. **Scrape** — `01.py` parses the registry HTML with BeautifulSoup, extracts item titles and prices, and writes a clean CSV.

2. **Probability model** — Each item is assigned a purchase probability using a bimodal Gaussian mixture (via `Registry.calculate_item_probabilities`). The two peaks are centered at 0.6× and 1.7× the average registry price, reflecting the empirical finding that guests gravitate toward items that feel either affordable or aspirational relative to the group anchor. The mixture weights are set using the proportion of close vs. non-close invitees.

3. **Monte Carlo simulation** — `Registry.simulate_registry_completion` runs *N* simulations. In each run, the number of buying guests is drawn from a Binomial distribution parameterized by the guest count and an empirical purchase-rate prior (sourced from fulfillment data across major retailers). Each buyer selects an available item via weighted sampling over item probabilities.

4. **Price threshold analysis** — `02.py` sweeps a range of price cutoffs and re-runs the simulation for each filtered registry to produce a fulfillment-rate vs. threshold curve. The analysis finds that restricting items to **≥ $50** raises the expected fulfillment rate to ~99%.

## Key Finding

Removing items priced below $50 is expected to raise the item fulfillment rate from ~45–60% to ~99% (95% CI: 98.7%–99.4%), given the current guest list size and empirical buyer rates.
