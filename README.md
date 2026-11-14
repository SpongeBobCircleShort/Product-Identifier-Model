# Product Identifier Model

A small prototype that reads a casual social-media post and names the Apple product it mentions, even through typos, shorthand and emoji. Two predictions from the notebook's saved run: "just copped the ip16pro… worth it??" → *iPhone 16 Pro*, "airpd max noise canceling >>>>" → *AirPods Max*.

## How it works

Everything below happens inside `Product Classification.ipynb`.

1. **Labels and aliases.** The notebook builds its own Apple-only alias map. iPhone and MacBook aliases are generated in loops (`IP 16 Pro Max`, `Iphone16ProMax`, `Apple phone 16 Pro Max`); iPad, Watch and AirPods aliases are written out by hand; a separate map adds M-series chip names (`M3 Max`, `M3Max`, `M3 Max chip`). That gives 132 classes: 43 iPhones (8 to 20, with Pro, Pro Max, Plus and SE variants), 4 iPads, 3 Apple Watches, 2 AirPods, 64 MacBook Air/Pro size and chip combinations, and 16 chips (M1 to M4, each as base, Pro, Max and Ultra).
2. **Synthetic posts.** A randomly chosen alias is dropped into one of 10 comment templates ("Just bought the {} — loving it so far!"), 50 times per product and 30 times per chip, for 6,280 training posts.
3. **Embeddings.** Each post is embedded with the `all-MiniLM-L6-v2` sentence-transformer.
4. **Classifier.** A logistic-regression model is trained on those embeddings to predict the label. `identify_apple_product(text)` runs a new post through the same embedding and classifier.

The notebook ends with three sets of test sentences: clean, conversational, and deliberately messy edge cases.

## Data files the notebook does not use yet

- `Product Map for Identifier Model.py` (the same map is stored in `Product Map for Identifier Model.json`): a hand-written map of 246 products, mostly from other brands (Dell, LG, HP, Samsung, Sony and more), with 878 aliases.
- `NER Group Mapping.csv`: 1,000 brand-to-category rows covering about 950 brands and 14 categories (for example Apple → Electronics & Tech). Some brands appear under more than one category.

The notebook loads neither file, so the model cannot return any of those products or categories.

## Run it

Written and run on Kaggle (Python 3.11). To run it locally:

```bash
pip install sentence-transformers scikit-learn pandas notebook
jupyter notebook "Product Classification.ipynb"
```

The first run downloads the `all-MiniLM-L6-v2` model from Hugging Face, so it needs network access. No random seed is set, so each run builds different training posts and can give different predictions from the saved ones.

## Limitations

- Training data is synthetic template sentences built from the aliases, not real posts, and there is no held-out evaluation yet; the test sentences in the notebook are spot checks.
- Some generated product names don't exist (for example future iPhone numbers and chip/size combinations Apple doesn't sell), so the label space is larger than the real catalogue.
- It always returns one of its 132 labels, even for a post about another brand or about no product at all.
- The saved test outputs show three kinds of mistake: it sometimes drops the chip tier (`MBP14M4Ultra` → *MacBook Pro 14-inch (M4)*), it fills in a chip generation or screen size the post doesn't give (`mb air 15 ultra chip` → *MacBook Air 15-inch (M2 Ultra)*), and it returns only one label for a post that names two products.
- The MacBook alias generator builds shorthand as `f"{model[:1]}B{model[1:]}"`, which gives `PBro 14 M4 Ultra` and `ABir 13 M1 Max`; the common `MBP` and `MBA` shorthand is not in the alias list.
