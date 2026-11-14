# Product Identifier Model

A small prototype that reads a casual social-media post and names the exact product it mentions, even through typos, shorthand and emoji: "just copped the ip16pro… worth it??" → *iPhone 16 Pro*, "airpd max noise canceling >>>>" → *AirPods Max*.

## How it works

1. **Alias map.** Each canonical product gets a list of the ways people actually write it (`IP 16 Pro Max`, `Iphone16ProMax`, `Apple phone 16 Pro Max`). The notebook generates these for Apple lines (iPhone, iPad, MacBook, Watch, AirPods); `Product Map for Identifier Model.py` adds a hand-written map of 246 products across other brands (Samsung, Sony, Dell, LG, HP and more) with 878 aliases.
2. **Embeddings.** Every alias is embedded with a sentence-transformer.
3. **Classifier.** A logistic-regression head is trained on those embeddings to predict the canonical product.
4. **Brand groups.** `NER Group Mapping.csv` maps about 1,000 brands to a category (for example Apple → Electronics & Tech), for grouping results downstream.

`Product Classification.ipynb` holds the whole pipeline plus three sets of test sentences: clean, conversational, and deliberately messy edge cases.

## Run it

The notebook was written on Kaggle and runs on any Python 3 environment:

```bash
pip install sentence-transformers scikit-learn pandas
jupyter notebook "Product Classification.ipynb"
```

## Limitations

- Training data is generated aliases, not real posts, and there is no held-out evaluation yet; the test sentences in the notebook are spot checks.
- Some generated product names don't exist (for example future iPhone numbers and chip/size combinations Apple doesn't sell), so the label space is larger than the real catalogue.
- Model numbers inside a sentence can be confused with nearby numbers (e.g. chip vs. screen size), as a few edge-case outputs show.
