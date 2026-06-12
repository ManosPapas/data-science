# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 19 · Cross-sell analytics — basket rules, recommenders, and the co-purchase network
#
# Three views of the same commercial question — *what should we put in front of this customer?*
# Association rules find what sells together (and separate real affinity from shared popularity
# with **lift**); an item-item recommender personalizes it; ranking metrics on held-out purchases
# decide whether it beats the only honest baseline (popularity); and the co-purchase **network**
# shows the catalogue's structure — hubs, communities, and the cross-sell dead ends.

# %%
from core.prelude import *

set_theme()
rng = np.random.default_rng(42)

# %% [markdown]
# ## 1. Synthetic transactions with real structure
# 3,000 baskets over a 12-product catalogue with two affinity clusters (coffee gear, tea gear)
# and one universally popular item (gift card) — the classic popularity confounder that
# confidence alone falls for.

# %%
coffee = ["espresso_machine", "grinder", "beans", "milk_frother"]
tea = ["teapot", "kettle", "loose_tea", "infuser"]
other = ["gift_card", "mug", "scale", "descaler"]
catalog_items = coffee + tea + other

rows = []
for basket_id in range(3000):
    if rng.random() < 0.45:  # coffee-leaning basket
        pool, base_p = coffee, 0.55
    elif rng.random() < 0.6:  # tea-leaning basket
        pool, base_p = tea, 0.55
    else:
        pool, base_p = catalog_items, 0.15
    for item in pool:
        if rng.random() < base_p:
            rows.append({"order": basket_id, "item": item})
    if rng.random() < 0.30:  # gift cards land in every kind of basket
        rows.append({"order": basket_id, "item": "gift_card"})
    if rng.random() < 0.25:
        rows.append({"order": basket_id, "item": rng.choice(other[1:])})
orders = pl.DataFrame(rows).unique()
print(f"{orders['order'].n_unique():,} baskets, {orders.height:,} line items")

# %% [markdown]
# ## 2. Frequent itemsets & association rules — confidence vs lift
# Support says how common a combination is; confidence is the cross-sell hit rate; **lift** is
# the one that separates real affinity from popularity. The gift card co-occurs with everything
# (decent confidence) but lifts nothing; grinder→beans lifts hard. Recommend on lift, report
# on confidence.

# %%
itemsets = basket.frequent_itemsets(orders, transaction="order", item="item", min_support=0.03)
itemsets.filter(pl.col("size") == 2).head(8)

# %%
rules = basket.association_rules(
    orders, transaction="order", item="item", min_support=0.03, min_confidence=0.15
)
print(rules.head(8))
gift_rules = rules.filter(pl.col("antecedent") == "gift_card")
if gift_rules.height:
    print(
        f"gift_card rules: max confidence {gift_rules['confidence'].max():.0%} "
        f"but max lift {gift_rules['lift'].max():.2f} -> popular, not predictive"
    )

# %% [markdown]
# ## 3. Personalize it — item-item collaborative filtering
# The recommender computes item-item cosine similarity over customers, then scores each
# customer's unseen items by what they already own. Held-out evaluation below; here, the sniff
# test: a grinder-owner should see coffee gear, not teapots.

# %%
# hold out each customer's last item (by row order) as the test target
orders_indexed = orders.with_row_index()
last_rows = orders_indexed.group_by("order").agg(pl.col("index").max().alias("index"))
test_set = orders_indexed.join(last_rows, on=["order", "index"]).select("order", "item")
train_set = orders_indexed.join(last_rows, on=["order", "index"], how="anti").select(
    "order", "item"
)

recommender = recommend.ItemItemRecommender().fit(train_set, user="order", item="item")
print(recommender.similar_items("grinder", k=4))

# demo on a basket that still has items in train (single-item baskets went whole to the holdout)
example_order = train_set.group_by("order").len().filter(pl.col("len") >= 2)["order"][0]
example_basket = train_set.filter(pl.col("order") == example_order)["item"].to_list()
print(f"\nbasket {example_order} holds {example_basket}")
print(recommender.recommend(example_order, k=3))

# %% [markdown]
# ## 4. Does it beat popularity? Ranking metrics on held-out purchases
# For every test basket: score all unseen items, mark the held-out purchase as the relevant one,
# and compare NDCG/precision/MRR against the popularity baseline. Popularity is genuinely hard
# to beat on aggregate data — a recommender earns its keep on the *personalized* margin.

# %%
popular = recommend.popularity_baseline(train_set, item="item", k=len(catalog_items))
popularity_scores = dict(
    zip(popular["item"].to_list(), popular["interactions"].to_list(), strict=True)
)

eligible = [
    order_id
    for order_id in test_set["order"].to_list()
    if train_set.filter(pl.col("order") == order_id).height > 0
]
sampled = rng.choice(np.array(eligible), size=min(400, len(eligible)), replace=False)

relevance_rows, model_rows, baseline_rows = [], [], []
for order_id in sampled:
    target = test_set.filter(pl.col("order") == order_id)["item"][0]
    seen = set(train_set.filter(pl.col("order") == order_id)["item"].to_list())
    candidates = [item for item in catalog_items if item not in seen]
    if target not in candidates:
        continue
    picks = recommender.recommend(order_id, k=len(candidates))
    model_scores = dict(zip(picks["item"].to_list(), picks["score"].to_list(), strict=True))
    relevance_rows.append([1.0 if item == target else 0.0 for item in candidates])
    model_rows.append([model_scores.get(item, 0.0) for item in candidates])
    baseline_rows.append([popularity_scores.get(item, 0.0) for item in candidates])

# ragged candidate lists -> pad to equal width with irrelevant zero-score slots
width = max(len(row) for row in relevance_rows)


def pad(matrix: list[list[float]]) -> np.ndarray:
    return np.array([row + [0.0] * (width - len(row)) for row in matrix])


for name, scored in [("item-item", pad(model_rows)), ("popularity", pad(baseline_rows))]:
    metrics = evaluate.ranking_metrics(pad(relevance_rows), scored, k=3)
    print(f"{name:10s} ndcg@3 {metrics['ndcg@3']:.3f}  mrr {metrics['mrr']:.3f}")

# %% [markdown]
# ## 5. The co-purchase network — structure of the catalogue
# Lift-filtered rules become a graph: nodes are products, edges are real affinities. PageRank
# finds the structurally central products (good anchor promotions), connected components reveal
# the merchandising communities, and isolated products are cross-sell dead ends that need a
# bridge offer.

# %%
edges = rules.filter(pl.col("lift") > 1.1).select(
    pl.col("antecedent").alias("source"),
    pl.col("consequent").alias("target"),
    pl.col("lift").alias("weight"),
)
print(graph.pagerank(edges, weight="weight").head(5))

communities = graph.connected_components(edges)
print(communities.group_by("component").agg(pl.col("node").sort()))

hubs = graph.degree_centrality(edges, weight="weight")
print(f"hub product: {hubs['node'][0]} (degree {hubs['degree'][0]:.0f})")

# %%
# The same structure, drawn: hubs central, the two communities visibly apart.
network.network(
    edges,
    weight="weight",
    seed=42,
    title="The co-purchase network — two product worlds, bridged by accessories",
)

# %% [markdown]
# **Takeaways:** lift, not confidence, is the cross-sell signal — the gift card reaches ~30% of
# baskets and still predicts nothing (lift ≈ 1), while grinder→beans is a genuine affinity; the
# item-item recommender recovers the two product worlds from co-purchases alone and beats the
# popularity baseline on held-out purchases (higher NDCG@3/MRR), which is the bar any
# recommender must clear before it earns shelf space; and the network view turns the rule list
# into merchandising structure — coffee and tea form separate communities bridged only by
# accessories, PageRank surfaces the anchor products worth promoting, and anything isolated in
# the graph won't be cross-sold without deliberately engineering a bridge (bundles, placement).
