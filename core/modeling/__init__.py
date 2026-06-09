"""Modeling: leakage-aware data prep — splitting and fit-on-train preprocessing.

Unlike the stateless transforms in ``features``, everything here learns from data and must be fit on
the training set only, so it lives behind sklearn's fit/transform.
"""
