"""
Shared rainfall building blocks used by more than one Mayim Tools
algorithm (Design Storm Ensembles, Precipitation data to DDF, Storm
Library & DDF Consistency Check, ...).

Moved here verbatim from the tools that first implemented them, so
that every tool computes the same quantity with the same code - one
implementation to test and one to defend in review. Zero QGIS/Qt
dependency, same as every core.py in this suite.

Modules
-------
csvio      - delimiter-sniffing CSV reader
ddf        - DDF table parsing, duration parsing, AEP/ARI conversion,
             DDFTable interpolation/inversion
events     - IETD storm-event catalogue and best-window extraction
patterns   - target duration table, Early/Middle/Late classification,
             normalised mass curves
rfa        - rainfall frequency analysis (AMS extraction, L-moments,
             GEV/Gumbel/GLO/LP3 fitting, ratio-diagram diagnostic)
"""
