# FeatureConverter in SeqIO vs AirIO

This doc outlines the issues with the existing FeatureConverter design in SeqIO
and describes the rationale for and details of new design in AirIO. AirIO
provides helpers that return a list of preprocessors corresponding to popular
SeqIO FeatureConverters in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/feature_converters.py.

## Issues

With SeqIO, it is common for users to configure a Task to perform model-agnostic
preprocessing (e.g. tokenization) and a FeatureConverter to perform
model-specific preprocessing. Over time, SeqIO FeatureConverters have become
monolithic objects obscuring the underlying preprocessing and have also grown to
encompass other preprocessing steps like packing, etc. that don’t belong in a
“feature converter”.

Here are the issues with FeatureConverters in more detail:

+   Feature Converters have become bloated over time, including preprocessing
    steps like packing, trimming, padding, etc. It’s non-trivial for users to
    understand what a feature converter does and modify its behavior. It’s also
    hard to modify a given FeatureConverter because it may break other use
    cases.

+   The pattern of inheriting existing Feature Converter and modifying its
    behavior has not scaled well. For instance the DecoderFeatureConverter in
    SeqIO inherits and overrides the PrefixLMFeatureConverter in SeqIO which
    inherits and overrides the LMFeatureConverter in SeqIO which inherits and
    overrides the FeatureConverter in SeqIO. This has made working with Feature
    Converters very hard.

+   Including packing in feature converters has made them particularly hard to
    use. If one needs to pack before mixing, packing (and hence the
    FeatureConverter) must be included as a Task/Mixture preprocessor, thus
    defeating the original purpose of decoupling Tasks and Feature Converters.

## Feature Converters in AirIO

AirIO replaces FeatureConverters with runtime preprocessors. It allows users to
pass a list of preprocessors to the `get_dataset` call, which are applied to the
dataset in addition to the Task preprocessors. This allows configuring
preprocessors that can vary over runs, e.g. train-specific preprocessors like
packing, trimming, padding, feature conversion, etc. and eval-specific
preprocessors like few-shot prompting, etc.

`runtime_preprocessors` can be passed to `airio.get_dataset` instead of a
`feature_converter`. Lists corresponding to common FeatureConverters are
provided in
https://github.com/google/airio/blob/main/airio/_src/pygrain/common/feature_converters.py.
for convenience. For instance, `feature_converter =
EncDecFeatureConverter(seq_lens, pack=True)`, would be replaced by:
`runtime_preprocessors = [trim, pack, pad, convert_to_t5x_encdec]`. This
preserves the ability to pass model-specific and other preprocessors at runtime
while making the interface simpler, easily modifiable and eliminating the
bloated FeatureConverter classes. The naming convention also clarifies where the
preprocessors are applied in the pipeline.
