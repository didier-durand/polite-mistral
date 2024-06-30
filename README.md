# polite-mistral

[![build workflow](https://github.com/didier-durand/qstensils/actions/workflows/build.yml/badge.svg)](https://github.com/didier-durand/polite-mistral/actions)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/1c826b70f5dd4b45b350c0337f75075d)](https://app.codacy.com/gh/didier-durand/polite-mistral/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![codecov](https://codecov.io/github/didier-durand/polite-mistral/graph/badge.svg?token=62HAxyOHom)](https://codecov.io/github/didier-durand/polite-mistral)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python code for a solution to cross-check, validate and gauge an LLM (potentially fine-tuned) answer with answers of other
LLMs to same question. The workflow leads to a ranking reflecting the confidence in the answer. 

The confidence is computed via standard NLP metrics (Euclidean Distance, Cosine Similarity) with 
one of such metrics produced from the embeddings 
of the answers produced by each involved embedding engine (EE).

For example, with 3 EEs and 2 LLMs, the Cosine Similarity is computed between the 2 LLM answers 
with their embeddings from each EE. It produces a triplet (x,y,z) of coordinates.

The confidence is the 1 - (distance((1,1,1),(x,yz)) / 2sqrt(3)) as this distance is the normalized
distance to the ideal value (1,1,1) for the triplet.
