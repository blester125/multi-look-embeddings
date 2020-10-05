# Multi Look Embeddings

Code used for experiments from [Multiple Word Embeddings for Increased Diversity of Representation](https://arxiv.org/abs/2009.14394)

Some scripts, utils, and [mead-baseline](https://github.com/dpressel/mead-baseline) configs for probing *why* multiple embeddings work.

Scripts should be run from the top level of this repo with `python -m scripts.{script_name}`

Train mead-baseline models with `mead-train --config configs/sst2.json`

# Citation

If you use this multiple embedding technique or any of these analysis tools in your research please cite the following:

```BibTex
@article{lester2020multiple,
  title={Multiple Word Embeddings for Increased Diversity of Representation},
  author={Lester, Brian and Pressel, Daniel and Hemmeter, Amy and Choudhury, Sagnik Ray and Bangalore, Srinivas},
  journal={arXiv preprint arXiv:2009.14394},
  year={2020}
}
```
