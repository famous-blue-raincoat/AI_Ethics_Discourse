# [iConference 2026] AI_Ethics_Discourse

We provide the code for topic analysis and visualization of AI ethics discourse in OpenAI. Please note that we can only share the code for topic modeling and visualization due to data privacy constraints.

## Usage


To run with PDF data (default):  `python topic_analysis.py --mode pdf --data_dir ./my_texts`

To run with JSON articles:  `python topic_analysis.py --mode article --json_path ./data.json`

To tune clustering tightness:  `python topic_analysis.py --min_cluster_size 10 --min_dist 0.01`




## Citation

Please cite our iConference 2026 paper if you find this code useful in your research:

```bibtex
@misc{wilfley2026competingvisionsethicalai,
      title={Competing Visions of Ethical AI: A Case Study of OpenAI}, 
      author={Melissa Wilfley and Mengting Ai and Madelyn Rose Sanfilippo},
      year={2026},
      eprint={2601.16513},
      archivePrefix={arXiv},
      primaryClass={cs.CY},
      url={https://arxiv.org/abs/2601.16513}, 
}
```

Feel free to reach out or open an issue if you have any questions or feedback! Thank you for your interest in our work!