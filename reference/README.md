# Published values used for comparison

Mordatch and Abbeel, *Emergence of Grounded Compositional Language in Multi-Agent Populations*, arXiv:1703.04908v2. The supplied PDF has SHA-256 `2f1d35067ac0a0feec5b9203a6a8c4caa7354156b940135e7c1eafeef853364c`. The document's sixth page contains Table 1 and Figures 4–5. The seventh page contains Figures 6–7.

| Table 1 condition | Train physical reward | Test physical reward |
| --- | ---: | ---: |
| No communication | −0.919 | −0.920 |
| Communication | −0.332 | −0.392 |
| Difference | +0.587 | +0.528 |

`figure6_vertices.csv` contains vertices of the three **published** paths. To extract them, convert page 7 with `pdftocairo -f 7 -l 7 -svg 1703.04908v2.pdf page7.svg` and read the long, red, blue and purple SVG paths in legend order. In the paths' own coordinates, x=44.541052 and x=490.941970 map to iterations 0 and 5,000. Values y=41.496931 and y=387.098209 map to counts 0 and 20. The CSV rounds to the nearest iteration and integer symbol count. The PDF plotter simplifies runs of constant values. The path endpoints correspond to iteration 4,998. Their final counts are 3, 6 and 9, respectively. The authors did not specify the frequency cutoff used to define an active word.

Figure 5 aggregates streams from 10,000 test trials and shows six time indices. The paper gives examples and directional findings for syntax, novel configurations and nonverbal strategies. It gives no trial-level observations, uncertainty intervals or numerical effect sizes for those claims. Figures 1–3 illustrate the environment, dynamics and policy, respectively.
