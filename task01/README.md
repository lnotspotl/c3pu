## Task 01

### Submit jobs

```bash
export CACHE_CONDA_ENV_PATH=<path-to-conda-env>
export CACHE_TASK_PATH=<path-to-task> # path to task01 folder
python3 submit_jobs.py --output_folder=<temp>
python3 combine_results.py --result_folder=<temp> --output_csv="results.csv"
```

### Results

The following table summarizes the `mpki` and `hit_rate` for *Belady*'s algorithm when tested on different traces.
Highlighted are members of three different trace families with the highest `mpki` value.

| trace           | mpki       | hit_rate |
|-----------------|------------|----------|
| astar_23B       | `1.975975`   | `0.818419` |
| astar_163B      | `1.444039`   | `0.432142` |
| **astar_313B**      | `27.901723`  | `0.382665` |
| bwaves_98B      | `0.282095`   | `0.113087` |
| bwaves_1609B    | `21.478848`  | `0.058776` |
| bwaves_1861B    | `21.472755`  | `0.059042` |
| bzip2_183B      | `0.836222`   | `0.851526` |
| bzip2_259B      | `0.883809`   | `0.837782` |
| bzip2_281B      | `0.003719`   | `0.936131` |
| gamess_196B     | `0.000000`   | `1.000000` |
| gamess_247B     | `0.000000`   | `1.000000` |
| gamess_316B     | `0.000000`   | `1.000000` |
| gcc_13B         | `6.729660`   | `0.550900` |
| gcc_39B         | `0.831830`   | `0.319267` |
| gcc_56B         | `1.000391`   | `0.760514` |
| leslie3d_94B    | `21.003264`  | `0.268859` |
| mcf_46B         | `49.163250`  | `0.399631` |
| mcf_158B        | `58.182457`  | `0.452240` |
| **mcf_250B**        | `65.969793`  | `0.467189` |
| milc_360B       | `22.735326`  | `0.051737` |
| **milc_409B**       | `25.195681`  | `0.065400` |
| milc_744B       | `18.362450`  | `0.029212` |
| namd_400B       | `0.032350`   | `0.739920` |
| namd_851B       | `0.016404`   | `0.865892` |
| namd_1907B      | `0.014253`   | `0.864859` |
| omnetpp_4B      | `2.022131`   | `0.926423` |
| omnetpp_17B     | `4.047157`   | `0.904279` |
| omnetpp_340B    | `12.631815`  | `0.419794` |
| perlbench_53B   | `0.068037`   | `0.962192` |
| perlbench_105B  | `0.062801`   | `0.973014` |
| perlbench_135B  | `0.069533`   | `0.968046` |
| povray_250B     | `0.000000`   | `1.000000` |
| povray_437B     | `0.000000`   | `1.000000` |
| povray_711B     | `0.000000`   | `1.000000` |
| sjeng_358B      | `0.274100`   | `0.250394` |
| sjeng_1109B     | `0.261058`   | `0.298648` |
| sjeng_1966B     | `0.270180`   | `0.292009` |
| sphinx3_883B    | `6.459224`   | `0.541340` |
| sphinx3_1339B   | `4.240573`   | `0.690700` |
| sphinx3_2520B   | `3.296253`   | `0.746955` |
