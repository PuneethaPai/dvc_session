Intent:
---
- This repo is an introductory project to understand the creation of ML models using DVC pipelines
- With [DagsHub](https://dagshub.com/) user can visualise and compare their experiments

## Setup:
1. Clone the repo:
2. Initialise dvc 
3. Configure dvc remote
4. Version data file with dvc
5. You are good to go. No define pipelines and build your first ML model and pipeline. Enjoy the reproducibility of experiments.
 
## Defining Pipelines
Idea is to build multiple model and ensemble them to create a stable strong model.

Your final pipeline structure will look like this in the end:
```bash
                                                  +-------------------+
                                                  | data/iris.csv.dvc |
                                                  +-------------------+
                                                            *
                                                            *
                                                            *
                                                      +-----------+
                                                      | split.dvc |
                                                      +-----------+
                                                            *
                                                            *
                                                            *
                                                    +---------------+
                                                ****| featurize.dvc |****
                                        ********    +---------------+    ********
                                ********           **              ***           *********
                        ********                ***                   **                  ********
                   *****                      **                        **                        ********
+--------------------+             +---------------+             +-------------------+                    *****
| train_logistic.dvc |**           | train_svc.dvc |             | train_forrest.dvc |            ********
+--------------------+  ********   +---------------+             +-------------------+    ********
                                ********           **              ***           *********
                                        ********     ***         **      ********
                                                *****   **     **   *****
                                                 +--------------------+
                                                 | train_ensemble.dvc |
                                                 +--------------------+
```
Order is `data -> train_test_split -> feature_extraction -> 3 models -> ensemble_model` and **Done!!**

With `data/iris.csv` versioned in DVC you can start with your first pipeline.

Define `Stage1` => `split.dvc` i.e train_test_split.
```bash
dvc run -f split.dvc\
 -d data/iris.csv\
 -d src/train_test_split.py\
 -o data/split\
 python src/train_test_split.py -i "data/iris.csv" -o "data/split/"
```

Define `Stage2` => `featurize.dvc` i.e Feature Engineering.
```bash
dvc run -f featurize.dvc\
 -d data/split\
 -d src/feature_engineering.py\
 -p pca\
 -o data/features\
 -o data/models/pca/model.gz\
 -O data/models/pca/params.yml\
 -M data/models/pca/metrics.csv\
 python src/feature_engineering.py -i "data/split/" -o "data/features/" -o "data/models/pca/"
```

Define `Stage3.a` => `train_logistic.dvc` i.e Fit Logistic Regression Model.
```bash
dvc run -f train_logistic.dvc\
 -d src/logistic_regression.py\
 -d data/features\
 -p logistic\
 -o data/models/logistic/model.gz\
 -O data/models/logistic/params.yml\
 -M data/models/logistic/metrics.csv\
 python src/logistic_regression.py -i "data/features/" -o "data/models/logistic/"
```

Define `Stage3.b` => `train_svc.dvc` i.e Fit Linear SVC Model.
```bash
dvc run -f train_svc.dvc\
 -d src/linear_svc.py\
 -d data/features\
 -p svc\
 -o data/models/svc/model.gz\
 -O data/models/svc/params.yml\
 -M data/models/svc/metrics.csv\
 python src/linear_svc.py -i "data/features/" -o "data/models/svc/"
```

Define `Stage3.c` => `train_forrest.dvc` i.e Fit Random Forrest Model.
```bash
dvc run -f train_forrest.dvc\
 -d src/random_forrest.py\
 -d data/features\
 -p forrest\
 -o data/models/r_forrest/model.gz\
 -O data/models/r_forrest/params.yml\
 -M data/models/r_forrest/metrics.csv\
 python src/random_forrest.py -i "data/features/" -o "data/models/r_forrest/"
```

Define `Stage4` => `train_ensemble.dvc` i.e Create an Ensemble Model.
```bash
dvc run -f train_ensemble.dvc\
 -d src/ensemble.py\
 -d data/features\
 -d data/models/logistic/model.gz\
 -d data/models/svc/model.gz\
 -d data/models/r_forrest/model.gz\
 -p ensemble\
 -o data/models/ensemble/model.gz\
 -O data/models/ensemble/params.yml\
 -M data/models/ensemble/metrics.csv\
 python src/ensemble.py -i "data/features/" -m "data/models/" -o "data/models/ensemble/"
```