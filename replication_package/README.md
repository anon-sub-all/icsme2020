### Replication package of RESICO tool

#### Previous requirements
Experiments were executed on Python 3.7.x, this version or a more updated version of Python is recommended.
External libraries used in the experiments are included in the `requirements.txt` file.
To install them simply execute the following command on a terminal within the root folder:
```
pip3 install -r requirements.txt
```

#### Replicating results
To replicate the results of the experiments, execute the following:
```
python3 run_scripts.py
```

Note: Out of the two trained models, only Decision Trees (DT) is saved on disk for the Qualitas evaluation.
Due to the size Random Forest (RF) requires (more than 100 GB), this model is not saved on disk.
We provided the code in case you also wanted to save this model.
Results reported on the paper for Qualitas corpus were calculated using the saved version of DT and RF.

To get the statistics in RQ1, execute the scripts in the`RScripts/` folder.
