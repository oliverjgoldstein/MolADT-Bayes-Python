# ZINC Timing

- `smiles_csv_to_string`: Read matched SMILES CSV rows into Python strings without chemistry parsing. 405387.7 mol/s with 0 failures.
- `moladt_csv_to_moladt`: Read cached MolADT CSV rows and decode the embedded MolADT payload into the local typed Molecule object. 772.4 mol/s with 0 failures.
- `smiles_to_json`: Parse matched SMILES strings into MolADT and serialize the result to JSON payloads. 836.0 mol/s with 0 failures.
- `sdf_to_moladt`: Read cached SDF files and parse them into the local typed Molecule object. 1120.2 mol/s with 0 failures.
- `sdf_to_smiles`: Read cached SDF files, validate the decoded molecules, and render them into the supported SMILES subset. 300.5 mol/s with 11 failures.
- `moladt_to_json`: Serialize parsed MolADT objects to JSON and write the JSON files used by the final decode stage. 1514.2 mol/s with 0 failures.
- `json_to_moladt`: Read JSON files and decode them back into the local typed Molecule object. 603.9 mol/s with 0 failures.
- `json_to_smiles`: Read JSON files, decode them into Molecule values, validate them, and render the supported SMILES subset. 295.0 mol/s with 11 failures.

Detailed per-item timings: `results/timing/paper/run_20260417_162547/details/zinc_timing_items.csv`

## Slowest Timed Items

### moladt_csv_to_moladt

- `ZINC000184874069` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/moladt_csv_library.csv:226299`: 44025.1 us, success=True.
- `ZINC000269174551` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/moladt_csv_library.csv:241525`: 43502.0 us, success=True.
- `ZINC000755026260` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/moladt_csv_library.csv:225073`: 40390.1 us, success=True.
- `ZINC001019201610` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/moladt_csv_library.csv:246750`: 39552.4 us, success=True.
- `ZINC001755342695` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/moladt_csv_library.csv:247175`: 39353.7 us, success=True.

### smiles_to_json

- `ZINC000982955090` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/smiles_library.csv:247479`: 52411.7 us, success=True.
- `ZINC001104830777` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/smiles_library.csv:242638`: 52385.6 us, success=True.
- `ZINC001099378542` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/smiles_library.csv:239074`: 52071.3 us, success=True.
- `ZINC000572617057` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/smiles_library.csv:248687`: 51348.0 us, success=True.
- `ZINC000555973366` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/smiles_library.csv:240265`: 50815.1 us, success=True.

### sdf_to_moladt

- `ZINC000652107285` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC000652107285.sdf`: 13836377.7 us, success=True.
- `ZINC001484605085` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC001484605085.sdf`: 2716254.7 us, success=True.
- `ZINC000977157626` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC000977157626.sdf`: 2670216.8 us, success=True.
- `ZINC000339650948` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC000339650948.sdf`: 1931855.6 us, success=True.
- `ZINC001426958091` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC001426958091.sdf`: 1598210.0 us, success=True.

### sdf_to_smiles

- `ZINC001371525335` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC001371525335.sdf`: 28212423.2 us, success=True.
- `ZINC000340363290` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC000340363290.sdf`: 7514420.9 us, success=True.
- `ZINC001274003328` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC001274003328.sdf`: 5191759.5 us, success=True.
- `ZINC001056601838` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC001056601838.sdf`: 3027429.5 us, success=True.
- `ZINC000970720724` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/sdf_library/ZINC000970720724.sdf`: 2686737.6 us, success=True.

### moladt_to_json

- `ZINC001098029726` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001098029726.moladt.json`: 168531.5 us, success=True.
- `ZINC000614399207` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC000614399207.moladt.json`: 149755.7 us, success=True.
- `ZINC000944028911` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC000944028911.moladt.json`: 92365.2 us, success=True.
- `ZINC001088328580` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001088328580.moladt.json`: 75130.5 us, success=True.
- `ZINC001114369656` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001114369656.moladt.json`: 71327.1 us, success=True.

### json_to_moladt

- `ZINC001617138383` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001617138383.moladt.json`: 5147422.6 us, success=True.
- `ZINC000986868682` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC000986868682.moladt.json`: 3864716.0 us, success=True.
- `ZINC001447134432` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001447134432.moladt.json`: 3702776.6 us, success=True.
- `ZINC000939793566` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC000939793566.moladt.json`: 3702239.5 us, success=True.
- `ZINC000965560633` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC000965560633.moladt.json`: 3687697.5 us, success=True.

### json_to_smiles

- `ZINC000728370438` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC000728370438.moladt.json`: 9646984.8 us, success=True.
- `ZINC001077287966` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001077287966.moladt.json`: 9571087.7 us, success=True.
- `ZINC001332907377` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001332907377.moladt.json`: 9542496.4 us, success=True.
- `ZINC001061952575` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001061952575.moladt.json`: 9462293.0 us, success=True.
- `ZINC001233847403` `/Users/Oliver/Documents/Computer Science/Year 14/MolADT-Bayes-Python/data/processed/zinc_timing/zinc15_250K_2D/full/json_library/ZINC001233847403.moladt.json`: 9351849.0 us, success=True.

Matched timing-library manifest: `results/timing/paper/run_20260417_162547/details/zinc_timing_corpus_manifest.csv`
Result-file index: `results/timing/paper/run_20260417_162547/timing_result_files.txt`
