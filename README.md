# UDBENCHMARK

**TODO**: general description

## Install

```term
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

pip install -r src/pi_FPE/requirements.txt
pip install -e src/pi_FPE

pip install -r src/pi_FPM/requirements.txt
pip install -e src/pi_FPM

pip install -r src/pi_LDE/requirements.txt
pip install -e src/pi_LDE

pip install -r src/pi_SpatTemp/requirements.txt
pip install -e src/pi_SpatTemp

```

## Usage

```term
run_fpe tests/data/input/subject_04_jointTrajectories_01.csv tests/data/input/subject_04_comTrajectories_01.csv tests/data/input/subject_04_angMomentum_01.csv tests/data/input/subject_04_comItensor_01.csv tests/data/input/subject_04_gaitEvents_01 tests/data/input/subject_04_testbedLabel_01.yaml outfolder
```

```term
run_fpm tests/data/input/subject_04_jointTrajectories_01.csv tests/data/input/subject_04_comTrajectories_01.csv tests/data/input/subject_04_gaitEvents_01 outfolder
```

```term
run_spattemp tests/data/input/subject_04_jointTrajectories_01.csv tests/data/input/subject_04_gaitEvents_01 outfolder
```

```term
run_lde tests/data/input/subject_04_comTrajectories_01.csv tests/data/input/subject_04_gaitEvents_01 outfolder
```
